// ==========================================================================
// Segmentation Workshop Explorer - Interactive JavaScript
// Red Apples Single-Color Dataset (CIELAB a*-channel only)
// ==========================================================================

// Real project dataset numbers (Red Apples on Textured Surfaces with CIELAB a*)
const DATA = {
  before: {
    threshold: 0.11,
    TN: 1722896,
    FP: 3288,
    FN: 61531,
    TP: 593612,
    acc: 0.9728,
    prec: 0.9945,
    rec: 0.9061,
    iou: 0.9016,
    dice: 0.9482
  },
  after: {
    threshold: 0.05,
    TN: 1715839,
    FP: 10345,
    FN: 54491,
    TP: 600652,
    acc: 0.9728,
    prec: 0.9831,
    rec: 0.9168,
    iou: 0.9026,
    dice: 0.9488
  }
};

const SCORE_DISTRIBUTION = window.SCORE_DISTRIBUTION_DATA || null;

const CELL_DETAILS = {
  TN: {
    tag: "TN: True Negative (ลบแท้)",
    status: "ทำนายถูกต้อง (Correct)",
    isOk: true,
    title: "พิกเซลพื้นผิว (โต๊ะไม้/หิน) ที่ไม่มีสีแดง ทายถูกว่าเป็นพื้นหลัง",
    desc: "พิกเซลเหล่านี้คือเนื้อไม้และเคาน์เตอร์หินอ่อนรอบผลแอปเปิล ซึ่งมีค่า a* ต่ำกว่าเกณฑ์ความแดง ระบบจึงระบาย Mask เป็นสีดำ (0) ได้อย่างแม่นยำกว่า 1.71 ล้านพิกเซล",
    gtText: "Ground Truth: ดำ (0/พื้นผิวโต๊ะ)",
    predText: "Predicted: ดำ (0/พื้นผิวโต๊ะ)",
    impact: "ช่วยรักษาค่า Accuracy รวมให้อยู่ในระดับสูงถึง 97.28% เนื่องจากพื้นที่พื้นผิวโต๊ะมีสัดส่วนใหญ่สุดในภาพ"
  },
  FP: {
    tag: "FP: False Positive (บวกเท็จ)",
    status: "ทำนายผิดพลาด (Error)",
    isOk: false,
    title: "พิกเซลพื้นผิวโต๊ะ/เงาที่มีโทนสีแดงเรื่อ ทายผิดเป็นแอปเปิล",
    desc: "พิกเซลเหล่านี้คือเสี้ยนไม้โอ๊กหรือไม้วอลนัตที่มีเฉดสีน้ำตาลแดง หรือขอบเงาตกกระทบที่มีค่า a* สูงกว่าพื้นหลังเล็กน้อย ทำให้ระบบหลงคิดว่าเป็นเนื้อแอปเปิล (ก่อนทำมี 3,288 พิกเซล และหลังทำมี 10,345 พิกเซลจากการใช้เกณฑ์ความไวสูง t=0.05)",
    gtText: "Ground Truth: ดำ (0/พื้นผิวหรือเงา)",
    predText: "Predicted: ขาว (1/ผลแอปเปิล)",
    impact: "ส่งผลกระทบต่อค่า Precision แต่ยังคงอยู่ในระดับสูงมาก (98.31% - 99.45%)"
  },
  FN: {
    tag: "FN: False Negative (ลบเท็จ)",
    status: "ทำนายผิดพลาด (Error)",
    isOk: false,
    title: "พิกเซลเนื้อแอปเปิล แต่ทายผิดเป็นพื้นหลัง (จุดสะท้อนแสง/รอยด่าง)",
    desc: "พิกเซลเหล่านี้คือจุดสะท้อนแสงสีขาวจ้า (Specular Glare) ซึ่งมีค่า a* ต่ำใกล้ 128 (สีเทาขาว) หรือผิวแอปเปิลที่มีสีด่างเหลือง ทำให้ระบบคิดว่าเป็นพื้นหลัง แต่หลังจากผ่าน Morphology พบว่า FN ลดลงจาก 61,531 เหลือ 54,491 พิกเซล (กู้คืนเนื้อแอปเปิลกลับมาได้ถึง 7,040 พิกเซล!)",
    gtText: "Ground Truth: ขาว (1/ผลแอปเปิล)",
    predText: "Predicted: ดำ (0/พื้นผิวโต๊ะ)",
    impact: "การลดลงของ FN ช่วยดันค่า Recall ให้ขยับสูงขึ้นจาก 90.61% เป็น 91.68%"
  },
  TP: {
    tag: "TP: True Positive (บวกแท้)",
    status: "ทำนายถูกต้อง (Correct)",
    isOk: true,
    title: "พิกเซลแอปเปิลสีแดง ที่ตรวจจับได้สำเร็จ",
    desc: "พิกเซลเนื้อผลแอปเปิลสีแดงจริงที่ตรวจจับสำเร็จด้วยช่อง a* จากเดิม 593,612 พิกเซล เพิ่มขึ้นเป็น 600,652 พิกเซลหลังทำ Morphology",
    gtText: "Ground Truth: ขาว (1/ผลแอปเปิล)",
    predText: "Predicted: ขาว (1/ผลแอปเปิล)",
    impact: "เป็นตัวตั้งหลักของทั้ง IoU (0.9026) และ Dice (0.9488) สะท้อนความสมบูรณ์ในการจับรูปทรงผลแอปเปิล"
  }
};

const RAW_OPERATING_POINTS = [
  {threshold: 0.00, fpr: 1.0000, tpr: 1.0000},
  {threshold: 0.02, fpr: 0.0600, tpr: 0.9300},
  {threshold: 0.05, fpr: 0.0160, tpr: 0.9200},
  {threshold: 0.11, fpr: 0.0019, tpr: 0.9061},
  {threshold: 0.25, fpr: 0.0003, tpr: 0.8280},
  {threshold: 0.50, fpr: 0.00005, tpr: 0.6500},
  {threshold: 0.75, fpr: 0.0000, tpr: 0.2000},
  {threshold: 1.00, fpr: 0.0000, tpr: 0.0000}
];

const MORPH_OPERATING_POINTS = [
  {threshold: 0.00, fpr: 1.0000, tpr: 1.0000},
  {threshold: 0.02, fpr: 0.0250, tpr: 0.9290},
  {threshold: 0.05, fpr: 0.0060, tpr: 0.9168},
  {threshold: 0.11, fpr: 0.0010, tpr: 0.8990},
  {threshold: 0.25, fpr: 0.0002, tpr: 0.8260},
  {threshold: 0.50, fpr: 0.00003, tpr: 0.6400},
  {threshold: 0.75, fpr: 0.0000, tpr: 0.1900},
  {threshold: 1.00, fpr: 0.0000, tpr: 0.0000}
];

let currentStage = 'before';
let currentCell = 'TN';

// Visibility state for ROC elements
const rocVisibility = {
  l1: true,
  l2: true,
  bestL1: true,
  bestL2: true
};

function toggleRocItem(key) {
  rocVisibility[key] = !rocVisibility[key];
  const btnId = key === 'bestL1' ? 'toggle-best-l1' : key === 'bestL2' ? 'toggle-best-l2' : 'toggle-' + key;
  const btn = document.getElementById(btnId);
  if (btn) {
    btn.classList.toggle('active', rocVisibility[key]);
    btn.setAttribute('aria-checked', rocVisibility[key] ? 'true' : 'false');
  }
  applyRocVisibility();
}

function toggleRocLine(key) {
  if (key === 'best') {
    const nextState = !(rocVisibility.bestL1 && rocVisibility.bestL2);
    rocVisibility.bestL1 = nextState;
    rocVisibility.bestL2 = nextState;
    ['bestL1', 'bestL2'].forEach(k => {
      const btn = document.getElementById(k === 'bestL1' ? 'toggle-best-l1' : 'toggle-best-l2');
      if (btn) {
        btn.classList.toggle('active', nextState);
        btn.setAttribute('aria-checked', nextState ? 'true' : 'false');
      }
    });
    applyRocVisibility();
  } else {
    toggleRocItem(key);
  }
}

function applyRocVisibility() {
  // Line 1 elements
  const l1El = document.getElementById('dynamic-guide-l1');
  if (l1El) l1El.style.display = rocVisibility.l1 ? '' : 'none';

  // Line 2 elements
  const l2El = document.getElementById('dynamic-guide-l2');
  if (l2El) l2El.style.display = rocVisibility.l2 ? '' : 'none';

  // Best Dot Line 1 group
  const bestL1El = document.getElementById('best-group-l1');
  if (bestL1El) bestL1El.style.display = rocVisibility.bestL1 ? '' : 'none';

  // Best Dot Line 2 group
  const bestL2El = document.getElementById('best-group-l2');
  if (bestL2El) bestL2El.style.display = rocVisibility.bestL2 ? '' : 'none';

  // Summary box stat rows
  const statL1 = document.getElementById('stat-row-l1');
  if (statL1) statL1.style.opacity = rocVisibility.l1 ? '1' : '0.35';

  const statL2 = document.getElementById('stat-row-l2');
  if (statL2) statL2.style.opacity = rocVisibility.l2 ? '1' : '0.35';
}

function buildSmoothRocPath(points, toX, toY) {
  if (!points.length) return '';
  if (points.length === 1) return `M ${toX(points[0].fpr)} ${toY(points[0].tpr)}`;

  let path = `M ${toX(points[0].fpr)} ${toY(points[0].tpr)}`;
  for (let i = 1; i < points.length - 1; i++) {
    const current = points[i];
    const next = points[i + 1];
    const midX = (toX(current.fpr) + toX(next.fpr)) / 2;
    const midY = (toY(current.tpr) + toY(next.tpr)) / 2;
    path += ` Q ${toX(current.fpr)} ${toY(current.tpr)} ${midX} ${midY}`;
  }

  const last = points[points.length - 1];
  path += ` L ${toX(last.fpr)} ${toY(last.tpr)}`;
  return path;
}

function initRocCurves() {
  const toX = fpr => 50 + fpr * 420;
  const toY = tpr => 360 - tpr * 320;

  const rawDescending = [...RAW_OPERATING_POINTS].sort((a, b) => b.threshold - a.threshold);
  const rawTrail = [{fpr: 0, tpr: 0}];
  rawDescending.forEach(p => {
    if (p.fpr > 0 || p.tpr > 0) rawTrail.push({fpr: p.fpr, tpr: p.tpr});
  });

  const morphDescending = [...MORPH_OPERATING_POINTS].sort((a, b) => b.threshold - a.threshold);
  const morphTrail = [{fpr: 0, tpr: 0}];
  morphDescending.forEach(p => {
    if (p.fpr > 0 || p.tpr > 0) morphTrail.push({fpr: p.fpr, tpr: p.tpr});
  });

  const guide1 = document.getElementById('dynamic-guide-l1');
  if (guide1) guide1.setAttribute('d', buildSmoothRocPath(rawTrail, toX, toY));

  const guide2 = document.getElementById('dynamic-guide-l2');
  if (guide2) guide2.setAttribute('d', buildSmoothRocPath(morphTrail, toX, toY));

  applyRocVisibility();
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  setStage('before');
  selectCell('TN');
  updateGraySlider(128);
  initRocCurves();
  if (SCORE_DISTRIBUTION) {
    document.getElementById('distribution-pixel-count').textContent = (
      SCORE_DISTRIBUTION.background_total + SCORE_DISTRIBUTION.foreground_total
    ).toLocaleString();
    document.getElementById('distribution-bin-count').textContent = SCORE_DISTRIBUTION.bin_count.toLocaleString();
  }
  updateDistributionSlider(0.11);
});

// Switch Stage (Before vs After)
function setStage(stage) {
  currentStage = stage;
  
  // Update Buttons
  document.getElementById('btn-stage-before').classList.toggle('active', stage === 'before');
  document.getElementById('btn-stage-after').classList.toggle('active', stage === 'after');

  const d = DATA[stage];

  // Update Matrix Numbers
  animateNumber('val-tn', d.TN);
  animateNumber('val-fp', d.FP);
  animateNumber('val-fn', d.FN);
  animateNumber('val-tp', d.TP);

  // Update Metrics
  document.getElementById('m-acc').textContent = (d.acc * 100).toFixed(2) + '%';
  document.getElementById('m-prec').textContent = (d.prec * 100).toFixed(2) + '%';
  document.getElementById('m-rec').textContent = (d.rec * 100).toFixed(2) + '%';
  document.getElementById('m-iou').textContent = d.iou.toFixed(4);
  document.getElementById('m-dice').textContent = d.dice.toFixed(4);
}

// Number animation
function animateNumber(elementId, targetNumber) {
  const el = document.getElementById(elementId);
  el.textContent = Number(targetNumber).toLocaleString();
}

// Select Matrix Cell (TN, FP, FN, TP)
function selectCell(cellKey) {
  currentCell = cellKey;

  // Clear selections
  ['cell-tn', 'cell-fp', 'cell-fn', 'cell-tp'].forEach(id => {
    document.getElementById(id).classList.remove('selected');
  });

  const activeEl = document.getElementById('cell-' + cellKey.toLowerCase());
  if (activeEl) activeEl.classList.add('selected');

  const info = CELL_DETAILS[cellKey];
  if (!info) return;

  document.getElementById('detail-tag').textContent = info.tag;
  
  const statusEl = document.getElementById('detail-status');
  statusEl.textContent = info.status;
  statusEl.className = 'detail-status ' + (info.isOk ? 'success' : 'error');

  document.getElementById('detail-title').textContent = info.title;
  document.getElementById('detail-desc').textContent = info.desc;

  // Update visual simulation
  const vis = document.getElementById('detail-visual');
  vis.innerHTML = `
    <div class="vis-box ${cellKey === 'TN' || cellKey === 'FP' ? 'vis-black' : 'vis-white'}">
      <span>${info.gtText}</span>
    </div>
    <span class="vis-arrow">➔</span>
    <div class="vis-box ${cellKey === 'TN' || cellKey === 'FN' ? 'vis-black' : 'vis-white'}">
      <span>${info.predText}</span>
    </div>
    <span class="vis-result ${info.isOk ? 'ok' : 'fail'}">${info.isOk ? '✅ แม่นยำ' : '❌ ผิดพลาด'}</span>
  `;

  document.getElementById('detail-impact-text').textContent = info.impact;
}

// Grayscale 0-255 Slider
function updateGraySlider(val) {
  val = parseInt(val, 10);
  document.getElementById('gray-slider').value = val;
  const box = document.getElementById('pixel-preview-box');
  const text = document.getElementById('pixel-hex-text');

  box.style.backgroundColor = `rgb(${val}, ${val}, ${val})`;
  
  let label = "Mid Gray (เทา)";
  if (val === 0) label = "0: Pure Black (พื้นผิวฉากหลัง)";
  else if (val === 255) label = "255: Pure White (เนื้อผลแอปเปิล)";
  else if (val === 128) label = "128: Mid Gray (ขอบ Trimap ที่ไม่ประเมิน)";

  text.textContent = `Value: ${val} / 255 (${label})`;
  text.style.color = val > 140 ? '#000000' : '#ffffff';
}

function setGrayPreset(val) {
  updateGraySlider(val);
}

function interpolateOperatingPoint(threshold, points) {
  const t = Math.max(points[0].threshold, Math.min(points[points.length - 1].threshold, threshold));

  for (let i = 0; i < points.length - 1; i++) {
    const left = points[i];
    const right = points[i + 1];
    if (t <= right.threshold) {
      const ratio = (t - left.threshold) / (right.threshold - left.threshold);
      return {
        fpr: left.fpr + ((right.fpr - left.fpr) * ratio),
        tpr: left.tpr + ((right.tpr - left.tpr) * ratio)
      };
    }
  }

  return points[points.length - 1];
}

function buildOperatingTrail(threshold, points, toX, toY) {
  const current = interpolateOperatingPoint(threshold, points);
  const descending = [...points].sort((a, b) => b.threshold - a.threshold);
  const trail = [{fpr: 0, tpr: 0}];

  descending.forEach(point => {
    if (point.threshold > threshold) trail.push({fpr: point.fpr, tpr: point.tpr});
  });
  trail.push({fpr: current.fpr, tpr: current.tpr});

  return buildSmoothRocPath(trail, toX, toY);
}

// ROC Dual Curve Simulation Slider (Retained as safe no-op stub)
function updateRocSlider(thresholdVal) {
  // Slider removed as full curves are permanently rendered
}


// Independent distribution control: this deliberately does not update the ROC slider.
function updateDistributionSlider(thresholdVal) {
  const parsed = parseFloat(thresholdVal);
  if (!Number.isFinite(parsed)) return;

  const t = Math.max(0.00, Math.min(1.00, parsed));
  const slider = document.getElementById('distribution-slider');
  const numberInput = document.getElementById('distribution-number');
  if (slider) slider.value = t.toFixed(3);
  if (numberInput) numberInput.value = t.toFixed(3);

  if (SCORE_DISTRIBUTION) {
    const thresholdIndex = Math.max(0, Math.min(
      SCORE_DISTRIBUTION.bin_count,
      Math.round(t * SCORE_DISTRIBUTION.bin_count)
    ));
    updateDistributionChart(
      t,
      SCORE_DISTRIBUTION.fpr[thresholdIndex],
      SCORE_DISTRIBUTION.tpr[thresholdIndex]
    );
  } else {
    const rawPoint = interpolateOperatingPoint(t, RAW_OPERATING_POINTS);
    updateDistributionChart(t, rawPoint.fpr, rawPoint.tpr);
  }
}

// Draw the detailed score distribution and its four confusion regions.
function updateDistributionChart(threshold, fpr, tpr) {
  const baseY = 240;
  const minScore = 0.00;
  const maxScore = 1.00;
  const xMin = 50;
  const xMax = 720;
  const toX = score => xMin + ((score - minScore) / (maxScore - minScore)) * (xMax - xMin);
  const binCount = SCORE_DISTRIBUTION ? SCORE_DISTRIBUTION.bin_count : 1000;

  const densityAt = (values, score) => {
    if (!values || !values.length) return 0;
    const position = (score * binCount) - 0.5;
    const left = Math.max(0, Math.min(values.length - 1, Math.floor(position)));
    const right = Math.max(0, Math.min(values.length - 1, left + 1));
    const fraction = Math.max(0, Math.min(1, position - Math.floor(position)));
    return values[left] + ((values[right] - values[left]) * fraction);
  };

  const negativeDensity = score => SCORE_DISTRIBUTION
    ? densityAt(SCORE_DISTRIBUTION.background_density, score)
    : 0;
  const positiveDensity = score => SCORE_DISTRIBUTION
    ? densityAt(SCORE_DISTRIBUTION.foreground_density, score)
    : 0;
  const negativeY = score => baseY - (148 * negativeDensity(score));
  const positiveY = score => baseY - (148 * positiveDensity(score));
  const sampleCount = binCount;

  const curvePath = curve => {
    const points = [];
    for (let i = 0; i <= sampleCount; i++) {
      const score = minScore + ((maxScore - minScore) * i / sampleCount);
      points.push(`${i === 0 ? 'M' : 'L'} ${toX(score).toFixed(2)} ${curve(score).toFixed(2)}`);
    }
    return points.join(' ');
  };

  const areaPath = (curve, start, end) => {
    const safeStart = Math.max(minScore, Math.min(maxScore, start));
    const safeEnd = Math.max(minScore, Math.min(maxScore, end));
    if (safeEnd <= safeStart) return '';

    const points = [`M ${toX(safeStart).toFixed(2)} ${baseY}`];
    const samples = Math.max(2, Math.ceil((safeEnd - safeStart) / (maxScore - minScore) * sampleCount));
    for (let i = 0; i <= samples; i++) {
      const score = safeStart + ((safeEnd - safeStart) * i / samples);
      points.push(`L ${toX(score).toFixed(2)} ${curve(score).toFixed(2)}`);
    }
    points.push(`L ${toX(safeEnd).toFixed(2)} ${baseY} Z`);
    return points.join(' ');
  };

  const setPath = (id, path) => {
    const element = document.getElementById(id);
    if (element) element.setAttribute('d', path);
  };

  setPath('dist-negative-curve', curvePath(negativeY));
  setPath('dist-positive-curve', curvePath(positiveY));
  setPath('dist-tn-area', areaPath(negativeY, minScore, threshold));
  setPath('dist-fp-area', areaPath(negativeY, threshold, maxScore));
  setPath('dist-fn-area', areaPath(positiveY, minScore, threshold));
  setPath('dist-tp-area', areaPath(positiveY, threshold, maxScore));
  setPath('dist-fp-hatch', areaPath(negativeY, threshold, maxScore));
  setPath('dist-fn-hatch', areaPath(positiveY, minScore, threshold));

  const thresholdX = toX(threshold);
  const thresholdLine = document.getElementById('distribution-threshold-line');
  if (thresholdLine) {
    thresholdLine.setAttribute('x1', thresholdX);
    thresholdLine.setAttribute('x2', thresholdX);
  }

  const handle = document.getElementById('distribution-threshold-handle');
  if (handle) {
    handle.setAttribute('d', `M${thresholdX - 10} 34 L${thresholdX + 10} 34 L${thresholdX} 46 Z`);
  }

  const thresholdLabel = document.getElementById('distribution-threshold-label');
  if (thresholdLabel) {
    thresholdLabel.setAttribute('x', Math.max(105, Math.min(665, thresholdX)));
    thresholdLabel.textContent = `Threshold = ${threshold.toFixed(3)}`;
  }

  const regionLabels = {
    tn: document.getElementById('dist-tn-label'),
    fp: document.getElementById('dist-fp-label'),
    fn: document.getElementById('dist-fn-label'),
    tp: document.getElementById('dist-tp-label')
  };
  if (regionLabels.tn) regionLabels.tn.setAttribute('x', (xMin + thresholdX) / 2);
  if (regionLabels.fp) regionLabels.fp.setAttribute('x', Math.min(xMax - 25, thresholdX + 28));
  if (regionLabels.fn) regionLabels.fn.setAttribute('x', Math.max(xMin + 25, thresholdX - 28));
  if (regionLabels.tp) regionLabels.tp.setAttribute('x', (thresholdX + xMax) / 2);

  const leftRoom = thresholdX - xMin;
  const rightRoom = xMax - thresholdX;
  if (regionLabels.tn) regionLabels.tn.style.opacity = leftRoom > 45 ? '1' : '0';
  if (regionLabels.fn) regionLabels.fn.style.opacity = leftRoom > 70 ? '1' : '0';
  if (regionLabels.fp) regionLabels.fp.style.opacity = rightRoom > 70 ? '1' : '0';
  if (regionLabels.tp) regionLabels.tp.style.opacity = rightRoom > 45 ? '1' : '0';

  const predictedNegative = document.getElementById('dist-pred-negative');
  const predictedPositive = document.getElementById('dist-pred-positive');
  if (predictedNegative) {
    predictedNegative.setAttribute('x', (xMin + thresholdX) / 2);
    predictedNegative.style.opacity = leftRoom > 100 ? '1' : '0';
  }
  if (predictedPositive) {
    predictedPositive.setAttribute('x', (thresholdX + xMax) / 2);
    predictedPositive.style.opacity = rightRoom > 100 ? '1' : '0';
  }

  document.getElementById('dist-stat-tn').textContent = `${((1 - fpr) * 100).toFixed(2)}%`;
  document.getElementById('dist-stat-fp').textContent = `${(fpr * 100).toFixed(2)}%`;
  document.getElementById('dist-stat-fn').textContent = `${((1 - tpr) * 100).toFixed(2)}%`;
  document.getElementById('dist-stat-tp').textContent = `${(tpr * 100).toFixed(2)}%`;
}
