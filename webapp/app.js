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
const PROCESS_PREVIEW = window.PROCESS_PREVIEW_DATA || null;

let rocRawTrail = [];
let rocMorphTrail = [];
let currentRocProcessIndex = 0;
let currentRocThreshold = 0.11;
let processPreviewScores = null;
let processPreviewRgb = null;
const ROC_VIEWBOX = {x: 0, y: 0, width: 500, height: 450};
let rocView = {...ROC_VIEWBOX};
let rocPan = null;

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
  if (key === 'l1' || key === 'l2') {
    updateRocProcess(currentRocProcessIndex, currentRocThreshold);
  }
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
  const l1Points = document.getElementById('roc-points-l1');
  if (l1Points) l1Points.style.display = rocVisibility.l1 ? '' : 'none';
  if (zoomL1) zoomL1.style.display = rocVisibility.l1 ? '' : 'none';
  if (zoomL1Points) zoomL1Points.style.display = rocVisibility.l1 ? '' : 'none';

  // Line 2 elements
  const l2El = document.getElementById('dynamic-guide-l2');
  if (l2El) l2El.style.display = rocVisibility.l2 ? '' : 'none';
  const l2Points = document.getElementById('roc-points-l2');
  if (l2Points) l2Points.style.display = rocVisibility.l2 ? '' : 'none';
  if (zoomL2) zoomL2.style.display = rocVisibility.l2 ? '' : 'none';
  if (zoomL2Points) zoomL2Points.style.display = rocVisibility.l2 ? '' : 'none';

  // Best Dot Line 1 group
  const bestL1El = document.getElementById('best-group-l1');
  if (bestL1El) bestL1El.style.display = rocVisibility.bestL1 ? '' : 'none';
  if (zoomBestL1) zoomBestL1.style.display = rocVisibility.bestL1 ? '' : 'none';

  // Best Dot Line 2 group
  const bestL2El = document.getElementById('best-group-l2');
  if (bestL2El) bestL2El.style.display = rocVisibility.bestL2 ? '' : 'none';
  if (zoomBestL2) zoomBestL2.style.display = rocVisibility.bestL2 ? '' : 'none';

  // Projection Crosshair Lines (visible if at least one line is visible)
  const projEl = document.getElementById('roc-projection-group');
  if (projEl) projEl.style.display = (rocVisibility.l1 || rocVisibility.l2) ? '' : 'none';
  const rawActiveDot = document.getElementById('roc-active-marker-l1');
  if (rawActiveDot) rawActiveDot.style.display = rocVisibility.l1 ? '' : 'none';
  const morphActiveDot = document.getElementById('roc-active-marker-l2');
  if (morphActiveDot) morphActiveDot.style.display = rocVisibility.l2 ? '' : 'none';
  if (zoomProjEl) zoomProjEl.style.display = (rocVisibility.l1 || rocVisibility.l2) ? '' : 'none';

  // Summary box stat rows
  const statL1 = document.getElementById('stat-row-l1');
  if (statL1) statL1.style.opacity = rocVisibility.l1 ? '1' : '0.35';

  const statL2 = document.getElementById('stat-row-l2');
  if (statL2) statL2.style.opacity = rocVisibility.l2 ? '1' : '0.35';
}

function buildSmoothRocPath(points, toX, toY) {
  if (!points || !points.length) return '';
  const pts = points.map(p => ({ x: Number(toX(p.fpr)), y: Number(toY(p.tpr)) }));
  const cleaned = pts;
  if (cleaned.length === 1) return `M ${cleaned[0].x.toFixed(2)} ${cleaned[0].y.toFixed(2)}`;
  if (cleaned.length === 2) return `M ${cleaned[0].x.toFixed(2)} ${cleaned[0].y.toFixed(2)} L ${cleaned[1].x.toFixed(2)} ${cleaned[1].y.toFixed(2)}`;

  // Catmull-Rom to Cubic Bezier: smooth C1 curve passing precisely through all evaluated points ("นับทุกจุด")
  const alpha = 0.35;
  let path = `M ${cleaned[0].x.toFixed(2)} ${cleaned[0].y.toFixed(2)}`;
  for (let i = 0; i < cleaned.length - 1; i++) {
    const p0 = i > 0 ? cleaned[i - 1] : cleaned[i];
    const p1 = cleaned[i];
    const p2 = cleaned[i + 1];
    const p3 = i + 2 < cleaned.length ? cleaned[i + 2] : p2;

    const c1x = p1.x + (p2.x - p0.x) * alpha / 3.0;
    const c1y = p1.y + (p2.y - p0.y) * alpha / 3.0;
    const c2x = p2.x - (p3.x - p1.x) * alpha / 3.0;
    const c2y = p2.y - (p3.y - p1.y) * alpha / 3.0;

    path += ` C ${c1x.toFixed(2)} ${c1y.toFixed(2)}, ${c2x.toFixed(2)} ${c2y.toFixed(2)}, ${p2.x.toFixed(2)} ${p2.y.toFixed(2)}`;
  }
  return path;
}

function actualRocPoints(thresholds, fprValues, tprValues) {
  const points = [];
  for (let i = thresholds.length - 1; i >= 0; i--) {
    points.push({threshold: thresholds[i], fpr: fprValues[i], tpr: tprValues[i]});
  }
  return points;
}

function renderRocStepPoints(groupId, points, toX, toY) {
  const group = document.getElementById(groupId);
  if (!group) return;
  group.replaceChildren();

  points.forEach((point, index) => {
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', toX(point.fpr).toFixed(3));
    circle.setAttribute('cy', toY(point.tpr).toFixed(3));
    circle.setAttribute('r', index === 0 || index === points.length - 1 ? '2.4' : '1.65');
    circle.setAttribute('tabindex', '0');

    const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    title.textContent = `Threshold ${Number(point.threshold).toFixed(6)} | FPR ${point.fpr.toFixed(6)} | TPR ${point.tpr.toFixed(6)}`;
    circle.appendChild(title);
    group.appendChild(circle);
  });
}

function decodeBase64Bytes(encoded) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function morphologyPass(mask, width, height, operation, kernel) {
  const result = new Uint8Array(mask.length);
  const center = Math.floor(kernel.length / 2);
  const offsets = [];
  kernel.forEach((row, ky) => row.forEach((active, kx) => {
    if (active) offsets.push([kx - center, ky - center]);
  }));

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = operation === 'erode' ? 1 : 0;
      for (const [dx, dy] of offsets) {
        const px = x + dx;
        const py = y + dy;
        const inside = px >= 0 && px < width && py >= 0 && py < height;
        const neighbor = inside ? mask[py * width + px] : (operation === 'erode' ? 1 : 0);
        if (operation === 'erode' && !neighbor) { value = 0; break; }
        if (operation === 'dilate' && neighbor) { value = 1; break; }
      }
      result[y * width + x] = value;
    }
  }
  return result;
}

function paintProcessCanvas(canvasId, mask, rgb, width, height, cutout = false) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const image = context.createImageData(width, height);

  for (let i = 0; i < mask.length; i++) {
    const dst = i * 4;
    if (cutout) {
      image.data[dst] = mask[i] ? rgb[i * 3] : 0;
      image.data[dst + 1] = mask[i] ? rgb[i * 3 + 1] : 0;
      image.data[dst + 2] = mask[i] ? rgb[i * 3 + 2] : 0;
    } else {
      const value = mask[i] ? 255 : 0;
      image.data[dst] = value;
      image.data[dst + 1] = value;
      image.data[dst + 2] = value;
    }
    image.data[dst + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function paintProcessInput() {
  if (!PROCESS_PREVIEW || !processPreviewRgb) return;
  const canvas = document.getElementById('process-canvas-input');
  if (!canvas) return;
  const {width, height} = PROCESS_PREVIEW;
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const image = context.createImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    image.data[i * 4] = processPreviewRgb[i * 3];
    image.data[i * 4 + 1] = processPreviewRgb[i * 3 + 1];
    image.data[i * 4 + 2] = processPreviewRgb[i * 3 + 2];
    image.data[i * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function renderProcessPreview(threshold) {
  if (!PROCESS_PREVIEW || !processPreviewScores || !processPreviewRgb) return;
  const {width, height, kernel} = PROCESS_PREVIEW;
  const raw = new Uint8Array(processPreviewScores.length);
  for (let i = 0; i < processPreviewScores.length; i++) {
    raw[i] = processPreviewScores[i] >= threshold ? 1 : 0;
  }

  const eroded = morphologyPass(raw, width, height, 'erode', kernel);
  const opened = morphologyPass(eroded, width, height, 'dilate', kernel);
  const dilated = morphologyPass(opened, width, height, 'dilate', kernel);
  const closed = morphologyPass(dilated, width, height, 'erode', kernel);

  paintProcessCanvas('process-canvas-raw', raw, processPreviewRgb, width, height);
  paintProcessCanvas('process-canvas-open', opened, processPreviewRgb, width, height);
  paintProcessCanvas('process-canvas-close', closed, processPreviewRgb, width, height);
  paintProcessCanvas('process-canvas-cutout', closed, processPreviewRgb, width, height, true);

  const count = values => values.reduce((sum, value) => sum + value, 0);
  const counts = document.getElementById('process-preview-counts');
  if (counts) {
    counts.textContent = `Foreground pixels: Raw ${count(raw).toLocaleString()} → Opening ${count(opened).toLocaleString()} → Closing ${count(closed).toLocaleString()}`;
  }
}

function initProcessPreview() {
  if (!PROCESS_PREVIEW) return;
  processPreviewRgb = decodeBase64Bytes(PROCESS_PREVIEW.rgb_u8_base64);
  const scoreBytes = decodeBase64Bytes(PROCESS_PREVIEW.score_f32_base64);
  processPreviewScores = new Float32Array(scoreBytes.buffer);
  paintProcessInput();

  const sample = document.getElementById('process-preview-sample');
  if (sample) sample.textContent = `${PROCESS_PREVIEW.sample} · ${PROCESS_PREVIEW.width}×${PROCESS_PREVIEW.height} px`;
  renderProcessPreview(currentRocThreshold);
}

function updateRocProcess(value, requestedThreshold = null) {
  if (!rocRawTrail.length || !rocMorphTrail.length) return;
  const maxIndex = Math.min(rocRawTrail.length, rocMorphTrail.length) - 1;
  currentRocProcessIndex = Math.max(0, Math.min(maxIndex, Math.round(Number(value))));
  const rawPoint = rocRawTrail[currentRocProcessIndex];
  const morphPoint = rocMorphTrail[currentRocProcessIndex];
  currentRocThreshold = requestedThreshold === null
    ? Number(rawPoint.threshold)
    : Math.max(0, Math.min(1, Number(requestedThreshold)));
  const toX = fpr => 50 + fpr * 420;
  const toY = tpr => 360 - tpr * 320;
  const rawProgress = rocRawTrail.slice(0, currentRocProcessIndex + 1);
  const morphProgress = rocMorphTrail.slice(0, currentRocProcessIndex + 1);

  // Grow or shrink each visible trail through the points processed so far.
  document.getElementById('dynamic-guide-l1')?.setAttribute(
    'd', buildSmoothRocPath(rawProgress, toX, toY)
  );
  document.getElementById('dynamic-guide-l2')?.setAttribute(
    'd', buildSmoothRocPath(morphProgress, toX, toY)
  );
  renderRocStepPoints('roc-points-l1', rawProgress, toX, toY);
  renderRocStepPoints('roc-points-l2', morphProgress, toX, toY);

  // Active Operating Point: prefer morphology if L2 visible, else raw
  const activePt = (rocVisibility.l2 ? morphPoint : rawPoint) || rawPoint;
  const ax = toX(activePt.fpr);
  const ay = toY(activePt.tpr);
  const rawAx = toX(rawPoint.fpr);
  const rawAy = toY(rawPoint.tpr);
  const morphAx = toX(morphPoint.fpr);
  const morphAy = toY(morphPoint.tpr);

  // Update Textbook-Style Projection Lines (FPRa / TPRa like User Reference Image)
  const projX = document.getElementById('roc-proj-x');
  if (projX) {
    projX.setAttribute('x1', ax.toFixed(2));
    projX.setAttribute('y1', ay.toFixed(2));
    projX.setAttribute('x2', ax.toFixed(2));
    projX.setAttribute('y2', '360');
  }
  const projY = document.getElementById('roc-proj-y');
  if (projY) {
    projY.setAttribute('x1', ax.toFixed(2));
    projY.setAttribute('y1', ay.toFixed(2));
    projY.setAttribute('x2', '50');
    projY.setAttribute('y2', ay.toFixed(2));
  }
  const rawDot = document.getElementById('roc-active-marker-l1');
  if (rawDot) {
    rawDot.setAttribute('cx', rawAx.toFixed(2));
    rawDot.setAttribute('cy', rawAy.toFixed(2));
  }
  const morphDot = document.getElementById('roc-active-marker-l2');
  if (morphDot) {
    morphDot.setAttribute('cx', morphAx.toFixed(2));
    morphDot.setAttribute('cy', morphAy.toFixed(2));
  }
  const xTag = document.getElementById('roc-proj-x-tag');
  if (xTag) xTag.setAttribute('transform', `translate(${ax.toFixed(2)}, 360)`);
  const yTag = document.getElementById('roc-proj-y-tag');
  if (yTag) yTag.setAttribute('transform', `translate(50, ${ay.toFixed(2)})`);
  const xVal = document.getElementById('roc-proj-x-val');
  if (xVal) xVal.textContent = `FPRₐ ${(activePt.fpr * 100).toFixed(2)}%`;
  const yVal = document.getElementById('roc-proj-y-val');
  if (yVal) yVal.textContent = `TPRₐ ${(activePt.tpr * 100).toFixed(2)}%`;

  const slider = document.getElementById('roc-process-slider');
  if (slider) slider.value = currentRocThreshold.toFixed(3);
  const setText = (id, text) => {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
  };
  setText('roc-process-step', `Step ${currentRocProcessIndex + 1} / ${maxIndex + 1} (นับครบทุกจุด 57 ระดับ)`);
  const thresholdInput = document.getElementById('roc-process-threshold');
  if (thresholdInput) thresholdInput.value = currentRocThreshold.toFixed(6);
  setText('roc-live-raw-fpr', rawPoint.fpr.toFixed(6));
  setText('roc-live-raw-tpr', rawPoint.tpr.toFixed(6));
  setText('roc-live-morph-fpr', morphPoint.fpr.toFixed(6));
  setText('roc-live-morph-tpr', morphPoint.tpr.toFixed(6));
  renderProcessPreview(currentRocThreshold);
  applyRocVisibility();
}

function updateRocProcessThreshold(value) {
  if (!rocRawTrail.length) return;
  const threshold = Math.max(0, Math.min(1, Number(value)));
  let index = rocRawTrail.findIndex(point => point.threshold < threshold) - 1;
  if (index < 0) index = 0;
  if (!rocRawTrail.some(point => point.threshold < threshold)) index = rocRawTrail.length - 1;
  updateRocProcess(index, threshold);
}

function stepRocProcess(direction) {
  const nextThreshold = Math.max(
    0,
    Math.min(1, currentRocThreshold + Number(direction) * 0.001)
  );
  updateRocProcessThreshold(Number(nextThreshold.toFixed(3)));
}

function clampRocView() {
  rocView.width = Math.max(62.5, Math.min(ROC_VIEWBOX.width, rocView.width));
  rocView.height = rocView.width * (ROC_VIEWBOX.height / ROC_VIEWBOX.width);
  rocView.x = Math.max(0, Math.min(ROC_VIEWBOX.width - rocView.width, rocView.x));
  rocView.y = Math.max(0, Math.min(ROC_VIEWBOX.height - rocView.height, rocView.y));
}

function applyRocView() {
  clampRocView();
  const svg = document.getElementById('roc-svg');
  if (svg) {
    svg.setAttribute('viewBox', [rocView.x, rocView.y, rocView.width, rocView.height]
      .map(value => value.toFixed(3)).join(' '));
  }
  const level = document.getElementById('roc-zoom-level');
  if (level) level.value = `${Math.round((ROC_VIEWBOX.width / rocView.width) * 100)}%`;
}

function zoomRocAt(factor, clientX = null, clientY = null) {
  const svg = document.getElementById('roc-svg');
  if (!svg) return;
  const rect = svg.getBoundingClientRect();
  const px = clientX === null ? 0.5 : Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
  const py = clientY === null ? 0.5 : Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
  const anchorX = rocView.x + px * rocView.width;
  const anchorY = rocView.y + py * rocView.height;
  const nextWidth = Math.max(62.5, Math.min(ROC_VIEWBOX.width, rocView.width * factor));
  const nextHeight = nextWidth * (ROC_VIEWBOX.height / ROC_VIEWBOX.width);
  rocView.x = anchorX - px * nextWidth;
  rocView.y = anchorY - py * nextHeight;
  rocView.width = nextWidth;
  rocView.height = nextHeight;
  applyRocView();
}

function zoomRocGraph(factor) {
  zoomRocAt(Number(factor));
}

function resetRocZoom() {
  rocView = {...ROC_VIEWBOX};
  applyRocView();
}

function initRocZoom() {
  const svg = document.getElementById('roc-svg');
  if (!svg) return;
  applyRocView();
  svg.addEventListener('wheel', event => {
    event.preventDefault();
    zoomRocAt(event.deltaY < 0 ? 0.82 : 1.22, event.clientX, event.clientY);
  }, {passive: false});
  svg.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    rocPan = {clientX: event.clientX, clientY: event.clientY, x: rocView.x, y: rocView.y};
    svg.setPointerCapture(event.pointerId);
    svg.classList.add('is-panning');
  });
  svg.addEventListener('pointermove', event => {
    if (!rocPan) return;
    const rect = svg.getBoundingClientRect();
    rocView.x = rocPan.x - (event.clientX - rocPan.clientX) * rocView.width / rect.width;
    rocView.y = rocPan.y - (event.clientY - rocPan.clientY) * rocView.height / rect.height;
    applyRocView();
  });
  const stopPan = event => {
    if (!rocPan) return;
    rocPan = null;
    svg.classList.remove('is-panning');
    if (svg.hasPointerCapture(event.pointerId)) svg.releasePointerCapture(event.pointerId);
  };
  svg.addEventListener('pointerup', stopPan);
  svg.addEventListener('pointercancel', stopPan);
  svg.addEventListener('dblclick', resetRocZoom);
}
function initRocCurves() {
  const toX = fpr => 50 + fpr * 420;
  const toY = tpr => 360 - tpr * 320;

  const hasActualCurves = SCORE_DISTRIBUTION
    && SCORE_DISTRIBUTION.raw_curve
    && SCORE_DISTRIBUTION.morphology_curve;
  const rawTrail = hasActualCurves
    ? actualRocPoints(
        SCORE_DISTRIBUTION.raw_curve.thresholds,
        SCORE_DISTRIBUTION.raw_curve.fpr,
        SCORE_DISTRIBUTION.raw_curve.tpr
      )
    : [...RAW_OPERATING_POINTS].sort((a, b) => b.threshold - a.threshold);
  const morphTrail = hasActualCurves
    ? actualRocPoints(
        SCORE_DISTRIBUTION.morphology_curve.thresholds,
        SCORE_DISTRIBUTION.morphology_curve.fpr,
        SCORE_DISTRIBUTION.morphology_curve.tpr
      )
    : [...MORPH_OPERATING_POINTS].sort((a, b) => b.threshold - a.threshold);

  rocRawTrail = rawTrail;
  rocMorphTrail = morphTrail;

  const guide1 = document.getElementById('dynamic-guide-l1');
  if (guide1) {
    guide1.setAttribute('d', buildSmoothRocPath(rawTrail, toX, toY));
  }

  const guide2 = document.getElementById('dynamic-guide-l2');
  if (guide2) {
    guide2.setAttribute('d', buildSmoothRocPath(morphTrail, toX, toY));
  }

  if (hasActualCurves) {
    renderRocStepPoints('roc-points-l1', rawTrail, toX, toY);
    renderRocStepPoints('roc-points-l2', morphTrail, toX, toY);
  }

  const resolutionLabel = document.getElementById('roc-resolution-label');
  if (resolutionLabel && hasActualCurves) {
    const rawThresholdCount = SCORE_DISTRIBUTION.raw_curve.thresholds.length;
    const morphThresholdCount = SCORE_DISTRIBUTION.morphology_curve.thresholds.length;
    resolutionLabel.textContent = `Exact ROC • Raw ${rawTrail.length}/${rawThresholdCount} steps • Morphology ${morphTrail.length}/${morphThresholdCount} steps`;
  }

  if (hasActualCurves) {
    const slider = document.getElementById('roc-process-slider');
    if (slider) {
      slider.min = 0;
      slider.max = 1;
      slider.step = 0.001;
    }
    updateRocProcessThreshold(0.11);
  }

  applyRocVisibility();
}

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  setStage('before');
  selectCell('TN');
  updateGraySlider(128);
  initRocCurves();
  initRocZoom();
  initProcessPreview();
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


// ROC Dual Curve Simulation Slider (Retained as safe no-op stub)
function updateRocSlider(thresholdVal) {
  updateRocProcessThreshold(thresholdVal);
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
