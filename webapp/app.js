// ==========================================================================
// Segmentation Workshop Explorer - Interactive JavaScript
// ==========================================================================

// Real project dataset numbers
const DATA = {
  before: {
    TN: 767091,
    FP: 656811,
    FN: 150671,
    TP: 564479,
    acc: 0.6225,
    prec: 0.4622,
    rec: 0.7893,
    iou: 0.4114,
    dice: 0.5830
  },
  after: {
    TN: 812274,
    FP: 611628,
    FN: 155347,
    TP: 559803,
    acc: 0.6414,
    prec: 0.4779,
    rec: 0.7828,
    iou: 0.4219,
    dice: 0.5935
  }
};

const CELL_DETAILS = {
  TN: {
    tag: "TN: True Negative (ลบแท้)",
    status: "ทำนายถูกต้อง (Correct)",
    isOk: true,
    title: "พิกเซลพื้นหลัง ที่ทายถูกว่าเป็นพื้นหลัง",
    desc: "พิกเซลเหล่านี้คือหญ้า พื้น ผนัง หรือวัตถุภายนอกที่ไม่มีสัตว์อยู่ และโปรแกรมระบาย Mask เป็นสีดำ (0) อย่างถูกต้อง",
    gtText: "Ground Truth: ดำ (0/พื้นหลัง)",
    predText: "Predicted: ดำ (0/พื้นหลัง)",
    impact: "ในภาพถ่าย สัดส่วนพื้นหลังมักมีขนาดใหญ่กว่าตัวสัตว์มาก ค่า TN ที่สูงจึงส่งผลให้ค่า Accuracy รวมสูงตามไปด้วย"
  },
  FP: {
    tag: "FP: False Positive (บวกเท็จ)",
    status: "ทำนายผิดพลาด (Error)",
    isOk: false,
    title: "พิกเซลพื้นหลัง แต่ทายผิดว่าเป็นสัตว์ (ขอบเกิน)",
    desc: "พิกเซลเหล่านี้เป็นพื้นหลัง แต่โปรแกรมกลับระบายเป็นสีขาว (1/สัตว์) มักเกิดจากสีของฉากหลังที่ดันแตกต่างจากขอบภาพ ทำให้ระบบคิดว่าเป็นสัตว์",
    gtText: "Ground Truth: ดำ (0/พื้นหลัง)",
    predText: "Predicted: ขาว (1/สัตว์)",
    impact: "ส่งผลให้ค่า Precision ต่ำลงโดยตรง เพราะมีพิกเซลขยะที่ไม่ใช่สัตว์ปะปนเข้ามาในผลทำนายเยอะเกินไป"
  },
  FN: {
    tag: "FN: False Negative (ลบเท็จ)",
    status: "ทำนายผิดพลาด (Error)",
    isOk: false,
    title: "พิกเซลสัตว์ แต่ทายผิดว่าเป็นพื้นหลัง (ตัวแหว่ง)",
    desc: "พิกเซลเหล่านี้คือเนื้อตัว ลำตัว ขน หรือหูของสัตว์ แต่โปรแกรมระบายเป็นสีดำ (0) พลาดทิ้งไป มักเกิดจากสัตว์มีสีใกล้เคียงกับขอบภาพ",
    gtText: "Ground Truth: ขาว (1/สัตว์)",
    predText: "Predicted: ดำ (0/พื้นหลัง)",
    impact: "ส่งผลให้ค่า Recall ลดลงโดยตรง บ่งชี้ว่าเราตรวจจับสัตว์ได้ไม่ครบถ้วน ลำตัวหรือขนบางส่วนแหว่งหายไป"
  },
  TP: {
    tag: "TP: True Positive (บวกแท้)",
    status: "ทำนายถูกต้อง (Correct)",
    isOk: true,
    title: "พิกเซลสัตว์ ที่ทายถูกว่าเป็นสัตว์ (ตรวจพบสำเร็จ)",
    desc: "พิกเซลส่วนที่เป็นตัวสัตว์เลี้ยงจริงๆ และโปรแกรมระบาย Mask เป็นสีขาว (1) ได้อย่างแม่นยำตรงเป้า",
    gtText: "Ground Truth: ขาว (1/สัตว์)",
    predText: "Predicted: ขาว (1/สัตว์)",
    impact: "เป็นตัวตั้งหลักของทั้ง Precision, Recall, IoU และ Dice ยิ่งค่า TP มากเท่าไหร่ การตัดขอบภาพยิ่งมีความทับซ้อนสูง"
  }
};

let currentStage = 'before';
let currentCell = 'TN';

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  setStage('before');
  selectCell('TN');
  updateGraySlider(128);
  updateRocSlider(0.40);
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
  if (val === 0) label = "0: Pure Black (พื้นหลัง)";
  else if (val === 255) label = "255: Pure White (ตัวสัตว์)";
  else if (val === 128) label = "128: Mid Gray (ขอบที่ไม่แน่นอน)";

  text.textContent = `Value: ${val} / 255 (${label})`;
  text.style.color = val > 140 ? '#000000' : '#ffffff';
}

function setGrayPreset(val) {
  updateGraySlider(val);
}

// ROC Curve Simulation Slider
function updateRocSlider(thresholdVal) {
  const t = parseFloat(thresholdVal);
  document.getElementById('slider-threshold-val').textContent = t.toFixed(2);

  // Model a smooth ROC mapping: as threshold increases, both FPR and TPR drop
  // At t=0.10: FPR ~ 0.85, TPR ~ 0.98
  // At t=0.40: FPR ~ 0.46, TPR ~ 0.79
  // At t=0.90: FPR ~ 0.05, TPR ~ 0.22
  const fpr = Math.max(0.01, Math.min(0.99, 1.0 - Math.pow(t, 0.7)));
  const tpr = Math.max(0.05, Math.min(0.99, 1.0 - Math.pow(t, 1.5)));

  document.getElementById('sim-fpr').textContent = fpr.toFixed(3);
  document.getElementById('sim-tpr').textContent = tpr.toFixed(3);

  // Map to SVG coordinates:
  // x-axis: 0.0 -> 50, 1.0 -> 470 (range 420)
  // y-axis: 0.0 -> 360, 1.0 -> 40 (range 320, inverted)
  const svgX = 50 + fpr * 420;
  const svgY = 360 - tpr * 320;

  const marker = document.getElementById('dynamic-marker');
  if (marker) {
    marker.setAttribute('cx', svgX);
    marker.setAttribute('cy', svgY);
  }
}

// Highlight Before or After point
function highlightPoint(stage) {
  if (stage === 'before') {
    updateRocSlider(0.40);
    setStage('before');
  } else {
    setStage('after');
  }
}
