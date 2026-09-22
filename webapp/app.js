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

// 51 Sampled dots along Line 1 (Threshold) and Line 2 (Morphology) at every 0.02 FPR
const LINE1_DOTS = [
  {fpr: 0.0, tpr: 0.7061}, {fpr: 0.02, tpr: 0.9206}, {fpr: 0.04, tpr: 0.9225}, {fpr: 0.06, tpr: 0.9243},
  {fpr: 0.08, tpr: 0.9262}, {fpr: 0.1, tpr: 0.9281}, {fpr: 0.12, tpr: 0.9299}, {fpr: 0.14, tpr: 0.9318},
  {fpr: 0.16, tpr: 0.9337}, {fpr: 0.18, tpr: 0.9356}, {fpr: 0.2, tpr: 0.9374}, {fpr: 0.22, tpr: 0.939},
  {fpr: 0.24, tpr: 0.9405}, {fpr: 0.26, tpr: 0.9421}, {fpr: 0.28, tpr: 0.9437}, {fpr: 0.3, tpr: 0.9452},
  {fpr: 0.32, tpr: 0.9468}, {fpr: 0.34, tpr: 0.9484}, {fpr: 0.36, tpr: 0.9499}, {fpr: 0.38, tpr: 0.9515},
  {fpr: 0.4, tpr: 0.9531}, {fpr: 0.42, tpr: 0.9546}, {fpr: 0.44, tpr: 0.9562}, {fpr: 0.46, tpr: 0.9577},
  {fpr: 0.48, tpr: 0.9593}, {fpr: 0.5, tpr: 0.9609}, {fpr: 0.52, tpr: 0.9624}, {fpr: 0.54, tpr: 0.964},
  {fpr: 0.56, tpr: 0.9656}, {fpr: 0.58, tpr: 0.9671}, {fpr: 0.6, tpr: 0.9687}, {fpr: 0.62, tpr: 0.9703},
  {fpr: 0.64, tpr: 0.9718}, {fpr: 0.66, tpr: 0.9734}, {fpr: 0.68, tpr: 0.975}, {fpr: 0.7, tpr: 0.9765},
  {fpr: 0.72, tpr: 0.9781}, {fpr: 0.74, tpr: 0.9797}, {fpr: 0.76, tpr: 0.9812}, {fpr: 0.78, tpr: 0.9828},
  {fpr: 0.8, tpr: 0.9844}, {fpr: 0.82, tpr: 0.9859}, {fpr: 0.84, tpr: 0.9875}, {fpr: 0.86, tpr: 0.989},
  {fpr: 0.88, tpr: 0.9906}, {fpr: 0.9, tpr: 0.9922}, {fpr: 0.92, tpr: 0.9937}, {fpr: 0.94, tpr: 0.9953},
  {fpr: 0.96, tpr: 0.9969}, {fpr: 0.98, tpr: 0.9984}, {fpr: 1.0, tpr: 1.0}
];

const LINE2_DOTS = [
  {fpr: 0.0, tpr: 0.7036}, {fpr: 0.02, tpr: 0.9184}, {fpr: 0.04, tpr: 0.9207}, {fpr: 0.06, tpr: 0.9229},
  {fpr: 0.08, tpr: 0.9252}, {fpr: 0.1, tpr: 0.9275}, {fpr: 0.12, tpr: 0.9297}, {fpr: 0.14, tpr: 0.932},
  {fpr: 0.16, tpr: 0.9343}, {fpr: 0.18, tpr: 0.9362}, {fpr: 0.2, tpr: 0.9377}, {fpr: 0.22, tpr: 0.9393},
  {fpr: 0.24, tpr: 0.9408}, {fpr: 0.26, tpr: 0.9424}, {fpr: 0.28, tpr: 0.9439}, {fpr: 0.3, tpr: 0.9455},
  {fpr: 0.32, tpr: 0.9471}, {fpr: 0.34, tpr: 0.9486}, {fpr: 0.36, tpr: 0.9502}, {fpr: 0.38, tpr: 0.9517},
  {fpr: 0.4, tpr: 0.9533}, {fpr: 0.42, tpr: 0.9548}, {fpr: 0.44, tpr: 0.9564}, {fpr: 0.46, tpr: 0.958},
  {fpr: 0.48, tpr: 0.9595}, {fpr: 0.5, tpr: 0.9611}, {fpr: 0.52, tpr: 0.9626}, {fpr: 0.54, tpr: 0.9642},
  {fpr: 0.56, tpr: 0.9657}, {fpr: 0.58, tpr: 0.9673}, {fpr: 0.6, tpr: 0.9689}, {fpr: 0.62, tpr: 0.9704},
  {fpr: 0.64, tpr: 0.972}, {fpr: 0.66, tpr: 0.9735}, {fpr: 0.68, tpr: 0.9751}, {fpr: 0.7, tpr: 0.9766},
  {fpr: 0.72, tpr: 0.9782}, {fpr: 0.74, tpr: 0.9798}, {fpr: 0.76, tpr: 0.9813}, {fpr: 0.78, tpr: 0.9829},
  {fpr: 0.8, tpr: 0.9844}, {fpr: 0.82, tpr: 0.986}, {fpr: 0.84, tpr: 0.9875}, {fpr: 0.86, tpr: 0.9891},
  {fpr: 0.88, tpr: 0.9907}, {fpr: 0.9, tpr: 0.9922}, {fpr: 0.92, tpr: 0.9938}, {fpr: 0.94, tpr: 0.9953},
  {fpr: 0.96, tpr: 0.9969}, {fpr: 0.98, tpr: 0.9984}, {fpr: 1.0, tpr: 1.0}
];

let currentStage = 'before';
let currentCell = 'TN';

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
  renderDualRocCurves();
  setStage('before');
  selectCell('TN');
  updateGraySlider(128);
  updateRocSlider(0.11);
});

// Render the 2 ROC lines and 0.02 FPR dots into the SVG
function renderDualRocCurves() {
  const svg = document.getElementById('roc-svg');
  if (!svg) return;

  // Coordinate mapping: FPR 0..1 -> x 50..470; TPR 0..1 -> y 360..40
  const toX = fpr => 50 + fpr * 420;
  const toY = tpr => 360 - tpr * 320;

  // Path Line 1: Threshold
  let d1 = `M ${toX(LINE1_DOTS[0].fpr)} ${toY(LINE1_DOTS[0].tpr)}`;
  for (let i = 1; i < LINE1_DOTS.length; i++) {
    d1 += ` L ${toX(LINE1_DOTS[i].fpr)} ${toY(LINE1_DOTS[i].tpr)}`;
  }
  const path1 = document.getElementById('roc-path-line1');
  if (path1) path1.setAttribute('d', d1);

  // Path Line 2: Morphology
  let d2 = `M ${toX(LINE2_DOTS[0].fpr)} ${toY(LINE2_DOTS[0].tpr)}`;
  for (let i = 1; i < LINE2_DOTS.length; i++) {
    d2 += ` L ${toX(LINE2_DOTS[i].fpr)} ${toY(LINE2_DOTS[i].tpr)}`;
  }
  const path2 = document.getElementById('roc-path-line2');
  if (path2) path2.setAttribute('d', d2);

  // Render 0.02 FPR dots for Line 1
  const gDots1 = document.getElementById('dots-line1-group');
  if (gDots1) {
    gDots1.innerHTML = '';
    LINE1_DOTS.forEach(pt => {
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('cx', toX(pt.fpr));
      c.setAttribute('cy', toY(pt.tpr));
      c.setAttribute('r', '3');
      c.setAttribute('class', 'sub-dot sub-dot-line1');
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `Line 1 (Threshold): FPR = ${pt.fpr.toFixed(2)}, TPR = ${pt.tpr.toFixed(4)}`;
      c.appendChild(title);
      gDots1.appendChild(c);
    });
  }

  // Render 0.02 FPR dots for Line 2
  const gDots2 = document.getElementById('dots-line2-group');
  if (gDots2) {
    gDots2.innerHTML = '';
    LINE2_DOTS.forEach(pt => {
      const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      c.setAttribute('cx', toX(pt.fpr));
      c.setAttribute('cy', toY(pt.tpr));
      c.setAttribute('r', '3');
      c.setAttribute('class', 'sub-dot sub-dot-line2');
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = `Line 2 (Morphology): FPR = ${pt.fpr.toFixed(2)}, TPR = ${pt.tpr.toFixed(4)}`;
      c.appendChild(title);
      gDots2.appendChild(c);
    });
  }
}

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

// ROC Dual Curve Simulation Slider
function updateRocSlider(thresholdVal) {
  const t = parseFloat(thresholdVal);
  document.getElementById('slider-threshold-val').textContent = t.toFixed(2);

  // Approximate mathematical model for Line 1 (Threshold) and Line 2 (Morphology):
  // At t=0.05: Line 1 FPR ~ 0.016, TPR ~ 0.920 | Line 2 FPR ~ 0.006, TPR ~ 0.917
  // At t=0.11: Line 1 FPR ~ 0.002, TPR ~ 0.906 | Line 2 FPR ~ 0.001, TPR ~ 0.899
  // At t=0.25: Line 1 FPR ~ 0.000, TPR ~ 0.828 | Line 2 FPR ~ 0.000, TPR ~ 0.826
  let fpr1 = 0.002;
  let tpr1 = 0.906;
  let fpr2 = 0.006;
  let tpr2 = 0.917;

  if (t < 0.11) {
    const ratio = (0.11 - t) / (0.11 - 0.03);
    fpr1 = 0.002 + ratio * 0.035;
    tpr1 = 0.906 + ratio * 0.018;
    fpr2 = 0.001 + ratio * 0.012;
    tpr2 = 0.899 + ratio * 0.022;
  } else {
    const ratio = (t - 0.11) / (0.50 - 0.11);
    fpr1 = Math.max(0.000, 0.002 * Math.exp(-ratio * 4));
    tpr1 = Math.max(0.600, 0.906 - ratio * 0.25);
    fpr2 = Math.max(0.000, 0.001 * Math.exp(-ratio * 4));
    tpr2 = Math.max(0.590, 0.899 - ratio * 0.25);
  }

  document.getElementById('sim-fpr-l1').textContent = fpr1.toFixed(3);
  document.getElementById('sim-tpr-l1').textContent = tpr1.toFixed(3);
  document.getElementById('sim-fpr-l2').textContent = fpr2.toFixed(3);
  document.getElementById('sim-tpr-l2').textContent = tpr2.toFixed(3);

  // SVG coordinate mapping
  const toX = fpr => 50 + fpr * 420;
  const toY = tpr => 360 - tpr * 320;

  const m1 = document.getElementById('dynamic-marker-l1');
  if (m1) {
    m1.setAttribute('cx', toX(fpr1));
    m1.setAttribute('cy', toY(tpr1));
  }

  const m2 = document.getElementById('dynamic-marker-l2');
  if (m2) {
    m2.setAttribute('cx', toX(fpr2));
    m2.setAttribute('cy', toY(tpr2));
  }
}

// Highlight Best Dots
function highlightPoint(stage) {
  if (stage === 'before') {
    updateRocSlider(0.11);
    setStage('before');
  } else {
    updateRocSlider(0.05);
    setStage('after');
  }
}
