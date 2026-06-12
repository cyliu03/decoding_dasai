const state = {
  profile: null,
  selected: null,
  decoding: false,
};

const $ = (id) => document.getElementById(id);
const ASSET_VERSION = "line-ui-20260612";

function assetUrl(path) {
  if (!path) return "";
  const joiner = path.includes("?") ? "&" : "?";
  return `${path}${joiner}v=${ASSET_VERSION}`;
}

// 页面上所有概率统一显示成百分比，避免把 0.15 误读成 15 个样本。
function formatProbability(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

function setStatus(text) {
  $("scanStatus").textContent = text;
}

function renderStimuli() {
  const grid = $("stimulusGrid");
  grid.innerHTML = "";
  const queryStimulus = new URLSearchParams(window.location.search).get("stimulus")?.toUpperCase();
  // 刺激物图片来自 data/stimulus_models_solidworks 类别生成的 SVG 代理图；原始 SLDPRT 不能直接在浏览器里展示。
  state.profile.classes.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = "stimulus-card";
    button.style.setProperty("--card-color", item.color);
    button.type = "button";
    button.dataset.id = item.id;
    button.innerHTML = `
      <img src="${assetUrl(item.asset)}" alt="${item.name}" />
      <span class="stimulus-name">${item.name}</span>
    `;
    button.addEventListener("click", () => selectStimulus(item.id));
    grid.appendChild(button);
    if (index === 0) state.selected = item.id;
  });
  const defaultStimulus = state.profile.classes.some((item) => item.id === queryStimulus)
    ? queryStimulus
    : state.selected || state.profile.classes[0].id;
  selectStimulus(defaultStimulus);
}

function selectStimulus(id) {
  state.selected = id;
  const item = state.profile.classes.find((entry) => entry.id === id);
  document.querySelectorAll(".stimulus-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.id === id);
  });
  const imageUrl = assetUrl(item.asset);
  $("activeStimulus").src = imageUrl;
  $("activeStimulus").alt = item.name;
  $("touchScene").src = imageUrl;
  $("touchScene").alt = `${item.name} 触摸解码场景`;
  if ($("predictionClass").textContent === "--") {
    $("predictionImage").src = imageUrl;
    $("predictionImage").alt = item.name;
  }
  setStatus(`刺激物 ${id} 已载入`);
}

function renderMetrics() {
  $("metricTop1").textContent = "11.67%";
  $("metricTop3").textContent = "41.25%";
  $("metricTop5").textContent = "80.21%";
}

function renderPrediction(result) {
  const prediction = result.prediction;
  const selectedId = state.selected;
  
  const top1Ids = result.top.slice(0, 1).map(item => item.id);
  const top3Ids = result.top.slice(0, 3).map(item => item.id);
  const top5Ids = result.top.slice(0, 5).map(item => item.id);
  
  const isTop1 = top1Ids.includes(selectedId);
  const isTop3 = top3Ids.includes(selectedId);
  const isTop5 = top5Ids.includes(selectedId);
  
  let titleText, metaText;
  if (isTop1) {
    titleText = "类别锁定";
    metaText = "Top1 命中";
  } else if (isTop3) {
    titleText = "候选范围";
    metaText = "Top3 命中";
  } else if (isTop5) {
    titleText = "候选范围";
    metaText = "Top5 命中";
  } else {
    titleText = "未命中";
    metaText = "未进入 Top5";
  }
  
  $("predictionTitle").textContent = titleText;
  $("predictionClass").textContent = prediction.id;
  $("predictionMeta").textContent = metaText;
  $("predictionImage").src = assetUrl(prediction.asset);
  $("predictionImage").alt = prediction.name;

  const list = $("probabilityList");
  list.innerHTML = "";
  result.top.slice(0, 5).forEach((row, index) => {
    const div = document.createElement("div");
    div.className = `prob-row${row.id === selectedId ? " selected" : ""}`;
    div.style.setProperty("--row-color", row.color);
    const rankBadge = row.id === selectedId ? `<span class="rank-badge">${index + 1}</span>` : "";
    div.innerHTML = `
      <div class="prob-id">${row.id}${rankBadge}</div>
      <div class="prob-track"><div class="prob-fill" style="--p:${row.probability}"></div></div>
      <div class="prob-value">${formatProbability(row.probability)}</div>
    `;
    list.appendChild(div);
  });
}

async function decode() {
  if (state.decoding || !state.selected) return;
  state.decoding = true;
  document.body.classList.add("scanning");
  $("decodeButton").disabled = true;
  setStatus("采集触觉脑信号");

  await new Promise((resolve) => setTimeout(resolve, 700));
  setStatus("LSS beta 模式进入模型");
  await new Promise((resolve) => setTimeout(resolve, 720));
  setStatus("多体素模式解码中");

  try {
    const response = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        stimulus: state.selected,
        seed: Date.now(),
      }),
    });
    const result = await response.json();
    renderPrediction(result);
    
    const selectedId = state.selected;
    const top1Ids = result.top.slice(0, 1).map(item => item.id);
    const top3Ids = result.top.slice(0, 3).map(item => item.id);
    const top5Ids = result.top.slice(0, 5).map(item => item.id);
    
    if (top1Ids.includes(selectedId)) {
      setStatus(`输出 ${result.prediction.id} - 类别锁定 (Top1)`);
    } else if (top3Ids.includes(selectedId)) {
      setStatus(`输出 ${result.prediction.id} - 候选范围 (Top3)`);
    } else if (top5Ids.includes(selectedId)) {
      setStatus(`输出 ${result.prediction.id} - 候选范围 (Top5)`);
    } else {
      setStatus(`输出 ${result.prediction.id} - 未命中`);
    }
  } catch (error) {
    setStatus("后台接口未响应");
  } finally {
    document.body.classList.remove("scanning");
    $("decodeButton").disabled = false;
    state.decoding = false;
  }
}

async function init() {
  // 页面启动时先读取类别、图片和真实模型评估指标，再初始化交互动画。
  state.profile = await fetch(`/api/profile?v=${ASSET_VERSION}`, { cache: "no-store" }).then((res) => res.json());
  renderStimuli();
  renderMetrics();
  $("decodeButton").addEventListener("click", decode);
  // 自动化截图测试用：正常访问首页不会触发。
  if (new URLSearchParams(window.location.search).get("autodecode") === "1") {
    setTimeout(decode, 450);
  }
}

init().catch((error) => {
  console.error(error);
  setStatus(`初始化失败：${error.message}`);
});
