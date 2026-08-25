const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const state = {
  file: null,
  objectUrl: null,
  duration: 0,
  start: 0,
  end: 0,
  previewingSelection: false,
  toastTimer: null,
  analysis: null,
  candidateIndex: 0,
  selectedSubject: null,
  analysisAbortController: null,
  tracking: null,
  trackingAbortController: null,
  correctionFrame: null,
  correctionSelection: null,
  correctionCount: 0,
  pendingCorrections: [],
  previewResult: null,
  exportResult: null,
  renderAbortController: null,
};

const video = $("#video");
const fileInput = $("#video-input");
const startRange = $("#start-range");
const endRange = $("#end-range");
const continueButton = $("#continue-button");
const playSelectionButton = $("#play-selection-button");
const setStartButton = $("#set-start-button");
const setEndButton = $("#set-end-button");

function formatTime(seconds, showTenths = true) {
  const safeSeconds = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const minutes = Math.floor(safeSeconds / 60);
  const remainder = safeSeconds - minutes * 60;
  const paddedMinutes = String(minutes).padStart(2, "0");
  const paddedSeconds = showTenths
    ? remainder.toFixed(1).padStart(4, "0")
    : String(Math.floor(remainder)).padStart(2, "0");
  return `${paddedMinutes}:${paddedSeconds}`;
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("is-visible");
  window.clearTimeout(state.toastTimer);
  state.toastTimer = window.setTimeout(() => toast.classList.remove("is-visible"), 2600);
}

function setControlsEnabled(enabled) {
  [startRange, endRange, continueButton, playSelectionButton, setStartButton, setEndButton]
    .forEach((element) => { element.disabled = !enabled; });
}

function renderRange() {
  const duration = state.duration || 1;
  const startPercent = (state.start / duration) * 100;
  const endPercent = (state.end / duration) * 100;
  $("#mask-left").style.width = `${startPercent}%`;
  $("#mask-right").style.width = `${100 - endPercent}%`;
  $("#selection-outline").style.left = `${startPercent}%`;
  $("#selection-outline").style.right = `${100 - endPercent}%`;
  $("#range-start").textContent = formatTime(state.start);
  $("#range-end").textContent = formatTime(state.end);
  $("#selected-duration").textContent = formatTime(state.end - state.start);
}

function syncRangeInputs(source) {
  const minGap = Math.min(0.35, Math.max(0.05, state.duration * 0.005));
  let start = Number(startRange.value);
  let end = Number(endRange.value);

  if (source === "start" && start > end - minGap) {
    start = Math.max(0, end - minGap);
    startRange.value = String(start);
  }
  if (source === "end" && end < start + minGap) {
    end = Math.min(state.duration, start + minGap);
    endRange.value = String(end);
  }

  state.start = start;
  state.end = end;
  video.currentTime = source === "start" ? start : end;
  renderRange();
}

function updatePlayhead() {
  if (!state.duration) return;
  const percent = Math.min(100, Math.max(0, (video.currentTime / state.duration) * 100));
  const playhead = $("#playhead");
  playhead.style.left = `${percent}%`;
  playhead.style.opacity = "1";
  $("#current-time").textContent = formatTime(video.currentTime);

  if (state.previewingSelection && video.currentTime >= state.end - 0.03) {
    video.pause();
    video.currentTime = state.start;
    state.previewingSelection = false;
    playSelectionButton.innerHTML = "<span>▶</span> 预览保留片段";
  }
}

function waitForEvent(element, eventName, timeoutMs = 10000) {
  return new Promise((resolve, reject) => {
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error(`Video ${eventName} timed out`));
    }, timeoutMs);
    const onEvent = () => { cleanup(); resolve(); };
    const onError = () => { cleanup(); reject(new Error(`Video ${eventName} failed`)); };
    const cleanup = () => {
      window.clearTimeout(timer);
      element.removeEventListener(eventName, onEvent);
      element.removeEventListener("error", onError);
    };
    element.addEventListener(eventName, onEvent, { once: true });
    element.addEventListener("error", onError, { once: true });
  });
}

async function seekTo(media, time) {
  const clamped = Math.min(Math.max(time, 0), Math.max(0, media.duration - 0.02));
  if (Math.abs(media.currentTime - clamped) < 0.01 && media.readyState >= 2) return;
  const sought = waitForEvent(media, "seeked");
  media.currentTime = clamped;
  await sought;
}

function createCanvasForVideo(sourceVideo, width = 320) {
  const sourceWidth = sourceVideo.videoWidth || 16;
  const sourceHeight = sourceVideo.videoHeight || 9;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = Math.max(1, Math.round(width * sourceHeight / sourceWidth));
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(sourceVideo, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function createSamplingVideo() {
  const samplingVideo = document.createElement("video");
  samplingVideo.src = state.objectUrl;
  samplingVideo.muted = true;
  samplingVideo.playsInline = true;
  samplingVideo.preload = "auto";
  samplingVideo.load();
  if (samplingVideo.readyState < 1) await waitForEvent(samplingVideo, "loadedmetadata");
  return samplingVideo;
}

async function generateFilmstrip() {
  const filmstrip = $("#filmstrip");
  filmstrip.innerHTML = '<div class="filmstrip-empty">正在生成画面缩略图…</div>';

  try {
    const samplingVideo = await createSamplingVideo();
    const count = window.innerWidth < 720 ? 7 : 12;
    const fragment = document.createDocumentFragment();

    for (let index = 0; index < count; index += 1) {
      const time = state.duration * ((index + 0.5) / count);
      await seekTo(samplingVideo, time);
      const canvas = createCanvasForVideo(samplingVideo, 190);
      const image = document.createElement("img");
      image.src = canvas.toDataURL("image/jpeg", 0.64);
      image.alt = "";
      fragment.appendChild(image);
    }

    filmstrip.innerHTML = "";
    filmstrip.appendChild(fragment);
    samplingVideo.removeAttribute("src");
    samplingVideo.load();
  } catch (error) {
    filmstrip.innerHTML = '<div class="filmstrip-empty">无法生成缩略图，但仍可拖动时间轴裁剪</div>';
    console.error(error);
  }
}

function isSupportedVideo(file) {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return file.type.startsWith("video/") || ["mp4", "mov", "mkv"].includes(extension);
}

function resetObjectUrl() {
  if (state.objectUrl?.startsWith("blob:")) URL.revokeObjectURL(state.objectUrl);
  state.objectUrl = null;
}

async function loadVideoFile(file, sourceUrl = null) {
  if (!file || !isSupportedVideo(file)) {
    showToast("请选择 MP4、MOV 或 MKV 视频");
    return false;
  }

  video.pause();
  resetObjectUrl();
  state.file = file;
  state.objectUrl = sourceUrl || URL.createObjectURL(file);
  video.src = state.objectUrl;
  video.load();

  try {
    if (video.readyState < 1) await waitForEvent(video, "loadedmetadata");
    if (!Number.isFinite(video.duration) || video.duration <= 0) throw new Error("Invalid duration");

    state.duration = video.duration;
    state.start = 0;
    state.end = video.duration;
    state.analysis = null;
    state.selectedSubject = null;
    state.tracking = null;
    state.correctionCount = 0;
    state.pendingCorrections = [];
    state.previewResult = null;
    state.exportResult = null;
    startRange.max = String(video.duration);
    endRange.max = String(video.duration);
    startRange.value = "0";
    endRange.value = String(video.duration);
    $("#empty-state").hidden = true;
    video.hidden = false;
    $("#video-corner-label").hidden = false;
    $("#video-file-meta").textContent = `${file.name} · ${formatFileSize(file.size)}`;
    $("#full-duration").textContent = formatTime(video.duration, false);
    setControlsEnabled(true);
    renderRange();
    updatePlayhead();
    generateFilmstrip();
    showToast("视频已载入，可以拖动两端裁剪");
    return true;
  } catch (error) {
    console.error(error);
    showToast("浏览器无法读取这个视频，请尝试 MP4（H.264）格式");
    setControlsEnabled(false);
    return false;
  }
}

const pipelineNumbers = {
  "#frame-pipeline": "02",
  "#detection-pipeline": "03",
  "#selection-pipeline": "04",
};

function setPipelineStatus(selector, status, detail) {
  const row = $(selector);
  row.classList.remove("is-next", "is-running", "is-done");
  row.classList.add(`is-${status}`);
  row.querySelector("span").textContent = status === "done" ? "✓" : pipelineNumbers[selector];
  if (detail) row.querySelector("small").textContent = detail;
}

function resetSubjectSelection() {
  state.selectedSubject = null;
  state.pendingCorrections = [];
  $$(".person-box, .person-marker").forEach((element) => element.classList.remove("is-selected"));
  $("#selection-summary").hidden = true;
  $("#confirm-subject-button").disabled = true;
  $("#confirm-subject-button").innerHTML = "确认选择这个人 <span>→</span>";
  setPipelineStatus("#selection-pipeline", "running", "点击人物编号圆点；漏检时直接点击人物躯干");
}

function resetAnalysisUI() {
  state.analysis = null;
  state.candidateIndex = 0;
  state.selectedSubject = null;
  state.pendingCorrections = [];
  $("#frame-loading").hidden = false;
  $("#analysis-error").hidden = true;
  $("#candidate-toolbar").hidden = true;
  $("#analysis-metrics").hidden = true;
  $("#selection-summary").hidden = true;
  $("#representative-frame").removeAttribute("src");
  $("#person-overlay").replaceChildren();
  $("#confirm-subject-button").disabled = true;
  $("#confirm-subject-button").innerHTML = "确认选择这个人 <span>→</span>";
  $("#frame-status").textContent = "正在上传视频并抽取候选画面…";
  $("#frame-status-detail").textContent = "视频只交给本机 127.0.0.1 处理";
  $("#detector-name").textContent = "正在连接人物检测服务…";
  setPipelineStatus("#frame-pipeline", "running", "在保留区间抽取 9 张候选画面");
  setPipelineStatus("#detection-pipeline", "next", "等待候选画面抽取");
  setPipelineStatus("#selection-pipeline", "next", "等待人物检测结果");
}

function showAnalysisError(title, hint) {
  $("#frame-loading").hidden = true;
  $("#analysis-error").hidden = false;
  $("#analysis-error-title").textContent = title || "人物分析失败";
  $("#analysis-error-hint").textContent = hint || "请返回第一步检查视频后重试。";
  $("#detector-name").textContent = "本地检测服务未完成分析";
}

function positionPersonOverlay() {
  if (!state.analysis) return;
  const candidate = state.analysis.candidates[state.candidateIndex];
  const frameStage = $("#frame-stage");
  const image = $("#representative-frame");
  const overlay = $("#person-overlay");
  if (!candidate || !image.complete || !image.naturalWidth) return;

  const stageWidth = frameStage.clientWidth;
  const stageHeight = frameStage.clientHeight;
  const scale = Math.min(stageWidth / candidate.width, stageHeight / candidate.height);
  const displayWidth = candidate.width * scale;
  const displayHeight = candidate.height * scale;
  overlay.style.left = `${(stageWidth - displayWidth) / 2}px`;
  overlay.style.top = `${(stageHeight - displayHeight) / 2}px`;
  overlay.style.width = `${displayWidth}px`;
  overlay.style.height = `${displayHeight}px`;
}

function selectSubject(candidate, box, clickPoint) {
  state.correctionCount = 0;
  state.pendingCorrections = [];
  state.selectedSubject = {
    analysisId: state.analysis.analysis_id,
    candidateId: candidate.id,
    timestamp: candidate.time,
    sourceWidth: candidate.width,
    sourceHeight: candidate.height,
    box: { ...box },
    label: box.label,
    detectionConfidence: box.confidence,
    click: { x: Number(clickPoint.x.toFixed(2)), y: Number(clickPoint.y.toFixed(2)) },
  };

  $$(".person-box, .person-marker").forEach((element) => {
    element.classList.toggle("is-selected", element.dataset.personId === box.id);
  });
  $("#selected-person-index").textContent = box.manual
    ? "手动"
    : String(box.id.split("-").pop()).padStart(2, "0");
  $("#selected-person-label").textContent = box.label;
  $("#selected-person-confidence").textContent = box.manual
    ? "人工点选锚点 · 不依赖人物检测结果"
    : `检测置信度 ${(box.confidence * 100).toFixed(0)}% · 已保存原画坐标`;
  $("#selection-summary").hidden = false;
  $("#confirm-subject-button").disabled = false;
  setPipelineStatus("#selection-pipeline", "done", `${box.label} 已成为身份锚点`);
  showToast(`已选择${box.label}，请确认后进入跟踪阶段`);
}

function candidateMarkerPoint(box) {
  const height = Math.max(1, box.y2 - box.y1);
  return { x: box.cx, y: box.y1 + height * 0.46 };
}

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function manualPersonBox(frame, point) {
  const likelyPeople = (frame.candidates || frame.boxes || []).filter((box) => {
    const boxHeight = box.y2 - box.y1;
    const boxWidth = box.x2 - box.x1;
    return boxHeight >= frame.height * 0.10 && boxHeight <= frame.height * 0.42
      && boxWidth >= frame.width * 0.018 && boxWidth <= frame.width * 0.18;
  });
  const typicalWidth = median(likelyPeople.map((box) => box.x2 - box.x1)) || frame.width * 0.055;
  const typicalHeight = median(likelyPeople.map((box) => box.y2 - box.y1)) || frame.height * 0.25;
  const boxWidth = Math.max(frame.width * 0.038, Math.min(frame.width * 0.12, typicalWidth));
  const boxHeight = Math.max(frame.height * 0.18, Math.min(frame.height * 0.38, typicalHeight));
  const x1 = Math.max(0, Math.min(frame.width - boxWidth, point.x - boxWidth / 2));
  const y1 = Math.max(0, Math.min(frame.height - boxHeight, point.y - boxHeight * 0.46));
  const box = {
    id: `manual-${Number(frame.time || 0).toFixed(3)}`,
    label: "手动点选人物",
    x1,
    y1,
    x2: x1 + boxWidth,
    y2: y1 + boxHeight,
    cx: x1 + boxWidth / 2,
    cy: y1 + boxHeight / 2,
    confidence: 1,
    performer_score: 1,
    role: "performer",
    manual: true,
  };
  box.selection_box = { ...box };
  return box;
}

function sourcePointFromEvent(event, overlay, frame) {
  const rect = overlay.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(frame.width, ((event.clientX - rect.left) / rect.width) * frame.width)),
    y: Math.max(0, Math.min(frame.height, ((event.clientY - rect.top) / rect.height) * frame.height)),
  };
}

function createCandidateVisual(frame, box, index, onSelect) {
  const fragment = document.createDocumentFragment();
  const displayBox = box.selection_box || box;
  const outline = document.createElement("div");
  outline.className = `person-box ${box.role === "performer" ? "is-performer" : "is-context"}`;
  outline.dataset.personId = box.id;
  outline.style.left = `${(displayBox.x1 / frame.width) * 100}%`;
  outline.style.top = `${(displayBox.y1 / frame.height) * 100}%`;
  outline.style.width = `${((displayBox.x2 - displayBox.x1) / frame.width) * 100}%`;
  outline.style.height = `${((displayBox.y2 - displayBox.y1) / frame.height) * 100}%`;

  const markerPoint = candidateMarkerPoint(box);
  const marker = document.createElement("button");
  marker.type = "button";
  marker.className = `person-marker ${box.role === "performer" ? "is-performer" : "is-context"}`;
  marker.dataset.personId = box.id;
  marker.style.left = `${(markerPoint.x / frame.width) * 100}%`;
  marker.style.top = `${(markerPoint.y / frame.height) * 100}%`;
  marker.textContent = String(index + 1).padStart(2, "0");
  marker.setAttribute("aria-label", `选择${box.label}`);
  marker.title = `${box.label} · ${box.role === "performer" ? "主舞候选" : "周边人物"}`;
  marker.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    onSelect(box, markerPoint);
  });
  fragment.append(outline, marker);
  return fragment;
}

function drawPersonBoxes(candidate) {
  const overlay = $("#person-overlay");
  overlay.replaceChildren();
  candidate.boxes.forEach((box, index) => {
    overlay.appendChild(createCandidateVisual(
      candidate,
      box,
      index,
      (selected, point) => selectSubject(candidate, selected, point),
    ));
  });
  overlay.onclick = (event) => {
    const point = sourcePointFromEvent(event, overlay, candidate);
    // Numbered dots are the explicit detector-candidate controls.  Any click
    // on the image itself is an explicit manual prompt, even when a bad giant
    // detector box happens to cover the intended person.
    const manual = manualPersonBox(candidate, point);
    candidate.boxes = [...candidate.boxes.filter((box) => !box.manual), manual];
    drawPersonBoxes(candidate);
    selectSubject(candidate, manual, point);
    showToast("已按你的点击建立人工身份锚点；检测大框不会抢走这次选择");
  };
}

function renderCandidate(index) {
  const candidates = state.analysis?.candidates || [];
  if (!candidates.length) return;
  state.candidateIndex = (index + candidates.length) % candidates.length;
  const candidate = candidates[state.candidateIndex];
  const image = $("#representative-frame");

  resetSubjectSelection();
  image.onload = positionPersonOverlay;
  image.src = candidate.image;
  image.dataset.time = String(candidate.time);
  drawPersonBoxes(candidate);
  $("#candidate-caption").textContent = `候选画面 ${state.candidateIndex + 1} / ${candidates.length}`;
  $("#candidate-reason").textContent = `${candidate.performer_count ?? candidate.people_count} 名主舞候选 · ${candidate.people_count} 人入镜`;
  $("#metric-people").textContent = String(candidate.performer_count ?? candidate.people_count);
  $("#metric-time").textContent = formatTime(candidate.time);

  const dots = $("#candidate-dots");
  dots.replaceChildren();
  candidates.forEach((_, dotIndex) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-dot";
    button.classList.toggle("is-active", dotIndex === state.candidateIndex);
    button.setAttribute("aria-label", `查看候选画面 ${dotIndex + 1}`);
    button.addEventListener("click", () => renderCandidate(dotIndex));
    dots.appendChild(button);
  });
  positionPersonOverlay();
}

async function analyzePeople() {
  if (!state.file) return;
  state.analysisAbortController?.abort();
  state.analysisAbortController = new AbortController();
  resetAnalysisUI();

  if (window.location.protocol === "file:") {
    showAnalysisError(
      "AI 服务尚未启动",
      "请关闭当前页面，在项目文件夹双击“启动一键直拍.cmd”。人物检测必须通过 http://127.0.0.1:4173 运行。",
    );
    return;
  }

  try {
    const form = new FormData();
    form.append("video", state.file, state.file.name || "video.mp4");
    form.append("start", String(state.start));
    form.append("end", String(state.end));
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      signal: state.analysisAbortController.signal,
    });
    const result = await response.json();
    if (!response.ok) {
      const failure = new Error(result.error?.message || "人物分析失败");
      failure.hint = result.error?.hint;
      throw failure;
    }
    if (!result.candidates?.length) throw new Error("没有返回可选择的候选画面");

    state.analysis = result;
    $("#frame-loading").hidden = true;
    $("#candidate-toolbar").hidden = false;
    $("#analysis-metrics").hidden = false;
    $("#metric-elapsed").textContent = `${Number(result.elapsed_seconds || 0).toFixed(1)}s`;
    $("#detector-name").textContent = `${result.detector.detector} · 本机 CPU`;
    setPipelineStatus("#frame-pipeline", "done", `比较了 ${result.source.sample_count} 张候选画面`);
    const expectedPerformers = result.scene_context?.expected_performers;
    setPipelineStatus(
      "#detection-pipeline",
      "done",
      expectedPerformers
        ? `整段推断约 ${expectedPerformers} 名主舞；周边人物已弱化但仍可点击`
        : "已检测人物并生成可点击圆点",
    );
    setPipelineStatus("#selection-pipeline", "running", "点击人物编号圆点；漏检时直接点击人物躯干");
    renderCandidate(0);
    return result;
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    showAnalysisError(error.message, error.hint || "请确认 AI 能力已安装，并保持启动窗口开启。" );
  }
}

function resetTrackingUI() {
  state.tracking = null;
  state.previewResult = null;
  state.exportResult = null;
  $("#tracking-loading").hidden = false;
  $("#tracking-error").hidden = true;
  $("#tracking-qa-grid").hidden = true;
  $("#tracking-qa-grid").replaceChildren();
  state.correctionFrame = null;
  state.correctionSelection = null;
  $("#tracking-correction-editor").hidden = true;
  $("#tracking-correction-overlay").replaceChildren();
  $("#confirm-tracking-correction").disabled = true;
  $("#pending-corrections-panel").hidden = true;
  $("#tracking-status").textContent = "正在从身份锚点向前、向后跟踪…";
  $("#tracking-status-detail").textContent = "补全身份区域，并比较整段视频中的多条候选路径";
  $("#tracking-metric-coverage").textContent = "—";
  $("#tracking-metric-confidence").textContent = "—";
  $("#tracking-metric-samples").textContent = "—";
  $("#tracking-reliability-fill").style.width = "0%";
  $("#tracking-reliability-label").textContent = "等待分析";
  $("#tracking-review-list").innerHTML = "<p>跟踪完成后显示遮挡或身份不确定区间。</p>";
  $("#tracking-engine").textContent = "正在启动全片双向身份路径…";
  $("#continue-camera-button").disabled = true;
}

function showTrackingError(title, hint) {
  $("#tracking-loading").hidden = true;
  $("#tracking-error").hidden = false;
  $("#tracking-error-title").textContent = title || "身份跟踪失败";
  $("#tracking-error-hint").textContent = hint || "请返回上一阶段重新选择目标人物。";
  $("#tracking-engine").textContent = "本地跟踪服务未完成分析";
  renderPendingCorrections();
}

function positionTrackingCorrectionOverlay() {
  const frame = state.correctionFrame;
  const stage = $("#tracking-correction-frame");
  const image = $("#tracking-correction-image");
  const overlay = $("#tracking-correction-overlay");
  if (!frame || !image.complete || !image.naturalWidth) return;
  const stageRect = stage.getBoundingClientRect();
  const imageRect = image.getBoundingClientRect();
  overlay.style.left = `${imageRect.left - stageRect.left}px`;
  overlay.style.top = `${imageRect.top - stageRect.top}px`;
  overlay.style.width = `${imageRect.width}px`;
  overlay.style.height = `${imageRect.height}px`;
}

function selectTrackingCorrection(box) {
  const frame = state.correctionFrame;
  if (!box.manual && frame && (frame.candidates || []).some((candidate) => candidate.manual)) {
    frame.candidates = frame.candidates.filter((candidate) => !candidate.manual);
    renderTrackingCorrectionBoxes(frame);
  }
  state.correctionSelection = box;
  $$("#tracking-correction-overlay .person-box, #tracking-correction-overlay .person-marker").forEach((element) => {
    element.classList.toggle("is-selected", element.dataset.personId === box.id);
  });
  const roleLabel = box.manual ? "手动点选人物" : (box.role === "performer" ? "主舞候选" : "周边人物");
  $("#tracking-correction-selection").textContent = box.manual
    ? "已建立人工身份锚点 · 不依赖这一帧的检测结果"
    : `已选择${roleLabel} · 检测置信度 ${Math.round((box.confidence || 0) * 100)}%`;
  $("#reset-tracking-correction").disabled = false;
  $("#confirm-tracking-correction").disabled = false;
}

function resetTrackingCorrectionSelection() {
  const frame = state.correctionFrame;
  const selected = state.correctionSelection;
  state.correctionSelection = null;
  if (frame && selected?.manual) {
    frame.candidates = (frame.candidates || []).filter((box) => box.id !== selected.id);
    renderTrackingCorrectionBoxes(frame);
  } else {
    $$("#tracking-correction-overlay .person-box, #tracking-correction-overlay .person-marker").forEach(
      (element) => element.classList.remove("is-selected"),
    );
  }
  $("#tracking-correction-selection").textContent = "本次点选已撤销，可重新点击编号圆点或人物躯干";
  $("#reset-tracking-correction").disabled = true;
  $("#confirm-tracking-correction").disabled = true;
}

function renderTrackingCorrectionBoxes(frame) {
  const overlay = $("#tracking-correction-overlay");
  overlay.replaceChildren();
  (frame.candidates || []).forEach((box, index) => {
    overlay.appendChild(createCandidateVisual(
      frame,
      box,
      index,
      (selected) => selectTrackingCorrection(selected),
    ));
  });
  overlay.onclick = (event) => {
    const point = sourcePointFromEvent(event, overlay, frame);
    // Only a numbered dot selects a detector candidate.  Clicking anywhere
    // else always creates a manual prompt, so a huge foreground bbox cannot
    // make an undetected person impossible to choose.
    const manual = manualPersonBox(frame, point);
    frame.candidates = [...(frame.candidates || []).filter((box) => !box.manual), manual];
    renderTrackingCorrectionBoxes(frame);
    selectTrackingCorrection(manual);
    showToast("已按躯干位置建立人工纠偏锚点；如误点可立即撤销");
  };
}

function openTrackingCorrection(frame) {
  if (!frame.correction_image) {
    showToast("这一帧没有可用于纠偏的原始画面");
    return;
  }
  state.correctionFrame = frame;
  state.correctionSelection = null;
  $("#tracking-correction-time").textContent = formatTime(frame.time);
  $("#tracking-correction-selection").textContent = "点击编号圆点；如果没有正确框，直接点击人物躯干";
  $("#reset-tracking-correction").disabled = true;
  $("#confirm-tracking-correction").disabled = true;
  const image = $("#tracking-correction-image");
  image.onload = positionTrackingCorrectionOverlay;
  image.src = frame.correction_image;
  renderTrackingCorrectionBoxes(frame);
  $("#tracking-correction-editor").hidden = false;
  window.setTimeout(() => {
    positionTrackingCorrectionOverlay();
    $("#tracking-correction-editor").scrollIntoView({ behavior: "smooth", block: "center" });
  }, 30);
}

function closeTrackingCorrection() {
  const frame = state.correctionFrame;
  const selected = state.correctionSelection;
  if (frame && selected?.manual) {
    frame.candidates = (frame.candidates || []).filter((box) => box.id !== selected.id);
  }
  state.correctionFrame = null;
  state.correctionSelection = null;
  $("#tracking-correction-editor").hidden = true;
  $("#tracking-correction-overlay").replaceChildren();
  $("#reset-tracking-correction").disabled = true;
  $("#confirm-tracking-correction").disabled = true;
}

function correctionTimeKey(timestamp) {
  return Number(timestamp).toFixed(3);
}

function renderPendingCorrections() {
  const panel = $("#pending-corrections-panel");
  const list = $("#pending-corrections-list");
  const pending = state.pendingCorrections || [];
  panel.hidden = pending.length === 0;
  $("#pending-corrections-count").textContent = String(pending.length);
  $("#apply-tracking-corrections").textContent = pending.length
    ? `统一应用 ${pending.length} 个纠偏点并重算`
    : "统一应用并重算轨迹";
  list.replaceChildren();
  pending.forEach((correction) => {
    const chip = document.createElement("span");
    chip.className = "pending-correction-chip";
    chip.append(document.createTextNode(`${formatTime(correction.timestamp)} · 已确认人物 `));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.title = "移除这个纠偏点";
    remove.textContent = "×";
    remove.addEventListener("click", () => {
      const key = correctionTimeKey(correction.timestamp);
      state.pendingCorrections = state.pendingCorrections.filter(
        (item) => correctionTimeKey(item.timestamp) !== key,
      );
      renderPendingCorrections();
    });
    chip.appendChild(remove);
    list.appendChild(chip);
  });
  $$(".tracking-qa-card[data-correction-time]").forEach((card) => {
    const hasPending = pending.some(
      (item) => correctionTimeKey(item.timestamp) === card.dataset.correctionTime,
    );
    card.classList.toggle("has-pending-correction", hasPending);
    const button = card.querySelector(".tracking-correct-button");
    if (button && !button.disabled) {
      button.textContent = hasPending ? "已暂存纠偏，可重新选择" : "这帧不对？重新选人";
    }
  });
}

function confirmTrackingCorrection() {
  const frame = state.correctionFrame;
  const box = state.correctionSelection;
  if (!frame || !box) return;
  state.correctionCount += 1;
  const correction = {
    analysisId: state.selectedSubject?.analysisId || state.analysis?.analysis_id,
    candidateId: `correction-${state.correctionCount}`,
    timestamp: frame.time,
    sourceWidth: frame.width,
    sourceHeight: frame.height,
    box: { ...box },
    label: `人工纠正锚点 ${state.correctionCount}`,
    detectionConfidence: box.confidence || 0,
    click: { x: box.cx, y: box.cy },
  };
  const key = correctionTimeKey(frame.time);
  const existingIndex = state.pendingCorrections.findIndex(
    (item) => correctionTimeKey(item.timestamp) === key,
  );
  if (existingIndex >= 0) state.pendingCorrections.splice(existingIndex, 1, correction);
  else state.pendingCorrections.push(correction);
  state.pendingCorrections.sort((first, second) => first.timestamp - second.timestamp);
  closeTrackingCorrection();
  renderPendingCorrections();
  showToast(`已暂存 ${formatTime(frame.time)} 的正确人物，可继续纠偏其他画面`);
}

function clearPendingCorrections() {
  state.pendingCorrections = [];
  renderPendingCorrections();
  showToast("已清空暂存的纠偏点");
}

function applyPendingCorrections() {
  if (!state.pendingCorrections.length || !state.tracking?.tracking_id) return;
  const corrections = state.pendingCorrections.map((item) => ({ ...item, box: { ...item.box } }));
  showToast(`正在统一应用 ${corrections.length} 个确认锚点，只重算一次轨迹`);
  return startTracking({
    isCorrection: true,
    reuseTrackingId: state.tracking.tracking_id,
    corrections,
  });
}

function renderTrackingResult(result) {
  state.tracking = result;
  const metrics = result.metrics;
  $("#tracking-loading").hidden = true;
  $("#tracking-error").hidden = true;
  $("#tracking-qa-grid").hidden = false;
  $("#tracking-metric-coverage").textContent = `${Math.round(metrics.coverage * 100)}%`;
  $("#tracking-metric-confidence").textContent = `${Math.round(metrics.average_confidence * 100)}%`;
  $("#tracking-metric-samples").textContent = String(metrics.samples);
  const reliablePercent = Math.round(metrics.reliable_ratio * 100);
  $("#tracking-reliability-fill").style.width = `${reliablePercent}%`;
  const reliabilityText = reliablePercent >= 80 ? "稳定" : reliablePercent >= 55 ? "建议复核" : "风险较高";
  $("#tracking-reliability-label").textContent = `${reliablePercent}% · ${reliabilityText}`;
  $("#tracking-engine").textContent = `${result.engine} · ${Number(result.elapsed_seconds || 0).toFixed(1)}s`;

  const grid = $("#tracking-qa-grid");
  grid.replaceChildren();
  const correctionSummary = result.correction_summary;
  if (correctionSummary?.applied_count) {
    const banner = document.createElement("section");
    banner.className = `correction-result-banner ${correctionSummary.all_hard_anchors_applied ? "is-applied" : "is-warning"}`;
    const title = document.createElement("strong");
    title.textContent = correctionSummary.all_hard_anchors_applied
      ? `${correctionSummary.applied_count} 个人工锚点已生效`
      : "有人工锚点未能应用";
    const detail = document.createElement("span");
    detail.textContent = correctionSummary.recomputed_full_track
      ? "高精度模式已把这些锚点作为同一个人物的条件帧，重新传播整段对象轨迹；锚点帧以“身份锚点”显示。"
      : correctionSummary.changed_samples
        ? `新轨迹改变了 ${correctionSummary.changed_samples} / ${metrics.samples} 个采样点；人工锚点帧以“身份锚点”显示。`
        : "人工锚点与上一版轨迹基本重合，因此总体置信指标可能不变；锚点帧仍会标为“身份锚点”。";
    banner.append(title, detail);
    grid.appendChild(banner);
  }
  const statusLabels = {
    anchor: "身份锚点",
    tracked: "稳定",
    low_confidence: "需复核",
    missing: "预测位置",
  };
  result.qa_frames.forEach((frame) => {
    const card = document.createElement("article");
    card.className = "tracking-qa-card";
    card.dataset.correctionTime = correctionTimeKey(frame.time);
    card.classList.toggle("is-review", frame.requires_review);
    const image = document.createElement("img");
    image.src = frame.image;
    image.alt = `${formatTime(frame.time)} 的身份跟踪复核画面`;
    const footer = document.createElement("footer");
    const time = document.createElement("strong");
    time.textContent = formatTime(frame.time);
    const badge = document.createElement("span");
    badge.textContent = frame.status === "anchor"
      ? statusLabels.anchor
      : `${statusLabels[frame.status] || "跟踪"} · ${Math.round(frame.confidence * 100)}%`;
    footer.append(time, badge);
    const correctButton = document.createElement("button");
    correctButton.type = "button";
    correctButton.className = "tracking-correct-button";
    correctButton.textContent = "这帧不对？重新选人";
    correctButton.disabled = !frame.correction_image;
    correctButton.addEventListener("click", () => openTrackingCorrection(frame));
    card.append(image, footer, correctButton);
    grid.appendChild(card);
  });
  renderPendingCorrections();

  const reviewList = $("#tracking-review-list");
  reviewList.replaceChildren();
  if (!result.low_confidence_ranges.length) {
    const paragraph = document.createElement("p");
    paragraph.textContent = "未发现连续的低置信或多人重叠区间。";
    reviewList.appendChild(paragraph);
  } else {
    result.low_confidence_ranges.forEach((range) => {
      const row = document.createElement("div");
      row.className = "tracking-review-item";
      const reason = document.createElement("span");
      reason.textContent = range.reason;
      const time = document.createElement("strong");
      time.textContent = `${formatTime(range.start)} — ${formatTime(range.end)}`;
      row.append(reason, time);
      reviewList.appendChild(row);
    });
  }
  if (metrics.reused_detections) {
    const changed = correctionSummary?.changed_samples || 0;
    showToast(`已应用 ${metrics.correction_anchor_count || 0} 个纠偏锚点，并更新 ${changed} 个轨迹采样点`);
  } else {
    showToast(metrics.requires_review
      ? "身份轨迹已生成，请重点检查橙色画面"
      : "身份轨迹已生成，抽样画面未发现明显风险");
  }
  $("#continue-camera-button").disabled = false;
}

async function startTracking(options = {}) {
  if (!state.file || !state.selectedSubject) return;
  const isCorrection = options?.isCorrection === true;
  const previousTracking = isCorrection ? state.tracking : null;
  state.trackingAbortController?.abort();
  state.trackingAbortController = new AbortController();
  $("#select-stage").hidden = true;
  $("#track-stage").hidden = false;
  $("#tracking-subject-label").textContent = state.selectedSubject.label || "已选择人物";
  $("#tracking-subject-time").textContent = `锚点画面 ${formatTime(state.selectedSubject.timestamp)}`;
  setStep(3);
  resetTrackingUI();
  if (isCorrection) {
    $("#tracking-status").textContent = `正在统一应用 ${options.corrections?.length || 0} 个纠偏点…`;
    $("#tracking-status-detail").textContent = "复用人物检测缓存，在相邻确认锚点之间做分段双向修正";
  }
  window.scrollTo({ top: 0, behavior: "smooth" });

  if (window.location.protocol === "file:") {
    showTrackingError("AI 服务尚未启动", "请通过“启动一键直拍.cmd”打开本地工作台后重试。");
    return;
  }

  const statusTimer = window.setTimeout(() => {
    $("#tracking-status").textContent = "正在处理保留片段中的人物轨迹…";
    $("#tracking-status-detail").textContent = "片段越长、画面人物越多，分析时间越长";
  }, 1800);
  try {
    const form = new FormData();
    form.append("video", state.file, state.file.name || "video.mp4");
    form.append("start", String(state.start));
    form.append("end", String(state.end));
    form.append("selection", JSON.stringify(state.selectedSubject));
    form.append("tracker_mode", "auto");
    if (isCorrection && options.reuseTrackingId) {
      form.append("reuse_tracking_id", options.reuseTrackingId);
      form.append("corrections", JSON.stringify(options.corrections || []));
    }
    const response = await fetch("/api/track", {
      method: "POST",
      body: form,
      signal: state.trackingAbortController.signal,
    });
    const result = await response.json();
    if (!response.ok) {
      const failure = new Error(result.error?.message || "身份跟踪失败");
      failure.hint = result.error?.hint;
      throw failure;
    }
    if (!result.keyframes?.length || !result.qa_frames?.length) {
      throw new Error("跟踪服务没有返回可复核的身份轨迹");
    }
    if (isCorrection) state.pendingCorrections = [];
    renderTrackingResult(result);
    return result;
  } catch (error) {
    if (error.name === "AbortError") {
      if (isCorrection && previousTracking) state.tracking = previousTracking;
      return;
    }
    console.error(error);
    if (isCorrection && previousTracking) state.tracking = previousTracking;
    showTrackingError(error.message, error.hint || "请返回上一阶段换一张人物更清晰的候选画面。" );
  } finally {
    window.clearTimeout(statusTimer);
  }
}

function showSelectionFromTracking() {
  state.trackingAbortController?.abort();
  $("#track-stage").hidden = true;
  $("#select-stage").hidden = false;
  setStep(2);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function currentCameraSettings() {
  return {
    outputMode: $("input[name='output-mode']:checked")?.value || "vertical",
    framing: $("input[name='framing']:checked")?.value || "standard",
    motion: $("input[name='motion']:checked")?.value || "balanced",
  };
}

function isHorizontalMode(settings = currentCameraSettings()) {
  return settings.outputMode === "horizontal-focus";
}

function setExportButtonLabel(settings = currentCameraSettings()) {
  $("#export-video-button").innerHTML = isHorizontalMode(settings)
    ? "导出高清横屏 MP4 <span>↓</span>"
    : "导出高清竖屏 MP4 <span>↓</span>";
}

function syncOutputModeUI(settings = currentCameraSettings()) {
  const horizontal = isHorizontalMode(settings);
  $("#vertical-camera-controls").hidden = false;
  $("#horizontal-focus-note").hidden = !horizontal;
  $("#render-preview-shell").classList.toggle("is-horizontal", horizontal);
  $("#render-preview-eyebrow").textContent = horizontal ? "横屏直拍预览" : "竖屏直拍预览";
  $("#render-settings-title").textContent = horizontal
    ? "保持横屏比例，适度放大并柔焦人物左右两侧"
    : "人物完整优先，再适度放大";
  $("#framing-wide-title").textContent = horizontal ? "原景宽松" : "全身宽松";
  $("#framing-wide-detail").textContent = horizontal ? "1.00×，保留完整场景" : "动作幅度大时使用";
  $("#framing-standard-title").textContent = horizontal ? "适度放大" : "标准直拍";
  $("#framing-standard-detail").textContent = horizontal ? "约 1.15×，兼顾动作与呼吸感" : "人物约占 2/3，脚底约留 8%";
  $("#framing-close-title").textContent = horizontal ? "重点突出" : "近景突出";
  $("#framing-close-detail").textContent = horizontal ? "约 1.30×，人物更加醒目" : "完整身体范围内放大";
  setExportButtonLabel(settings);
}

function compactTrackingPayload() {
  return {
    tracking_id: state.tracking.tracking_id,
    source: state.tracking.source,
    anchor: state.tracking.anchor,
    sample_fps: state.tracking.sample_fps,
    keyframes: state.tracking.keyframes,
  };
}

function resetRenderUI() {
  const settings = currentCameraSettings();
  const horizontal = isHorizontalMode(settings);
  syncOutputModeUI(settings);
  state.previewResult = null;
  state.exportResult = null;
  const preview = $("#render-preview-video");
  preview.pause();
  preview.removeAttribute("src");
  preview.removeAttribute("poster");
  preview.load();
  preview.hidden = true;
  $("#render-loading").hidden = false;
  $("#render-loading").classList.add("is-idle");
  $("#render-error").hidden = true;
  $("#render-status").textContent = "请选择要生成的直拍类型";
  $("#render-status-detail").textContent = horizontal
    ? "当前：横屏直拍 · 保持横屏比例，放大后仅柔焦左右两侧"
    : "当前：竖屏直拍 · 9:16 动态裁剪并跟随人物";
  $("#camera-qa-strip").hidden = true;
  $("#camera-qa-strip").replaceChildren();
  $("#render-metric-ratio").textContent = horizontal ? "横屏" : "9:16";
  $("#render-metric-zoom").textContent = "—";
  $("#render-metric-time").textContent = "—";
  $("#camera-plan-summary").textContent = "等待生成镜头计划…";
  $("#regenerate-preview").disabled = false;
  $("#export-video-button").disabled = true;
  $("#download-video-link").hidden = true;
}

function showRenderError(title, hint) {
  $("#render-loading").hidden = true;
  $("#render-loading").classList.remove("is-idle");
  $("#render-error").hidden = false;
  $("#render-error-title").textContent = title || "预览生成失败";
  $("#render-error-hint").textContent = hint || "请确认 FFmpeg 可用后重试。";
  $("#regenerate-preview").disabled = false;
}

function renderCameraQa(frames, outputMode = "vertical") {
  const strip = $("#camera-qa-strip");
  strip.replaceChildren();
  (frames || []).forEach((frame) => {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = frame.image;
    image.alt = outputMode === "horizontal-focus"
      ? `${formatTime(frame.time)} 的横屏主体柔焦检查画面`
      : `${formatTime(frame.time)} 的 9:16 镜头窗口检查画面`;
    const caption = document.createElement("figcaption");
    caption.textContent = formatTime(frame.time);
    figure.append(image, caption);
    strip.appendChild(figure);
  });
  strip.hidden = !frames?.length;
}

function applyPreviewResult(result) {
  state.previewResult = result;
  state.exportResult = null;
  const preview = $("#render-preview-video");
  preview.src = `${result.output_url}?render=${result.render_id}`;
  if (result.poster) preview.poster = result.poster;
  preview.hidden = false;
  preview.load();
  const horizontal = result.output_mode === "horizontal-focus";
  $("#render-preview-shell").classList.toggle("is-horizontal", horizontal);
  $("#render-loading").hidden = true;
  $("#render-error").hidden = true;
  $("#render-metric-ratio").textContent = `${result.media.width}×${result.media.height}`;
  $("#render-metric-zoom").textContent = `${Number(result.camera.applied_zoom).toFixed(2)}×`;
  $("#render-metric-time").textContent = `${Number(result.elapsed_seconds || 0).toFixed(1)}s`;
  $("#camera-plan-summary").textContent = horizontal
    ? `${result.camera.framing_label} ${Number(result.camera.applied_zoom).toFixed(2)}× · ${result.camera.motion_label} · 中央清晰约 ${result.camera.clear_zone_percent}% · ${result.camera.focus_keyframes} 个关键点`
    : `${result.camera.framing_label} · ${result.camera.motion_label} · 足底约 ${result.camera.target_footroom_percent}% · ${result.camera.camera_keyframes} 个镜头关键点`;
  $("#regenerate-preview").disabled = false;
  $("#export-video-button").disabled = false;
  $("#download-video-link").hidden = true;
  renderCameraQa(result.camera_qa_frames, result.output_mode);
  showToast(horizontal ? "横屏柔焦预览已生成，请播放检查人物清晰区" : "9:16 低清预览已生成，请播放检查构图");
}

async function requestRender(mode = "preview") {
  if (!state.file || !state.tracking) return;
  state.renderAbortController?.abort();
  state.renderAbortController = new AbortController();
  const isPreview = mode === "preview";
  const exportButton = $("#export-video-button");
  const regenerateButton = $("#regenerate-preview");
  const settings = currentCameraSettings();
  const horizontal = isHorizontalMode(settings);
  regenerateButton.disabled = true;
  exportButton.disabled = true;

  if (isPreview) {
    state.previewResult = null;
    state.exportResult = null;
    $("#render-preview-video").hidden = true;
    $("#render-error").hidden = true;
    $("#render-loading").hidden = false;
    $("#render-loading").classList.remove("is-idle");
    $("#render-status").textContent = horizontal
      ? "正在生成横屏主体柔焦预览…"
      : "正在规划自然镜头并渲染低清预览…";
    $("#render-status-detail").textContent = horizontal
      ? "保持横屏比例动态放大，人物上下清晰，只柔焦左右两侧"
      : "先确保人物完整，再应用放大、死区与速度限制";
    $("#camera-qa-strip").hidden = true;
    $("#download-video-link").hidden = true;
  } else {
    exportButton.innerHTML = "正在导出高清 MP4… <span>⋯</span>";
    showToast(horizontal ? "正在按所选放大档位导出横屏成片，请保持页面开启" : "正在导出 1080×1920 高清成片，请保持页面开启");
  }

  const statusTimer = window.setTimeout(() => {
    if (isPreview) {
      $("#render-status").textContent = horizontal ? "正在逐帧合成横屏柔焦预览…" : "正在逐帧生成 9:16 预览…";
      $("#render-status-detail").textContent = "片段越长，渲染所需时间越长";
    }
  }, 2200);

  try {
    const form = new FormData();
    form.append("video", state.file, state.file.name || "video.mp4");
    form.append("start", String(state.start));
    form.append("end", String(state.end));
    form.append("tracking", JSON.stringify(compactTrackingPayload()));
    form.append("output_mode", settings.outputMode);
    form.append("framing", settings.framing);
    form.append("motion", settings.motion);
    form.append("mode", mode);
    const response = await fetch("/api/render", {
      method: "POST",
      body: form,
      signal: state.renderAbortController.signal,
    });
    const result = await response.json();
    if (!response.ok) {
      const failure = new Error(result.error?.message || "直拍视频生成失败");
      failure.hint = result.error?.hint;
      throw failure;
    }
    if (!result.output_url || !result.media?.width) throw new Error("渲染服务没有返回有效视频");

    if (isPreview) {
      applyPreviewResult(result);
    } else {
      state.exportResult = result;
      const link = $("#download-video-link");
      link.href = `${result.output_url}?render=${result.render_id}`;
      link.download = result.download_name || (horizontal ? "一键直拍-横屏柔焦.mp4" : "一键直拍-9x16.mp4");
      link.textContent = `再次下载成片 · ${formatFileSize(result.media.file_size)}`;
      link.hidden = false;
      link.click();
      showToast(`高清成片已生成 · ${result.media.width}×${result.media.height}`);
    }
    return result;
  } catch (error) {
    if (error.name === "AbortError") return;
    console.error(error);
    if (isPreview) showRenderError(error.message, error.hint || "请确认本地 FFmpeg 可用，并保持启动窗口开启。" );
    else showToast(`高清导出失败：${error.message}`);
  } finally {
    window.clearTimeout(statusTimer);
    regenerateButton.disabled = false;
    setExportButtonLabel(settings);
    exportButton.disabled = !state.previewResult;
  }
}

function markPreviewStale() {
  const hadPreview = Boolean(state.previewResult);
  resetRenderUI();
  if (hadPreview) showToast("设置已更改，请重新生成预览");
}

function showExportStage() {
  if (!state.tracking) return;
  $("#track-stage").hidden = true;
  $("#export-stage").hidden = false;
  setStep(4);
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (!state.previewResult) {
    resetRenderUI();
    return Promise.resolve();
  }
  return Promise.resolve(state.previewResult);
}

function showTrackingFromExport() {
  state.renderAbortController?.abort();
  $("#render-preview-video").pause();
  $("#export-stage").hidden = true;
  $("#track-stage").hidden = false;
  setStep(3);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setStep(activeStep) {
  const displayStep = Math.min(activeStep, 3);
  $$(".step").forEach((step) => {
    const number = Number(step.dataset.step);
    step.classList.toggle("is-active", number === displayStep);
    step.classList.toggle("is-complete", number < displayStep);
    if (number === displayStep) step.setAttribute("aria-current", "step");
    else step.removeAttribute("aria-current");
  });
}

function showSelectionStage() {
  video.pause();
  state.previewingSelection = false;
  $("#trim-stage").hidden = true;
  $("#select-stage").hidden = false;
  $("#analysis-range").textContent = `${formatTime(state.start)} — ${formatTime(state.end)}`;
  setStep(2);
  window.scrollTo({ top: 0, behavior: "smooth" });
  return analyzePeople();
}

function showTrimStage() {
  state.analysisAbortController?.abort();
  $("#select-stage").hidden = true;
  $("#trim-stage").hidden = false;
  setStep(1);
}

fileInput.addEventListener("change", (event) => loadVideoFile(event.target.files?.[0]));
startRange.addEventListener("input", () => syncRangeInputs("start"));
endRange.addEventListener("input", () => syncRangeInputs("end"));
video.addEventListener("timeupdate", updatePlayhead);
video.addEventListener("seeked", updatePlayhead);

setStartButton.addEventListener("click", () => {
  const newStart = Math.min(video.currentTime, state.end - 0.05);
  startRange.value = String(Math.max(0, newStart));
  syncRangeInputs("start");
});

setEndButton.addEventListener("click", () => {
  const newEnd = Math.max(video.currentTime, state.start + 0.05);
  endRange.value = String(Math.min(state.duration, newEnd));
  syncRangeInputs("end");
});

playSelectionButton.addEventListener("click", async () => {
  if (state.previewingSelection) {
    video.pause();
    state.previewingSelection = false;
    playSelectionButton.innerHTML = "<span>▶</span> 预览保留片段";
    return;
  }
  video.currentTime = state.start;
  state.previewingSelection = true;
  playSelectionButton.innerHTML = "<span>Ⅱ</span> 暂停预览";
  try { await video.play(); } catch { showToast("请在视频播放器中点击播放"); }
});

continueButton.addEventListener("click", showSelectionStage);
$("#back-to-trim").addEventListener("click", showTrimStage);
$("#retry-analysis").addEventListener("click", analyzePeople);
$("#previous-candidate").addEventListener("click", () => renderCandidate(state.candidateIndex - 1));
$("#next-candidate").addEventListener("click", () => renderCandidate(state.candidateIndex + 1));
$("#confirm-subject-button").addEventListener("click", startTracking);
$("#retry-tracking").addEventListener("click", startTracking);
$("#cancel-tracking-correction").addEventListener("click", closeTrackingCorrection);
$("#reset-tracking-correction").addEventListener("click", resetTrackingCorrectionSelection);
$("#confirm-tracking-correction").addEventListener("click", confirmTrackingCorrection);
$("#clear-tracking-corrections").addEventListener("click", clearPendingCorrections);
$("#apply-tracking-corrections").addEventListener("click", applyPendingCorrections);
$("#back-to-selection").addEventListener("click", showSelectionFromTracking);
$("#continue-camera-button").addEventListener("click", showExportStage);
$("#retry-render").addEventListener("click", () => requestRender("preview"));
$("#regenerate-preview").addEventListener("click", () => requestRender("preview"));
$("#export-video-button").addEventListener("click", () => requestRender("export"));
$("#back-to-tracking").addEventListener("click", showTrackingFromExport);
$$('input[name="output-mode"], input[name="framing"], input[name="motion"]').forEach((input) => {
  input.addEventListener("change", markPreviewStale);
});
$("#representative-frame").addEventListener("load", positionPersonOverlay);
window.addEventListener("resize", positionPersonOverlay);
window.addEventListener("resize", positionTrackingCorrectionOverlay);

const dialog = $("#principles-dialog");
$("#open-principles").addEventListener("click", () => dialog.showModal());

const stage = $("#video-stage");
["dragenter", "dragover"].forEach((eventName) => {
  stage.addEventListener(eventName, (event) => {
    event.preventDefault();
    stage.style.outline = "3px solid #d7ff64";
    stage.style.outlineOffset = "-6px";
  });
});
["dragleave", "drop"].forEach((eventName) => {
  stage.addEventListener(eventName, (event) => {
    event.preventDefault();
    stage.style.outline = "";
    stage.style.outlineOffset = "";
  });
});
stage.addEventListener("drop", (event) => loadVideoFile(event.dataTransfer.files?.[0]));

window.addEventListener("beforeunload", resetObjectUrl);

// Development smoke-test hook: load a same-origin clip with ?demo=path/to/video.mp4.
// It is inert for normal users and lets CI/headless browsers exercise real video logic.
const demoParams = new URLSearchParams(window.location.search);
const demoClipUrl = demoParams.get("demo");
if (demoClipUrl) {
  const demoSource = new URL(demoClipUrl, window.location.href).href;
  fetch(demoSource)
    .then((response) => {
      if (!response.ok) throw new Error(`Demo clip HTTP ${response.status}`);
      return response.blob();
    })
    .then((blob) => loadVideoFile(
      new File([blob], "demo-clip.mp4", { type: "video/mp4" }),
    ))
    .then(async (loaded) => {
      const targetStage = demoParams.get("stage");
      if (!loaded || !["select", "track", "export"].includes(targetStage)) return;
      const analysis = await showSelectionStage();
      if (targetStage === "select" || !analysis) return;
      const candidate = analysis.candidates[0];
      const requestedId = demoParams.get("person") || candidate.boxes[0]?.id;
      const box = candidate.boxes.find((item) => item.id === requestedId);
      if (!box) throw new Error(`Demo person ${requestedId} was not detected`);
      selectSubject(candidate, box, { x: box.cx, y: box.cy });
      const tracking = await startTracking();
      if (targetStage === "export" && tracking) {
        await showExportStage();
        const requestedMode = demoParams.get("outputMode");
        const outputModeInput = requestedMode
          ? $$('input[name="output-mode"]').find((input) => input.value === requestedMode)
          : null;
        if (outputModeInput) {
          outputModeInput.checked = true;
          syncOutputModeUI();
        }
        const preview = await requestRender("preview");
        if (preview && demoParams.get("autoexport") === "1") await requestRender("export");
      }
    })
    .catch((error) => console.error("Demo clip could not be loaded", error));
}
