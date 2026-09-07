const $ = (id) => document.getElementById(id);

let connected = false;
let running = false;
let voiceWs = null;
let capture = null;
let player = null;
let talkPhase = "idle";
let wakeArmed = false;
let recording = false;
let turns = 0;
let lastTtsEl = null;
let lastSttEl = null;
let msgChain = Promise.resolve();
let voiced = false;
let silenceMs = 0;
let voicedMs = 0;

async function api(path, options = {}) {
  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok && !data.error) return { ok: false, error: "请求失败（HTTP " + res.status + "）" };
    return data;
  } catch (err) {
    return { ok: false, error: "设备端程序没有响应。" };
  }
}

function logLine(text) {
  const box = $("logs");
  if (!box) return;
  const line = document.createElement("div");
  line.className = "log-line";
  line.textContent = text;
  box.prepend(line);
}

function setRuntime(on, label) {
  connected = !!on;
  const pill = $("runtimePill");
  pill.className = "runtime-pill " + (on ? "is-online" : "is-offline");
  pill.querySelector("span").textContent = label;
}

function setPhase(name, title, hint) {
  talkPhase = name;
  $("phaseTitle").textContent = title;
  $("phaseHint").textContent = hint;
  $("userStage").textContent = "用户 · " + title;
  const orb = $("orbWrap");
  if (orb) {
    orb.classList.toggle("is-recording", name === "recording");
    orb.classList.toggle("is-processing", name === "processing");
    orb.classList.toggle("is-speaking", name === "speaking");
  }
  document.querySelectorAll("#phaseTrack [data-phase]").forEach((el) => {
    el.classList.toggle("is-active", el.dataset.phase === name);
  });
}

function setMsg(text, ok) {
  $("connMsg").textContent = text || "";
  $("controlMessage").textContent = text || "";
  logLine(text);
}

function addBubble(role, text, { append = false } = {}) {
  const empty = $("emptyConversation");
  if (empty) empty.remove();
  const box = $("conversation");
  if (append && role === "user" && lastSttEl && lastSttEl.parentNode === box) {
    const body = lastSttEl.querySelector(".message__body");
    if (body) body.textContent = text;
    return lastSttEl;
  }
  if (append && role === "assistant" && lastTtsEl && lastTtsEl.parentNode === box) {
    const body = lastTtsEl.querySelector(".message__body");
    if (body) {
      const prev = body.textContent || "";
      if (prev && !/[。！？!?]$/.test(prev.trim())) {
        body.textContent += text;
        return lastTtsEl;
      }
    }
  }
  const el = document.createElement("div");
  el.className = "message " + (role === "user" ? "message--user" : "message--assistant");
  el.innerHTML = '<div class="message__body"></div>';
  el.querySelector(".message__body").textContent = text;
  box.prepend(el);
  if (role === "user") lastSttEl = el;
  else {
    lastTtsEl = el;
    turns += 1;
    $("turnCount").textContent = turns + " 轮";
  }
  return el;
}

function downsample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const ratio = fromRate / toRate;
  const n = Math.floor(input.length / ratio);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = input[Math.min(input.length - 1, Math.floor(i * ratio))];
  return out;
}

function floatToS16Bytes(f32) {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i += 1) {
    const s = Math.max(-1, Math.min(1, f32[i]));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return new Uint8Array(out.buffer);
}

function rms(f32) {
  let sum = 0;
  for (let i = 0; i < f32.length; i += 1) sum += f32[i] * f32[i];
  return Math.sqrt(sum / Math.max(1, f32.length));
}

async function startCapture(onFrame) {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
  });
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === "suspended") await ctx.resume();
  const source = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  proc.onaudioprocess = (ev) => {
    const input = ev.inputBuffer.getChannelData(0);
    const resampled = downsample(input, ctx.sampleRate, 16000);
    onFrame(floatToS16Bytes(resampled), rms(resampled));
  };
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(proc);
  proc.connect(mute);
  mute.connect(ctx.destination);
  return {
    stop() {
      try { proc.disconnect(); } catch (err) { /* ignore */ }
      try { source.disconnect(); } catch (err) { /* ignore */ }
      stream.getTracks().forEach((t) => t.stop());
      ctx.close().catch(() => {});
    },
  };
}

function createPcmPlayer(sampleRate) {
  const rate = sampleRate || 16000;
  let ctx = null;
  let next = 0;
  let leftover = new Uint8Array(0);
  const sources = [];
  let gate = Promise.resolve();
  async function ensure() {
    if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (ctx.state === "suspended") await ctx.resume();
    return ctx;
  }
  async function enqueue(bytes) {
    if (!bytes || !bytes.length) return;
    const audio = await ensure();
    let data = bytes;
    if (leftover.length) {
      const merged = new Uint8Array(leftover.length + bytes.length);
      merged.set(leftover, 0);
      merged.set(bytes, leftover.length);
      data = merged;
      leftover = new Uint8Array(0);
    }
    if (data.length % 2 === 1) {
      leftover = data.slice(data.length - 1);
      data = data.slice(0, data.length - 1);
    }
    if (data.length < 2) return;
    const aligned = new Uint8Array(data.byteLength);
    aligned.set(data);
    const samples = new Int16Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 2);
    const buffer = audio.createBuffer(1, samples.length, rate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i += 1) channel[i] = samples[i] / 32768;
    const src = audio.createBufferSource();
    src.buffer = buffer;
    src.connect(audio.destination);
    const now = audio.currentTime;
    if (next < now + 0.02) next = now + 0.02;
    src.start(next);
    next += buffer.duration;
    sources.push(src);
  }
  return {
    async playChunk(bytes) {
      const copy = bytes.slice ? bytes.slice() : new Uint8Array(bytes);
      const done = gate.then(() => enqueue(copy));
      gate = done.catch(() => {});
      return done;
    },
    stop() {
      gate = Promise.resolve();
      leftover = new Uint8Array(0);
      sources.forEach((src) => { try { src.stop(); } catch (err) { /* ignore */ } });
      sources.length = 0;
      next = 0;
    },
    playbackDone() {
      if (!ctx) return Promise.resolve();
      const remaining = Math.max(0, next - ctx.currentTime);
      return new Promise((resolve) => setTimeout(resolve, remaining * 1000 + 120));
    },
  };
}

async function saveAndConnect() {
  await api("/api/settings", {
    method: "PUT",
    body: JSON.stringify({
      url: $("url").value.trim(),
      robot_id: $("robot").value.trim(),
      schema_version: Number($("schema").value || 1),
    }),
  });
  const data = await api("/api/connect", { method: "POST", body: "{}" });
  if (!data.ok) {
    setRuntime(false, "未连接");
    setMsg(data.error || "连接失败", false);
    return false;
  }
  setRuntime(true, "已连接");
  const live = [];
  if (data.robot_publish_enabled) live.push("机械臂真机");
  if (data.massage_enabled) live.push("按摩真机");
  setMsg(live.length ? "已连上主机（" + live.join("、") + "）" : "已连上主机（演练）", true);
  if (live.length) {
    $("safetyCard").textContent = "真机已开：" + live.join("、") + "。急停必须在手边。";
    $("teachingProgress").textContent = "真机";
    $("teachingHelp").textContent = "按左肩、右肩、背部、左腰、右腰顺序记录。说「记录完成」或点结束示教。";
  }
  return true;
}

function handleEvent(ev) {
  if (!ev || !ev.type) return;
  if (ev.type === "wake" && ev.status === "waiting") {
    wakeArmed = false;
    recording = false;
    if (talkPhase === "speaking") return;
    setPhase("waiting", "等待唤醒", "请说「小溪小溪」");
    $("assistantStage").textContent = "小溪 · 等待";
    return;
  }
  if (ev.type === "wake" && ev.status === "hit") {
    wakeArmed = true;
    recording = true;
    voiced = false;
    silenceMs = 0;
    voicedMs = 0;
    lastSttEl = null;
    lastTtsEl = null;
    setPhase("recording", "请说话", "已唤醒，请说内容");
    return;
  }
  if (ev.type === "wake" && ev.status === "listen") {
    setPhase("recording", "正在听并识别", "请继续说");
    return;
  }
  if (ev.type === "wake" && ev.status === "idle") {
    wakeArmed = false;
    recording = false;
    setPhase("waiting", "等待唤醒", "没有听到内容，请再说「小溪小溪」");
    return;
  }
  if (ev.type === "wake" && ev.status === "rejected") {
    wakeArmed = false;
    recording = false;
    setPhase("waiting", "声纹未通过", ev.message || "未通过声纹，请本人再说");
    $("assistantStage").textContent = "小溪 · 拒绝";
    logLine("声纹拒绝：" + (ev.display_name || ev.reason || "未授权"));
    return;
  }
  if (ev.type === "speaker") {
    logLine(
      "声纹 mode=" + (ev.mode || "-")
      + " accepted=" + ev.accepted
      + " speaker=" + (ev.display_name || "-")
      + " score=" + (ev.score == null ? "-" : ev.score)
      + " reason=" + (ev.reason || "-")
    );
  }
  if (ev.type === "stt") {
    addBubble("user", ev.text || "", { append: true });
    setPhase("processing", "等待回复", "识别：" + (ev.text || ""));
  }
  if (ev.type === "tts_text") {
    addBubble("assistant", ev.text || "", { append: true });
    $("assistantStage").textContent = "小溪 · 回答";
    setPhase("speaking", "正在播放", "小溪正在说话");
  }
  if (ev.type === "bridge") {
    const st = String(ev.status || "");
    logLine("机械臂：" + st);
    if (st === "dry_run") $("safetyCard").textContent = "演练模式：指令已校验，不会发布真机动作。";
    if (st.startsWith("error:")) {
      addBubble("assistant", st.slice(6));
      setMsg(st.slice(6), false);
    }
  }
  if (ev.type === "error") setMsg(ev.message || "出错", false);
  if (ev.type === "robot_command") {
    const first = (ev.commands || [])[0] || {};
    if (first.action === "massage") {
      $("teachingPanel").hidden = false;
      $("teachingNextPart").textContent = first.op === "start_teaching" ? "请按顺序记点" : (first.op || "massage");
    }
  }
}

async function startVoice() {
  if (running) return;
  const ok = await saveAndConnect();
  if (!ok) return;
  player = createPcmPlayer(16000);
  const proto = location.protocol === "https:" ? "wss" : "ws";
  voiceWs = new WebSocket(proto + "://" + location.host + "/ws/voice");
  await new Promise((resolve, reject) => {
    voiceWs.onopen = resolve;
    voiceWs.onerror = () => reject(new Error("ws"));
  });
  async function onVoiceMessage(ev) {
    if (typeof ev.data !== "string") {
      const buf = ev.data instanceof Blob ? await ev.data.arrayBuffer() : ev.data;
      setPhase("speaking", "正在播放", "小溪正在说话");
      try { await player.playChunk(new Uint8Array(buf)); } catch (err) { /* autoplay */ }
      return;
    }
    const evj = JSON.parse(ev.data);
    handleEvent(evj);
    if (evj.type === "turn_end") {
      const wait = player && player.playbackDone ? player.playbackDone() : Promise.resolve();
      await wait;
      await new Promise((resolve) => setTimeout(resolve, 400));
      wakeArmed = false;
      recording = false;
      if (running) {
        setPhase("waiting", "等待唤醒", "可以说下一次「小溪小溪」");
        $("assistantStage").textContent = "小溪 · 等待";
      }
    }
  }
  voiceWs.onmessage = (ev) => {
    msgChain = msgChain.then(() => onVoiceMessage(ev)).catch((err) => console.warn("voice message", err));
  };
  try {
    capture = await startCapture((bytes, energy) => {
      if (!running || talkPhase === "speaking") return;
      if (voiceWs && voiceWs.readyState === 1) voiceWs.send(bytes);
      if (!wakeArmed) return;
      const dt = 1000 * (bytes.length / 2) / 16000;
      if (energy >= 0.012) {
        voiced = true;
        voicedMs += dt;
        silenceMs = 0;
      } else if (voiced) {
        silenceMs += dt;
      }
      if (recording && voiced && voicedMs >= 200 && silenceMs >= 800) {
        recording = false;
        if (voiceWs && voiceWs.readyState === 1) {
          voiceWs.send(JSON.stringify({ type: "listen", state: "stop" }));
        }
        setPhase("processing", "等待回复", "已说完，正在回复");
      }
    });
  } catch (err) {
    setMsg("浏览器不允许使用麦克风，请在权限里允许。", false);
    voiceWs.close();
    return;
  }
  running = true;
  $("startBtn").disabled = true;
  $("stopBtn").disabled = false;
  setPhase("waiting", "等待唤醒", "请说「小溪小溪」");
  voiceWs.onclose = () => {
    if (running) setMsg("语音通道断开。", false);
  };
}

async function stopVoice() {
  running = false;
  wakeArmed = false;
  if (capture) { capture.stop(); capture = null; }
  if (player) { player.stop(); player = null; }
  if (voiceWs) {
    try { voiceWs.close(); } catch (err) { /* ignore */ }
    voiceWs = null;
  }
  await api("/api/abort", { method: "POST", body: "{}" });
  $("startBtn").disabled = false;
  $("stopBtn").disabled = true;
  setPhase("idle", "已停止", "可以再次启动");
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => {
    document.querySelectorAll(".tab").forEach((el) => el.classList.remove("is-active"));
    document.querySelectorAll(".tab-view").forEach((el) => el.classList.remove("is-active"));
    tab.classList.add("is-active");
    document.querySelector('[data-view="' + tab.dataset.tab + '"]').classList.add("is-active");
    if (tab.dataset.tab === "speakers") refreshSpeakers();
  };
});

$("saveBtn").onclick = saveAndConnect;
$("startBtn").onclick = startVoice;
$("stopBtn").onclick = stopVoice;
$("clearBtn").onclick = () => {
  $("conversation").innerHTML = "";
  $("logs").innerHTML = "";
  turns = 0;
  $("turnCount").textContent = "0 轮";
};
$("clearLogsBtn").onclick = () => { $("logs").innerHTML = ""; };
function teachingNote(data) {
  const detail = data.detail || {};
  const count = detail.point_count;
  const next = detail.next_part;
  const reply = data.message || detail.reply || detail.message || data.error || "";
  if (typeof count === "number") {
    $("teachingProgress").textContent = count + " / " + (detail.required || 5);
  }
  if (next) $("teachingNextPart").textContent = "下一个：" + next;
  return reply;
}
$("captureTeachingBtn").onclick = async () => {
  const data = await api("/api/massage/teaching/capture", { method: "POST", body: "{}" });
  setMsg(teachingNote(data) || "已记点", !!data.ok);
};
$("finishTeachingBtn").onclick = async () => {
  const data = await api("/api/massage/teaching/finish", { method: "POST", body: "{}" });
  setMsg(teachingNote(data) || "已结束示教", !!data.ok);
};

let enrollment = null;
let speakerBusy = false;

function speakerModeLabel(mode) {
  return { off: "关闭", shadow: "影子验证", enforce: "强制门控" }[mode] || mode || "关闭";
}

function renderEnrollment(data) {
  enrollment = data || null;
  const card = $("enrollmentCard");
  if (!card) return;
  if (!enrollment) {
    card.hidden = true;
    return;
  }
  card.hidden = false;
  const count = enrollment.sample_count || 0;
  const need = enrollment.required_samples || 3;
  $("enrollmentProgress").textContent = count + " / " + need;
  $("enrollmentPrompt").textContent = enrollment.prompt || "请按提示朗读";
  $("commitSpeakerBtn").disabled = !enrollment.complete || speakerBusy;
  $("captureSpeakerSampleBtn").disabled = !!enrollment.complete || speakerBusy;
}

function renderSpeaker(data) {
  const profiles = Array.isArray(data && data.profiles) ? data.profiles : [];
  const summary = $("speakerSummary");
  if (summary) {
    const rows = [
      ["当前模式", speakerModeLabel(data.mode)],
      ["阈值", data.threshold == null ? "-" : Number(data.threshold).toFixed(2)],
      ["授权用户", profiles.length + " 位"],
      ["模型", data.ready ? "已就绪" : (data.error || "未就绪")],
    ];
    summary.innerHTML = rows.map((row) => (
      '<div class="speaker-summary-row"><span>' + row[0] + "</span><b>" + row[1] + "</b></div>"
    )).join("");
  }
  const modeSel = $("speakerModeSelect");
  if (modeSel && document.activeElement !== modeSel) modeSel.value = data.mode || "off";
  const voiceBusy = !!data.voice_busy;
  const enrollLocked = voiceBusy || speakerBusy || !!enrollment;
  if ($("speakerEnrollBtn")) $("speakerEnrollBtn").disabled = enrollLocked;
  if ($("speakerNameInput")) $("speakerNameInput").disabled = enrollLocked;
  if ($("captureProbeBtn")) $("captureProbeBtn").disabled = voiceBusy || speakerBusy || !!enrollment;
  if ($("speakerHelp")) {
    $("speakerHelp").textContent = data.error
      ? data.error
      : voiceBusy
        ? "登记和试听前请先停止语音。"
        : "登记使用机器人麦克风。默认关闭，不拦截唤醒。声纹挡旁人，挡不了录音回放。";
  }
  const list = $("speakerList");
  if (list) {
    list.innerHTML = "";
    if (!profiles.length) {
      const empty = document.createElement("div");
      empty.className = "speaker-empty";
      empty.textContent = "还没有授权用户。";
      list.appendChild(empty);
    } else {
      profiles.forEach((profile) => {
        const item = document.createElement("div");
        item.className = "speaker-item";
        const head = document.createElement("header");
        const name = document.createElement("strong");
        name.textContent = profile.display_name || profile.speaker_id;
        const meta = document.createElement("span");
        meta.textContent = (profile.sample_count || 0) + " 条样本";
        head.appendChild(name);
        head.appendChild(meta);
        const foot = document.createElement("footer");
        const del = document.createElement("button");
        del.type = "button";
        del.textContent = "删除";
        del.disabled = voiceBusy || speakerBusy;
        del.onclick = () => deleteSpeaker(profile);
        foot.appendChild(del);
        item.appendChild(head);
        item.appendChild(foot);
        list.appendChild(item);
      });
    }
  }
  renderEnrollment(data.enrollment || enrollment);
}

async function refreshSpeakers() {
  const data = await api("/api/speakers");
  if (data && data.error && !data.mode) {
    if ($("speakerHelp")) $("speakerHelp").textContent = data.error;
    return data;
  }
  enrollment = data.enrollment || null;
  renderSpeaker(data);
  return data;
}

async function applySpeakerMode() {
  const mode = $("speakerModeSelect").value;
  const data = await api("/api/speakers", { method: "PUT", body: JSON.stringify({ mode }) });
  if (!data.ok && data.error) {
    $("speakerModeHelp").textContent = data.error;
    return;
  }
  $("speakerModeHelp").textContent = "已切换为" + speakerModeLabel(data.mode) + "。下一句指令生效。";
  renderSpeaker(data);
}

async function startEnrollment() {
  const name = ($("speakerNameInput").value || "").trim();
  if (!name) {
    $("speakerHelp").textContent = "请先输入用户名称。";
    return;
  }
  speakerBusy = true;
  $("speakerHelp").textContent = "正在创建声纹登记会话…";
  try {
    const data = await api("/api/speakers/enrollments", {
      method: "POST",
      body: JSON.stringify({ display_name: name }),
    });
    if (!data.ok) {
      $("speakerHelp").textContent = data.error || "无法开始登记";
      return;
    }
    $("speakerNameInput").value = "";
    $("speakerHelp").textContent = "登记进行中。请先停止语音，再用机器人麦朗读提示句。";
    $("enrollmentMessage").textContent = "点击录制后开始朗读。";
    renderEnrollment(data);
  } finally {
    speakerBusy = false;
    await refreshSpeakers();
  }
}

async function captureSpeakerSample() {
  if (!enrollment) return;
  speakerBusy = true;
  $("enrollmentMessage").textContent = "正在收音，请自然朗读提示句，读完后停顿一下…";
  renderEnrollment(enrollment);
  try {
    const data = await api("/api/speakers/enrollments/" + encodeURIComponent(enrollment.session_id) + "/samples", {
      method: "POST",
      body: "{}",
    });
    if (!data.ok) {
      $("enrollmentMessage").textContent = data.error || "本条未计入";
      return;
    }
    const accepted = !data.sample || data.sample.accepted !== false;
    const reason = data.sample && data.sample.reason ? String(data.sample.reason) : "";
    $("enrollmentMessage").textContent = accepted
      ? ("本条录制有效" + (data.sample && data.sample.audio_ms ? "：" + (data.sample.audio_ms / 1000).toFixed(1) + " 秒" : "。"))
      : ("本条未计入" + (reason ? "：" + reason : "，请重新录制"));
    renderEnrollment(data);
  } finally {
    speakerBusy = false;
    await refreshSpeakers();
  }
}

async function commitSpeaker() {
  if (!enrollment) return;
  speakerBusy = true;
  $("enrollmentMessage").textContent = "正在保存声纹…";
  try {
    const data = await api("/api/speakers/enrollments/" + encodeURIComponent(enrollment.session_id) + "/commit", {
      method: "POST",
      body: "{}",
    });
    if (!data.ok) {
      $("enrollmentMessage").textContent = data.error || "保存失败";
      return;
    }
    $("enrollmentMessage").textContent = "";
    renderEnrollment(null);
    $("speakerHelp").textContent = "已保存「" + ((data.profile && data.profile.display_name) || "") + "」。可先用影子模式试听。";
  } finally {
    speakerBusy = false;
    await refreshSpeakers();
  }
}

async function cancelSpeaker() {
  if (!enrollment) return;
  speakerBusy = true;
  try {
    await api("/api/speakers/enrollments/" + encodeURIComponent(enrollment.session_id), { method: "DELETE" });
    renderEnrollment(null);
    $("speakerHelp").textContent = "已取消本次登记。";
  } finally {
    speakerBusy = false;
    await refreshSpeakers();
  }
}

async function deleteSpeaker(profile) {
  if (!window.confirm("确认删除「" + (profile.display_name || "") + "」的声纹授权吗？")) return;
  const data = await api("/api/speakers/" + encodeURIComponent(profile.speaker_id), { method: "DELETE" });
  $("speakerHelp").textContent = data.ok ? "已删除该授权。" : (data.error || "删除失败");
  await refreshSpeakers();
}

async function captureProbe() {
  speakerBusy = true;
  $("probeResult").textContent = "正在收音，请说一条真实指令…";
  try {
    const data = await api("/api/speakers/probes", { method: "POST", body: "{}" });
    if (!data.ok && data.error) {
      $("probeResult").textContent = data.error;
      return;
    }
    $("probeResult").textContent = "score=" + (data.score == null ? "-" : data.score)
      + " speaker=" + (data.display_name || "-")
      + " accepted=" + data.accepted
      + " reason=" + (data.reason || "-");
  } finally {
    speakerBusy = false;
  }
}

if ($("speakerModeBtn")) $("speakerModeBtn").onclick = applySpeakerMode;
if ($("speakerEnrollBtn")) $("speakerEnrollBtn").onclick = startEnrollment;
if ($("captureSpeakerSampleBtn")) $("captureSpeakerSampleBtn").onclick = captureSpeakerSample;
if ($("commitSpeakerBtn")) $("commitSpeakerBtn").onclick = commitSpeaker;
if ($("cancelSpeakerBtn")) $("cancelSpeakerBtn").onclick = cancelSpeaker;
if ($("captureProbeBtn")) $("captureProbeBtn").onclick = captureProbe;

(async () => {
  const settings = await api("/api/settings");
  if (settings.url) $("url").value = settings.url;
  if (settings.robot_id) $("robot").value = settings.robot_id;
  const status = await api("/api/status");
  if (status.wake_word) $("wakeWord").textContent = status.wake_word;
  if (status.connected) setRuntime(true, "已连接");
  if (status.kws_ready) logLine("唤醒已就绪：" + (status.wake_word || "小溪小溪"));
  else if (status.kws_error) logLine("唤醒未就绪：" + status.kws_error);
  const live = [];
  if (status.robot_publish_enabled) live.push("机械臂真机");
  if (status.massage_enabled) live.push("按摩真机");
  $("safetyCard").textContent = live.length
    ? "真机已开：" + live.join("、") + "。急停必须在手边。"
    : "演练模式：指令会校验，不会发布真机动作。";
  if (status.speaker_mode && status.speaker_mode !== "off") {
    logLine("声纹模式：" + speakerModeLabel(status.speaker_mode));
  }
  refreshSpeakers();
})();
