const $ = (id) => document.getElementById(id);
let busy = false;
let connected = false;
let sayController = null;
let activePlayer = null;

async function api(path, options = {}) {
  try {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json", ...(options.headers || {}) },
      ...options,
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok && !data.error) {
      return { ok: false, error: "请求失败（HTTP " + res.status + "）" };
    }
    return data;
  } catch (err) {
    return { ok: false, error: "设备端程序没有响应，请重新打开 Voice Hub 设备端。" };
  }
}

function setLamp(on, label) {
  connected = !!on;
  const lamp = $("lamp");
  lamp.className = "lamp " + (on ? "on" : "off");
  lamp.textContent = label;
  if (!busy) {
    $("say").disabled = !on;
    $("talk").disabled = !on;
  }
}

function setMsg(text, ok) {
  const el = $("connMsg");
  el.textContent = text || "";
  el.className = "msg " + (ok === true ? "ok" : ok === false ? "err" : "");
}

function setBusy(value) {
  busy = value;
  ["save", "connect", "say"].forEach((id) => {
    $(id).disabled = value;
  });
  $("stop").disabled = !value;
  if (!value) {
    $("say").disabled = !connected;
    $("talk").disabled = !connected;
  }
}

function applySettings(s) {
  $("url").value = s.url || "";
  $("robot").value = s.robot_id || "";
  $("schema").value = s.schema_version || 1;
  const warn = $("warn");
  if (s.warning) {
    warn.hidden = false;
    warn.textContent = s.warning;
  } else {
    warn.hidden = true;
    warn.textContent = "";
  }
}

let lastTtsEl = null;

function appendEvent(ev) {
  if (!ev || ev.type === "turn_end") return;
  const kind = ev.type || "event";
  const log = $("events");
  const first = log.firstElementChild;
  if (kind === "stt") {
    lastTtsEl = null;
    if (first && first.dataset.kind === "stt") {
      first.textContent = "识别：" + (ev.text || "");
      return;
    }
  }
  if (kind === "tts_text") {
    if (lastTtsEl && lastTtsEl.parentNode === log) {
      lastTtsEl.textContent += ev.text || "";
      return;
    }
  }
  if (kind === "audio" && first && first.dataset.kind === "audio") {
    const prev = Number(first.dataset.bytes || 0) + (ev.bytes || 0);
    first.dataset.bytes = String(prev);
    first.textContent = "音频：" + prev + " 字节";
    return;
  }
  const li = document.createElement("li");
  li.dataset.kind = kind === "tts_text" ? "tts" : kind;
  if (kind === "stt") li.textContent = "识别：" + (ev.text || "");
  else if (kind === "tts_text") {
    li.textContent = "回复：" + (ev.text || "");
    lastTtsEl = li;
  }
  else if (kind === "audio") {
    li.dataset.bytes = String(ev.bytes || 0);
    li.textContent = "音频：" + (ev.bytes || 0) + " 字节";
  }
  else if (kind === "error") li.textContent = "错误：" + (ev.message || JSON.stringify(ev));
  else if (kind === "hello_ok") li.textContent = "握手成功：" + (ev.adapter || ev.robot_id || "");
  else if (kind === "hello_reject") li.textContent = "拒绝：" + (ev.reason || "");
  else if (kind === "bridge") li.textContent = "机械臂：" + (ev.status || "");
  else li.textContent = kind + " " + JSON.stringify(ev);
  log.prepend(li);
}

function playWav(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  const blob = new Blob([bytes], { type: "audio/wav" });
  const player = $("player");
  if (player.src && player.src.startsWith("blob:")) URL.revokeObjectURL(player.src);
  player.src = URL.createObjectURL(blob);
  const played = player.play();
  if (played && typeof played.catch === "function") {
    played.catch(() => setMsg("浏览器拦住了自动播放，请点下方播放键。", true));
  }
}

function decodeB64(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) bytes[i] = bin.charCodeAt(i);
  return bytes;
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
      if (!bytes || !bytes.length) return;
      const copy = bytes.slice ? bytes.slice() : new Uint8Array(bytes);
      const done = gate.then(() => enqueue(copy));
      gate = done.catch(() => {});
      return done;
    },
    stop() {
      gate = Promise.resolve();
      leftover = new Uint8Array(0);
      sources.forEach((src) => {
        try {
          src.stop();
        } catch (err) {
          /* already ended */
        }
      });
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

function pcmToWavBlob(pcmBytes, sampleRate) {
  const rate = sampleRate || 16000;
  const pcm = pcmBytes instanceof Uint8Array ? pcmBytes : new Uint8Array(pcmBytes);
  const header = new ArrayBuffer(44);
  const view = new DataView(header);
  const writeStr = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + pcm.length, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, pcm.length, true);
  const out = new Uint8Array(44 + pcm.length);
  out.set(new Uint8Array(header), 0);
  out.set(pcm, 44);
  return new Blob([out], { type: "audio/wav" });
}

function setReplay(pcmParts) {
  const player = $("player");
  if (player.src && player.src.startsWith("blob:")) URL.revokeObjectURL(player.src);
  if (!pcmParts.length) {
    player.removeAttribute("src");
    return;
  }
  let total = 0;
  pcmParts.forEach((p) => {
    total += p.length;
  });
  const pcm = new Uint8Array(total);
  let offset = 0;
  pcmParts.forEach((p) => {
    pcm.set(p, offset);
    offset += p.length;
  });
  player.src = URL.createObjectURL(pcmToWavBlob(pcm, 16000));
}

function parseSseBlock(raw) {
  let event = "message";
  const dataLines = [];
  raw.split("\n").forEach((line) => {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  });
  if (!dataLines.length) return null;
  const text = dataLines.join("\n");
  try {
    return { event, data: JSON.parse(text) };
  } catch (err) {
    return { event, data: { type: event, message: text } };
  }
}

async function refreshStatus() {
  if (busy) return;
  const s = await api("/api/status");
  if (s.error && !s.connected) return;
  setLamp(!!s.connected, s.connected ? "已连接" : "未连接");
  if (s.connected && s.hello && s.hello.adapter) {
    setMsg("已连上 " + s.hello.adapter + (s.bridge_error ? "（本机无机械臂插件，只试音）" : ""), true);
  }
}

async function saveSettings() {
  const data = await api("/api/settings", {
    method: "PUT",
    body: JSON.stringify({
      url: $("url").value.trim(),
      robot_id: $("robot").value.trim(),
      schema_version: Number($("schema").value || 1),
    }),
  });
  if (data.error) {
    setMsg(data.error, false);
    return null;
  }
  applySettings(data);
  return data;
}

$("save").onclick = async () => {
  setBusy(true);
  const data = await saveSettings();
  setBusy(false);
  if (data) setMsg("已保存到本机配置（不含云密钥）", true);
};

$("connect").onclick = async () => {
  setBusy(true);
  setMsg("正在连接…");
  const saved = await saveSettings();
  if (!saved) {
    setBusy(false);
    return;
  }
  const data = await api("/api/connect", { method: "POST", body: "{}" });
  setBusy(false);
  if (!data.ok) {
    setLamp(false, "未连接");
    setMsg(data.error || "连接失败", false);
    return;
  }
  setLamp(true, "已连接");
  setMsg("连接成功" + (data.bridge_error ? "（本机无机械臂插件，只试音）" : ""), true);
  appendEvent(data.hello || { type: "hello_ok" });
};

async function stopPlayback() {
  if (conversation) await stopConversation();
  if (activePlayer) activePlayer.stop();
  if (sayController) sayController.abort();
  await api("/api/abort", { method: "POST", body: "{}" });
}

$("stop").onclick = async () => {
  setMsg("已停止", true);
  await stopPlayback();
};

$("disconnect").onclick = async () => {
  await stopPlayback();
  await api("/api/disconnect", { method: "POST", body: "{}" });
  setBusy(false);
  setLamp(false, "未连接");
  setMsg("已断开", true);
};

$("say").onclick = async () => {
  const text = $("text").value.trim();
  if (!text) {
    setMsg("请输入要说的话。", false);
    return;
  }
  setBusy(true);
  setMsg("等待主机回复…");
  const player = createPcmPlayer(16000);
  activePlayer = player;
  sayController = typeof AbortController !== "undefined" ? new AbortController() : null;
  const pcmParts = [];
  let heard = false;
  let sawText = false;
  try {
    const res = await fetch("/api/say/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ text }),
      signal: sayController ? sayController.signal : undefined,
    });
    const ctype = (res.headers.get("content-type") || "").toLowerCase();
    if (!ctype.includes("text/event-stream")) {
      const data = await res.json().catch(() => ({}));
      setBusy(false);
      setMsg(data.error || "发送失败", false);
      if ((data.error || "").includes("重新连接") || (data.error || "").includes("还没有连上")) {
        setLamp(false, "未连接");
      }
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      buf = buf.replace(/\r\n/g, "\n");
      let sep;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const raw = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const parsed = parseSseBlock(raw);
        if (!parsed) continue;
        const ev = parsed.data || {};
        if (parsed.event === "audio") {
          heard = true;
          appendEvent({ type: "audio", bytes: ev.bytes || 0 });
          if (ev.b64) {
            const bytes = decodeB64(ev.b64);
            pcmParts.push(bytes);
            try {
              await player.playChunk(bytes);
              setMsg("正在播放…", true);
            } catch (err) {
              setReplay(pcmParts);
              setMsg("浏览器拦住了自动播放，请点下方播放键。", true);
            }
          }
          continue;
        }
        appendEvent(ev);
        if (parsed.event === "tts_text" || ev.type === "tts_text") {
          sawText = true;
          if (!heard) setMsg("已收到文字，正在合成语音…", true);
        }
        if (parsed.event === "error" || ev.type === "error") {
          setMsg(ev.message || "发送失败", false);
          if ((ev.message || "").includes("重新连接") || (ev.message || "").includes("还没有连上")) {
            setLamp(false, "未连接");
          }
        }
      }
    }
    setReplay(pcmParts);
    if (heard) setMsg("已收到回复", true);
    else if (sawText) setMsg("主机回了文字，但没有音频（主机 TTS 可能未配置）。", true);
    else setMsg("没有收到回复，请重试。", false);
  } catch (err) {
    if (err && err.name === "AbortError") {
      setMsg("已停止", true);
    } else {
      setMsg("设备端程序没有响应，请重新打开 Voice Hub 设备端。", false);
    }
  }
  sayController = null;
  activePlayer = null;
  setBusy(false);
};

$("text").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
    ev.preventDefault();
    $("say").click();
  }
});
$("url").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") {
    ev.preventDefault();
    $("connect").click();
  }
});

let conversation = false;
let voiceWs = null;
let capture = null;
let talkPhase = "idle";

function setPhase(name, label) {
  talkPhase = name;
  const el = $("phase");
  el.textContent = label;
  el.className = "lamp " + (name === "idle" ? "off" : name === "listening" || name === "recording" ? "talk" : "on");
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

function openVoiceWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return new WebSocket(proto + "://" + location.host + "/ws/voice");
}

async function startConversation() {
  if (!connected) {
    setMsg("请先连接主机。", false);
    return;
  }
  conversation = true;
  $("talk").textContent = "结束对话";
  setBusy(true);
  $("talk").disabled = false;
  $("stop").disabled = false;
  const player = createPcmPlayer(16000);
  activePlayer = player;
  let recording = false;
  let voiced = false;
  let silenceMs = 0;
  let voicedMs = 0;
  let recStarted = 0;
  const SILENCE_END = 800;
  const MIN_SPEECH = 200;
  const MAX_UTTER = 12000;
  const ECHO_TAIL = 400;
  let pendingMs = 0;
  let noise = 0.004;

  function isSpeech(energy) {
    noise = Math.min(0.02, noise * 0.97 + energy * 0.03);
    return energy >= Math.max(0.012, noise * 4.5);
  }

  function isQuiet(energy) {
    return energy < Math.max(0.007, noise * 2.0);
  }

  function sendListen(state) {
    if (voiceWs && voiceWs.readyState === 1) {
      voiceWs.send(JSON.stringify({ type: "listen", state: state }));
    }
  }

  function beginUtterance() {
    if (recording || !conversation) return;
    recording = true;
    voiced = false;
    silenceMs = 0;
    voicedMs = 0;
    recStarted = performance.now();
    sendListen("start");
    setPhase("recording", "正在听…");
    setMsg("请说话，停顿后自动发送。", true);
  }

  function endUtterance() {
    if (!recording) return;
    recording = false;
    sendListen("stop");
    setPhase("waiting", "等待回复…");
    setMsg("已说完，正在识别…", true);
    window.setTimeout(() => {
      if (conversation && talkPhase === "waiting") {
        setPhase("listening", "你说，停顿后发送");
        setMsg("这轮没有等到回复，请再说一次。", false);
      }
    }, 12000);
  }

  try {
    voiceWs = openVoiceWs();
    await new Promise((resolve, reject) => {
      voiceWs.onopen = resolve;
      voiceWs.onerror = () => reject(new Error("ws"));
    });
    capture = await startCapture((bytes, energy) => {
      if (!conversation || talkPhase === "playing" || talkPhase === "waiting") return;
      const now = performance.now();
      const dt = 1000 * (bytes.length / 2) / 16000;
      if (!recording) {
        if (isSpeech(energy)) {
          pendingMs += dt;
          if (pendingMs >= MIN_SPEECH) beginUtterance();
        } else {
          pendingMs = 0;
          setPhase("listening", "等待你说话");
        }
        if (!recording) return;
      }
      if (recording && voiceWs && voiceWs.readyState === 1) voiceWs.send(bytes);
      if (isSpeech(energy)) {
        voiced = true;
        voicedMs += dt;
        silenceMs = 0;
      } else if (voiced && isQuiet(energy)) {
        silenceMs += dt;
      }
      if (recording && voiced && voicedMs >= MIN_SPEECH && silenceMs >= SILENCE_END) endUtterance();
      if (recording && now - recStarted > MAX_UTTER) endUtterance();
    });
    setPhase("listening", "等待你说话");
    setMsg("连续对话已开始。对着麦克风说话即可。", true);

    voiceWs.onmessage = async (ev) => {
      if (typeof ev.data !== "string") {
        const buf = ev.data instanceof Blob ? await ev.data.arrayBuffer() : ev.data;
        const bytes = new Uint8Array(buf);
        appendEvent({ type: "audio", bytes: bytes.length });
        setPhase("playing", "正在播放");
        try {
          await player.playChunk(bytes);
        } catch (err) {
          setMsg("浏览器拦住了自动播放，请点下方播放键。", true);
        }
        return;
      }
      const evj = JSON.parse(ev.data);
      appendEvent(evj);
      if (evj.type === "stt") setMsg("识别：" + (evj.text || ""), true);
      if (evj.type === "tts_text") setMsg("回复：" + (evj.text || ""), true);
      if (evj.type === "error") {
        setMsg(evj.message || "对话出错", false);
        if ((evj.message || "").includes("连")) setLamp(false, "未连接");
      }
      if (evj.type === "turn_end" && conversation) {
        const wait = player.playbackDone ? player.playbackDone() : Promise.resolve();
        await wait;
        await new Promise((resolve) => setTimeout(resolve, ECHO_TAIL));
        pendingMs = 0;
        recording = false;
        if (conversation) {
          setPhase("listening", "你说，停顿后发送");
          setMsg("在听，说完停一下就会识别。", true);
        }
      }
    };
    voiceWs.onclose = () => {
      if (conversation) setMsg("语音通道断开，请重新开始对话。", false);
    };
  } catch (err) {
    conversation = false;
    $("talk").textContent = "开始连续对话";
    setBusy(false);
    if (err && err.name === "NotAllowedError") setMsg("浏览器不允许使用麦克风，请在权限里允许。", false);
    else setMsg("打不开麦克风或语音通道。", false);
    setPhase("idle", "未开始");
  }
}

async function stopConversation() {
  conversation = false;
  if (voiceWs && voiceWs.readyState === 1) {
    try {
      voiceWs.send(JSON.stringify({ type: "listen", state: "stop" }));
      voiceWs.send(JSON.stringify({ type: "abort" }));
    } catch (err) { /* ignore */ }
    voiceWs.close();
  }
  voiceWs = null;
  if (capture) {
    capture.stop();
    capture = null;
  }
  if (activePlayer) activePlayer.stop();
  await api("/api/abort", { method: "POST", body: "{}" });
  $("talk").textContent = "开始连续对话";
  setPhase("idle", "未开始");
  setBusy(false);
  setMsg("连续对话已结束。", true);
}

$("talk").onclick = async () => {
  if (conversation) await stopConversation();
  else await startConversation();
};

function renderMemory(data) {
  if (!data || data.error) {
    $("memMsg").textContent = data && data.error ? data.error : "";
    $("memMsg").className = "msg err";
    return;
  }
  $("memCity").value = data.city || "";
  $("memName").value = data.name || "";
  $("memPath").textContent = data.path ? "主机文件：" + data.path : "记忆存在主机上，按机械臂区分。";
  const list = $("memTurns");
  list.innerHTML = "";
  (data.turns || []).slice().reverse().forEach((item) => {
    const li = document.createElement("li");
    li.textContent = (item.role === "user" ? "你：" : "助手：") + (item.content || "");
    list.appendChild(li);
  });
  $("memMsg").textContent = "已加载";
  $("memMsg").className = "msg ok";
}

async function loadMemory() {
  if (!connected) {
    $("memMsg").textContent = "请先连接主机。";
    $("memMsg").className = "msg err";
    return;
  }
  const data = await api("/api/memory");
  renderMemory(data);
}

$("memLoad").onclick = () => loadMemory();
$("memSave").onclick = async () => {
  const data = await api("/api/memory", {
    method: "PUT",
    body: JSON.stringify({ city: $("memCity").value.trim(), name: $("memName").value.trim() }),
  });
  renderMemory(data);
  if (data && data.ok) {
    $("memMsg").textContent = "记忆已保存。下次问天气会用这个城市。";
    $("memMsg").className = "msg ok";
  }
};
$("memClear").onclick = async () => {
  if (!window.confirm("清空这台机械臂上的对话记忆？")) return;
  const data = await api("/api/memory", { method: "DELETE" });
  renderMemory(data);
  if (data && data.ok) {
    $("memCity").value = "";
    $("memName").value = "";
    $("memMsg").textContent = "记忆已清空。";
    $("memMsg").className = "msg ok";
  }
};

const _origConnect = $("connect").onclick;
$("connect").onclick = async () => {
  await _origConnect();
  if (connected) loadMemory();
};

document.addEventListener("DOMContentLoaded", async () => {
  const settings = await api("/api/settings");
  applySettings(settings);
  await refreshStatus();
  if (connected) loadMemory();
  setInterval(refreshStatus, 4000);
});
