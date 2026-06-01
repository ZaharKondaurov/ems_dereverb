/**
 * FSPEN live client: mic → WebSocket → enhanced audio + scrolling spectrograms.
 */

const N_COLS = 512;
const N_FREQ_DEFAULT = 513;
/** Rendered spectrogram height (downsampled from full STFT bins). */
const SPEC_DISPLAY_ROWS = 256;
const TARGET_SR = 48000;
const AUDIO_BUFFER_SIZE = 4096;
/** Start playback after this much audio is buffered (absorbs inference jitter). */
const MIN_PLAY_SAMPLES = Math.floor(TARGET_SR * 0.15);
const PLAY_RING_CAP = TARGET_SR * 4;
/** Headphone output gain (browser DAC often runs hot). */
const PLAYBACK_GAIN = 0.82;
function dbToColor(db) {
  const t = Math.max(0, Math.min(1, (db + 80) / 80));
  const r = Math.min(255, Math.max(0, 255 * Math.min(1, t * 2.5)));
  const g = Math.min(255, Math.max(0, 255 * Math.max(0, (t - 0.2) * 1.5)));
  const b = Math.min(255, Math.max(0, 255 * Math.max(0, (t - 0.6) * 2.5)));
  return [r | 0, g | 0, b | 0];
}

function resampleLinear(samples, fromRate, toRate) {
  if (fromRate === toRate || samples.length === 0) {
    return samples;
  }
  const outLen = Math.max(1, Math.round((samples.length * toRate) / fromRate));
  const out = new Float32Array(outLen);
  const ratio = fromRate / toRate;
  for (let i = 0; i < outLen; i++) {
    const src = i * ratio;
    const j = Math.floor(src);
    const frac = src - j;
    const a = samples[j] ?? 0;
    const b = samples[j + 1] ?? a;
    out[i] = a * (1 - frac) + b * frac;
  }
  return out;
}

/** Max-pool STFT column to fewer display rows. */
function downsampleColumn(col, nFreq, nRows) {
  const out = new Float32Array(nRows);
  for (let r = 0; r < nRows; r++) {
    const f0 = Math.floor((r * nFreq) / nRows);
    const f1 = Math.floor(((r + 1) * nFreq) / nRows);
    let m = -80;
    for (let f = f0; f < f1; f++) {
      if (col[f] > m) m = col[f];
    }
    out[r] = m;
  }
  return out;
}

class ScrollingSpectrogram {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.nFreq = N_FREQ_DEFAULT;
    this.nRows = SPEC_DISPLAY_ROWS;
    this.nCols = N_COLS;
    this.data = new Float32Array(this.nRows * this.nCols);
    this.data.fill(-80);
    this.imageData = this.ctx.createImageData(this.nCols, this.nRows);
  }

  setFreqBins(n) {
    if (n === this.nFreq) return;
    this.nFreq = n;
  }

  pushColumns(cols) {
    if (!cols || !cols.length) return;
    const w = this.nCols;
    const nRows = this.nRows;
    for (const col of cols) {
      if (col.length !== this.nFreq) continue;
      const dcol = downsampleColumn(col, this.nFreq, nRows);
      for (let f = 0; f < nRows; f++) {
        const row = f * w;
        this.data.copyWithin(row, row + 1, row + w);
        this.data[row + w - 1] = dcol[f];
      }
    }
    this.draw();
  }

  draw() {
    const nRows = this.nRows;
    const w = this.nCols;
    const px = this.imageData.data;
    for (let f = 0; f < nRows; f++) {
      const row = nRows - 1 - f;
      for (let t = 0; t < w; t++) {
        const db = this.data[f * w + t];
        const [r, g, b] = dbToColor(db);
        const i = (row * w + t) * 4;
        px[i] = r;
        px[i + 1] = g;
        px[i + 2] = b;
        px[i + 3] = 255;
      }
    }
    this.ctx.putImageData(this.imageData, 0, 0);
  }

  clear() {
    this.data.fill(-80);
    this.draw();
  }
}

/** Continuous playback from a ring buffer (avoids gaps between WebSocket chunks). */
class StreamPlayback {
  constructor(audioContext) {
    this.ctx = audioContext;
    this.ring = new Float32Array(PLAY_RING_CAP);
    this.w = 0;
    this.r = 0;
    this.available = 0;
    this.lastSample = 0;
    this.primed = false;

    this.gain = this.ctx.createGain();
    this.gain.gain.value = PLAYBACK_GAIN;

    this.node = this.ctx.createScriptProcessor(2048, 0, 1);
    this.node.onaudioprocess = (e) => {
      const out = e.outputBuffer.getChannelData(0);
      for (let i = 0; i < out.length; i++) {
        if (this.primed && this.available > 0) {
          out[i] = this._readOne();
          this.lastSample = out[i];
        } else {
          out[i] = this.lastSample;
        }
      }
    };
    this.node.connect(this.gain);
    this.gain.connect(this.ctx.destination);
  }

  _readOne() {
    const s = this.ring[this.r];
    this.r = (this.r + 1) % PLAY_RING_CAP;
    this.available--;
    return s;
  }

  _writeOne(s) {
    if (this.available >= PLAY_RING_CAP - 1) {
      return false;
    }
    this.ring[this.w] = s;
    this.w = (this.w + 1) % PLAY_RING_CAP;
    this.available++;
    return true;
  }

  reset() {
    this.w = 0;
    this.r = 0;
    this.available = 0;
    this.lastSample = 0;
    this.primed = false;
  }

  push(samples) {
    if (!samples.length) return;
    if (this.available + samples.length >= PLAY_RING_CAP - 1) {
      return;
    }

    for (let i = 0; i < samples.length; i++) {
      if (!this._writeOne(samples[i])) break;
    }
    if (!this.primed && this.available >= MIN_PLAY_SAMPLES) {
      this.primed = true;
    }
  }

  disconnect() {
    this.node.disconnect();
    this.node.onaudioprocess = null;
  }
}

const specIn = new ScrollingSpectrogram(document.getElementById("specIn"));
const specOut = new ScrollingSpectrogram(document.getElementById("specOut"));

const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");
const chkEnhanced = document.getElementById("chkEnhanced");
const statusEl = document.getElementById("status");

let ws = null;
let audioCtx = null;
let captureSampleRate = TARGET_SR;
let mediaStream = null;
let captureProcessor = null;
let playback = null;
let running = false;
let levelSmooth = 0;

function setStatus(text) {
  statusEl.textContent = text;
}

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws`;
}

function sendConfig() {
  if (playback) {
    playback.reset();
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: "config", enhanced: chkEnhanced.checked }));
  }
}

function onAudioProcess(e) {
  if (!running || !ws || ws.readyState !== WebSocket.OPEN) return;

  let input = e.inputBuffer.getChannelData(0);
  input = resampleLinear(input, captureSampleRate, TARGET_SR);

  let peak = 0;
  for (let i = 0; i < input.length; i++) {
    const v = Math.abs(input[i]);
    if (v > peak) peak = v;
  }
  levelSmooth = levelSmooth * 0.9 + peak * 0.1;

  ws.send(
    JSON.stringify({
      type: "audio",
      data: Array.from(input),
      sr: TARGET_SR,
    })
  );
}

async function initAudio() {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Microphone API unavailable (use HTTPS or localhost)");
  }

  audioCtx = new AudioContext();
  await audioCtx.resume();
  captureSampleRate = audioCtx.sampleRate;

  playback = new StreamPlayback(audioCtx);

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });

  const source = audioCtx.createMediaStreamSource(mediaStream);
  captureProcessor = audioCtx.createScriptProcessor(AUDIO_BUFFER_SIZE, 1, 1);
  captureProcessor.onaudioprocess = onAudioProcess;

  source.connect(captureProcessor);
  const silent = audioCtx.createGain();
  silent.gain.value = 0;
  captureProcessor.connect(silent);
  silent.connect(audioCtx.destination);
}

async function start() {
  if (running) return;
  running = true;
  btnStart.disabled = true;
  btnStop.disabled = false;
  specIn.clear();
  specOut.clear();
  levelSmooth = 0;

  try {
    setStatus("Allow microphone access…");
    await initAudio();

    setStatus("Connecting…");
    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      playback.reset();
      ws.send(JSON.stringify({ type: "reset" }));
      sendConfig();
      const srNote =
        captureSampleRate !== TARGET_SR
          ? ` · resampled ${captureSampleRate}→${TARGET_SR} Hz`
          : "";
      setStatus(`Streaming — use headphones${srNote}`);
    };

    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.type === "status" && msg.flush_playback && playback) {
        playback.reset();
        return;
      }
      if (msg.type !== "result") return;

      if (msg.n_freq) {
        specIn.setFreqBins(msg.n_freq);
        specOut.setFreqBins(msg.n_freq);
      }
      specIn.pushColumns(msg.spec_in_cols);
      specOut.pushColumns(msg.spec_out_cols);

      const audio = new Float32Array(msg.audio || []);
      playback.push(audio);

      const mode = msg.enhanced ? "enhanced" : "bypass";
      const warm = msg.warmup ? "ready" : "warmup";
      const lvl = (levelSmooth * 100).toFixed(0);
      const bufMs = ((playback.available / TARGET_SR) * 1000).toFixed(0);
      const q = msg.out_q != null ? ` · q ${msg.out_q}` : "";
      setStatus(`${mode} · ${warm} · mic ${lvl}% · buf ${bufMs}ms${q}`);
    };

    ws.onclose = () => {
      setStatus("Disconnected");
      stop(false);
    };

    ws.onerror = () => setStatus("WebSocket error");
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    stop(false);
  }
}

function stop(user = true) {
  running = false;
  btnStart.disabled = false;
  btnStop.disabled = true;

  if (captureProcessor) {
    captureProcessor.onaudioprocess = null;
    captureProcessor.disconnect();
    captureProcessor = null;
  }
  if (playback) {
    playback.disconnect();
    playback = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  if (user) setStatus("Stopped");
}

btnStart.addEventListener("click", () => start());
btnStop.addEventListener("click", () => stop());
chkEnhanced.addEventListener("change", sendConfig);
