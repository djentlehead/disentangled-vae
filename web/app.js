/* Style Transfer Studio -- frontend.
 *
 * Talks to the FastAPI backend in server/main.py:
 *   GET  /api/health        -> model status + window/pitch-range metadata
 *   GET  /api/examples      -> bundled example MIDI filenames
 *   POST /api/roll-preview  -> binarized piano roll for a file/example, pre-generate
 *   POST /api/generate      -> runs the model, returns roll + latents + midi + audio
 *
 * No build step -- vanilla JS, talks to the API with fetch()/FormData.
 */

(() => {
  "use strict";

  const COLORS = {
    rhythm: "#5CD9B5", // lcd-glow
    pitch:  "#E8401A", // signal
  };

  const el = (id) => document.getElementById(id);

  const dom = {
    healthDot: el("healthDot"),
    healthText: el("healthText"),

    infoToggle: el("infoToggle"),
    infoContent: el("infoContent"),
    infoSeconds: el("infoSeconds"),
    infoPitchRange: el("infoPitchRange"),

    segBtns: Array.from(document.querySelectorAll(".seg-btn")),
    recombineGroup: el("recombineGroup"),
    exploreGroup: el("exploreGroup"),

    contentZone: el("contentZone"),
    contentFileInput: el("contentFileInput"),
    contentBrowseBtn: el("contentBrowseBtn"),
    contentExampleSelect: el("contentExampleSelect"),
    contentEmptyView: el("contentEmptyView"),
    contentFilledView: el("contentFilledView"),
    contentFilename: el("contentFilename"),
    contentClearBtn: el("contentClearBtn"),
    contentPreviewCanvas: el("contentPreviewCanvas"),

    styleZone: el("styleZone"),
    styleFileInput: el("styleFileInput"),
    styleBrowseBtn: el("styleBrowseBtn"),
    styleExampleSelect: el("styleExampleSelect"),
    styleEmptyView: el("styleEmptyView"),
    styleFilledView: el("styleFilledView"),
    styleFilename: el("styleFilename"),
    styleClearBtn: el("styleClearBtn"),
    stylePreviewCanvas: el("stylePreviewCanvas"),

    rhythmScale: el("rhythmScale"), rhythmScaleVal: el("rhythmScaleVal"),
    rhythmNoise: el("rhythmNoise"), rhythmNoiseVal: el("rhythmNoiseVal"),
    pitchScale: el("pitchScale"), pitchScaleVal: el("pitchScaleVal"),
    pitchNoise: el("pitchNoise"), pitchNoiseVal: el("pitchNoiseVal"),

    rhythmSigma: el("rhythmSigma"), rhythmSigmaVal: el("rhythmSigmaVal"),
    pitchSigma: el("pitchSigma"), pitchSigmaVal: el("pitchSigmaVal"),
    seedInput: el("seedInput"),
    randomSeedBtn: el("randomSeedBtn"),

    threshold: el("threshold"), thresholdVal: el("thresholdVal"),
    minLen: el("minLen"), minLenVal: el("minLenVal"),
    gapMerge: el("gapMerge"), gapMergeVal: el("gapMergeVal"),
    beatSteps: el("beatSteps"), beatStepsVal: el("beatStepsVal"),
    bpmInput: el("bpmInput"),

    resetBtn: el("resetBtn"),
    generateBtn: el("generateBtn"),
    generateBtnLabel: el("generateBtnLabel"),
    errorBanner: el("errorBanner"),

    outputEmpty: el("outputEmpty"),
    outputResult: el("outputResult"),
    outputModeLabel: el("outputModeLabel"),
    statNotes: el("statNotes"),
    statDensity: el("statDensity"),
    rollCanvas: el("rollCanvas"),
    rhythmBarCanvas: el("rhythmBarCanvas"),
    pitchBarCanvas: el("pitchBarCanvas"),
    audioPlayer: el("audioPlayer"),
    audioCaption: el("audioCaption"),
    downloadBtn: el("downloadBtn"),
    outputFooter: el("outputFooter"),
  };

  const state = {
    mode: "recombine",
    content: null, // { type: 'file', file } | { type: 'example', name }
    style: null,
    healthy: false,
    busy: false,
    meta: { fs: 8, seqLen: 256, pitchLo: 40, nPitch: 32 },
    lastResult: null,
    lastModeLabel: "",
    lastContentRoll: null,
    lastStyleRoll: null,
  };

  // ---------- canvas helpers ----------

  function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas._cssW = w;
    canvas._cssH = h;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return ctx;
  }

  function drawRoll(canvas, roll, { mode = "mono", color = "#4c9be8", grid = false } = {}) {
    if (!roll || !roll.length) return;
    const ctx = fitCanvas(canvas);
    const w = canvas._cssW, h = canvas._cssH;
    ctx.clearRect(0, 0, w, h);

    const rows = roll.length;
    const cols = roll[0].length;
    const cellW = w / rows;
    const cellH = h / cols;

    if (grid) {
      ctx.strokeStyle = "rgba(255,255,255,0.06)";
      ctx.lineWidth = 1;
      const fs = state.meta.fs || 8;
      for (let t = 0; t <= rows; t += fs) {
        const x = Math.round(t * cellW) + 0.5;
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
      }
      for (let p = 0; p <= cols; p += 12) {
        const y = Math.round(h - p * cellH) + 0.5;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      }
    }

    for (let t = 0; t < rows; t++) {
      const row = roll[t];
      for (let p = 0; p < cols; p++) {
        if (!row[p]) continue;
        const x = t * cellW;
        const y = h - (p + 1) * cellH;
        if (mode === "gradient") {
          // lcd teal: dark at low pitch, bright lcd-glow at high pitch
          const lightness = 35 + (p / cols) * 40;
          ctx.fillStyle = `hsl(162 65% ${lightness}%)`;
        } else {
          ctx.fillStyle = color;
        }
        ctx.fillRect(x, y, Math.max(1, cellW - 0.4), Math.max(1, cellH - 0.4));
      }
    }
  }

  function drawLatentBars(canvas, values, color) {
    if (!values || !values.length) return;
    const ctx = fitCanvas(canvas);
    const w = canvas._cssW, h = canvas._cssH;
    ctx.clearRect(0, 0, w, h);

    const n = values.length;
    const barW = w / n;
    const mid = h / 2;
    const maxAbs = Math.max(1e-6, ...values.map((v) => Math.abs(v)));

    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, mid + 0.5); ctx.lineTo(w, mid + 0.5); ctx.stroke();

    ctx.fillStyle = color;
    for (let i = 0; i < n; i++) {
      const v = values[i];
      const barH = (Math.abs(v) / maxAbs) * (h / 2 - 4);
      const x = i * barW;
      const y = v >= 0 ? mid - barH : mid;
      ctx.fillRect(x, y, Math.max(1, barW - 1), Math.max(1, barH));
    }
  }

  // ---------- slider binding ----------

  function bindSlider(input, output, formatFn) {
    const update = () => { output.textContent = formatFn ? formatFn(input.value) : input.value; };
    input.addEventListener("input", update);
    update();
  }

  bindSlider(dom.rhythmScale, dom.rhythmScaleVal, (v) => parseFloat(v).toFixed(2));
  bindSlider(dom.rhythmNoise, dom.rhythmNoiseVal, (v) => parseFloat(v).toFixed(2));
  bindSlider(dom.pitchScale, dom.pitchScaleVal, (v) => parseFloat(v).toFixed(2));
  bindSlider(dom.pitchNoise, dom.pitchNoiseVal, (v) => parseFloat(v).toFixed(2));
  bindSlider(dom.rhythmSigma, dom.rhythmSigmaVal, (v) => parseFloat(v).toFixed(1));
  bindSlider(dom.pitchSigma, dom.pitchSigmaVal, (v) => parseFloat(v).toFixed(1));
  bindSlider(dom.threshold, dom.thresholdVal, (v) => parseFloat(v).toFixed(2));
  bindSlider(dom.minLen, dom.minLenVal);
  bindSlider(dom.gapMerge, dom.gapMergeVal);
  bindSlider(dom.beatSteps, dom.beatStepsVal, (v) => (v === "0" ? "off" : v));

  dom.randomSeedBtn.addEventListener("click", () => {
    dom.seedInput.value = Math.floor(Math.random() * 1_000_000);
  });

  // ---------- reset to defaults ----------

  const PARAM_DEFAULTS = {
    rhythmScale: "1", rhythmNoise: "0",
    pitchScale: "1", pitchNoise: "0",
    rhythmSigma: "1", pitchSigma: "1", seedInput: "42",
    threshold: "0.35", minLen: "2", gapMerge: "2", beatSteps: "0", bpmInput: "120",
  };

  dom.resetBtn.addEventListener("click", () => {
    for (const [key, value] of Object.entries(PARAM_DEFAULTS)) {
      const input = dom[key];
      input.value = value;
      input.dispatchEvent(new Event("input"));
    }
    document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
  });

  // ---------- presets ----------

  const EXPLORE_PRESETS = {
    // σ controls generation range; threshold/cleanup tuned to match expected output density
    subtle:     { rhythmSigma: "0.5", pitchSigma: "0.5",  threshold: "0.45", minLen: "3", gapMerge: "2", beatSteps: "0" },
    balanced:   { rhythmSigma: "1.0", pitchSigma: "1.0",  threshold: "0.35", minLen: "2", gapMerge: "2", beatSteps: "0" },
    expressive: { rhythmSigma: "1.5", pitchSigma: "1.5",  threshold: "0.30", minLen: "2", gapMerge: "3", beatSteps: "0" },
    wild:       { rhythmSigma: "2.5", pitchSigma: "2.5",  threshold: "0.25", minLen: "1", gapMerge: "1", beatSteps: "0" },
  };
  const RECOMBINE_PRESETS = {
    faithful:  { rhythmScale: "1.0", rhythmNoise: "0.0",  pitchScale: "1.0", pitchNoise: "0.0",  threshold: "0.40", minLen: "2", gapMerge: "2", beatSteps: "0" },
    hybrid:    { rhythmScale: "1.0", rhythmNoise: "0.15", pitchScale: "1.0", pitchNoise: "0.15", threshold: "0.35", minLen: "2", gapMerge: "2", beatSteps: "0" },
    amplified: { rhythmScale: "1.4", rhythmNoise: "0.1",  pitchScale: "1.4", pitchNoise: "0.1",  threshold: "0.30", minLen: "2", gapMerge: "3", beatSteps: "0" },
  };

  document.querySelectorAll(".preset-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const group = chip.dataset.group;
      const vals = group === "explore"
        ? EXPLORE_PRESETS[chip.dataset.preset]
        : RECOMBINE_PRESETS[chip.dataset.preset];
      if (!vals) return;
      for (const [key, value] of Object.entries(vals)) {
        const input = dom[key];
        if (!input) continue;
        input.value = value;
        input.dispatchEvent(new Event("input"));
      }
      document.querySelectorAll(`.preset-chip[data-group="${group}"]`)
        .forEach((c) => c.classList.toggle("active", c === chip));
    });
  });

  // ---------- info banner ----------

  (function initInfoBanner() {
    const stored = localStorage.getItem("sts_info_expanded");
    let expanded = stored === null ? true : stored === "true";
    const apply = () => {
      dom.infoToggle.setAttribute("aria-expanded", String(expanded));
      dom.infoContent.hidden = !expanded;
    };
    apply();
    dom.infoToggle.addEventListener("click", () => {
      expanded = !expanded;
      apply();
      localStorage.setItem("sts_info_expanded", String(expanded));
    });
  })();

  // ---------- mode switch ----------

  dom.segBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      dom.segBtns.forEach((b) => { b.classList.remove("active"); b.setAttribute("aria-selected", "false"); });
      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      state.mode = btn.dataset.mode;
      const exploring = state.mode === "explore";
      dom.recombineGroup.hidden = exploring;
      dom.exploreGroup.hidden = !exploring;
      clearError();
    });
  });

  // ---------- dropzones ----------

  function setupDropzone(opts) {
    const {
      zone, fileInput, browseBtn, exampleSelect,
      emptyView, filledView, filenameEl, clearBtn, previewCanvas,
      previewColor, onChange,
    } = opts;

    function acceptFile(file) {
      if (!/\.(mid|midi)$/i.test(file.name)) {
        showError(`"${file.name}" doesn't look like a MIDI file (expected .mid/.midi).`);
        return;
      }
      exampleSelect.value = "";
      onChange({ type: "file", file });
      filenameEl.textContent = file.name;
      emptyView.hidden = true;
      filledView.hidden = false;
      previewFor(file, null, previewCanvas, previewColor);
    }

    browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
    fileInput.addEventListener("change", () => {
      if (fileInput.files.length) acceptFile(fileInput.files[0]);
    });

    emptyView.addEventListener("click", (e) => {
      if (e.target.closest("select") || e.target.closest("button")) return;
      fileInput.click();
    });
    zone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
    });

    exampleSelect.addEventListener("click", (e) => e.stopPropagation());
    exampleSelect.addEventListener("change", () => {
      const name = exampleSelect.value;
      if (!name) return;
      fileInput.value = "";
      onChange({ type: "example", name });
      filenameEl.textContent = name;
      emptyView.hidden = true;
      filledView.hidden = false;
      previewFor(null, name, previewCanvas, previewColor);
    });

    clearBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.value = "";
      exampleSelect.value = "";
      onChange(null);
      filledView.hidden = true;
      emptyView.hidden = false;
    });

    ["dragenter", "dragover"].forEach((evt) =>
      zone.addEventListener(evt, (e) => { e.preventDefault(); zone.classList.add("dragover"); })
    );
    ["dragleave", "dragend", "drop"].forEach((evt) =>
      zone.addEventListener(evt, () => zone.classList.remove("dragover"))
    );
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer.files.length) acceptFile(e.dataTransfer.files[0]);
    });
  }

  async function previewFor(file, exampleName, canvas, color) {
    try {
      const fd = new FormData();
      if (file) fd.append("midi", file);
      else fd.append("example", exampleName);
      const res = await fetch("/api/roll-preview", { method: "POST", body: fd });
      if (!res.ok) return;
      const data = await res.json();
      canvas.parentElement.hidden = false;
      drawRoll(canvas, data.roll, { mode: "mono", color, grid: false });
      if (canvas === dom.contentPreviewCanvas) state.lastContentRoll = data.roll;
      if (canvas === dom.stylePreviewCanvas) state.lastStyleRoll = data.roll;
    } catch (_) {
      // Preview is a nice-to-have; silently skip on failure, Generate will surface real errors.
    }
  }

  setupDropzone({
    zone: dom.contentZone, fileInput: dom.contentFileInput, browseBtn: dom.contentBrowseBtn,
    exampleSelect: dom.contentExampleSelect, emptyView: dom.contentEmptyView,
    filledView: dom.contentFilledView, filenameEl: dom.contentFilename,
    clearBtn: dom.contentClearBtn, previewCanvas: dom.contentPreviewCanvas,
    previewColor: COLORS.rhythm,
    onChange: (src) => { state.content = src; },
  });

  setupDropzone({
    zone: dom.styleZone, fileInput: dom.styleFileInput, browseBtn: dom.styleBrowseBtn,
    exampleSelect: dom.styleExampleSelect, emptyView: dom.styleEmptyView,
    filledView: dom.styleFilledView, filenameEl: dom.styleFilename,
    clearBtn: dom.styleClearBtn, previewCanvas: dom.stylePreviewCanvas,
    previewColor: COLORS.pitch,
    onChange: (src) => { state.style = src; },
  });

  // ---------- errors ----------

  function showError(msg) {
    dom.errorBanner.textContent = msg;
    dom.errorBanner.hidden = false;
  }
  function clearError() {
    dom.errorBanner.hidden = true;
    dom.errorBanner.textContent = "";
  }

  // ---------- health ----------

  function formatParams(n) {
    if (!n) return "0";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(0) + "K";
    return String(n);
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      state.healthy = !!data.ok;
      if (data.ok) {
        dom.healthDot.className = "status-dot ok";
        dom.healthText.textContent = `Model loaded · ${formatParams(data.params)} params`;
        state.meta = {
          fs: data.fs || 8, seqLen: data.seq_len || 256,
          pitchLo: data.pitch_lo || 40, nPitch: data.n_pitch || 32,
        };
        dom.infoSeconds.textContent = Math.round(state.meta.seqLen / state.meta.fs);
        dom.infoPitchRange.textContent = `${state.meta.pitchLo}–${state.meta.pitchLo + state.meta.nPitch - 1}`;
      } else {
        dom.healthDot.className = "status-dot err";
        dom.healthText.textContent = data.error ? "Model failed to load" : "Model not loaded";
      }
    } catch (_) {
      state.healthy = false;
      dom.healthDot.className = "status-dot err";
      dom.healthText.textContent = "Server offline";
    }
    updateGenerateAvailability();
  }

  function updateGenerateAvailability() {
    dom.generateBtn.disabled = state.busy || !state.healthy;
  }

  // ---------- examples ----------

  async function loadExamples() {
    try {
      const res = await fetch("/api/examples");
      const data = await res.json();
      for (const select of [dom.contentExampleSelect, dom.styleExampleSelect]) {
        for (const name of data.examples || []) {
          const opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          select.appendChild(opt);
        }
      }
    } catch (_) {
      // Examples are optional convenience -- fine if this fails.
    }
  }

  // ---------- generate ----------

  function describeSource(src) {
    if (!src) return null;
    return src.type === "file" ? src.file.name : src.name;
  }

  function buildModeLabel() {
    if (state.mode === "explore") {
      return `Exploration — σ_r=${parseFloat(dom.rhythmSigma.value).toFixed(1)}, ` +
             `σ_p=${parseFloat(dom.pitchSigma.value).toFixed(1)}, seed=${dom.seedInput.value}`;
    }
    const c = describeSource(state.content);
    const s = describeSource(state.style);
    return s ? `Recombination — rhythm: ${c}, pitch: ${s}` : `Reconstruction — ${c}`;
  }

  function setBusy(busy) {
    state.busy = busy;
    dom.generateBtn.classList.toggle("busy", busy);
    dom.generateBtnLabel.textContent = busy ? "Generating…" : "Generate";
    updateGenerateAvailability();
  }

  async function generate() {
    clearError();
    if (state.mode === "recombine" && !state.content) {
      showError("Add a Content MIDI file, or pick an example, first.");
      return;
    }

    const fd = new FormData();
    fd.append("exploration", state.mode === "explore" ? "true" : "false");
    fd.append("threshold", dom.threshold.value);
    fd.append("min_len", dom.minLen.value);
    fd.append("gap_merge", dom.gapMerge.value);
    fd.append("beat_steps", dom.beatSteps.value);
    fd.append("bpm", dom.bpmInput.value);

    if (state.mode === "explore") {
      fd.append("rhythm_sigma", dom.rhythmSigma.value);
      fd.append("pitch_sigma", dom.pitchSigma.value);
      fd.append("seed", dom.seedInput.value);
    } else {
      if (state.content.type === "file") fd.append("content_midi", state.content.file);
      else fd.append("content_example", state.content.name);
      if (state.style) {
        if (state.style.type === "file") fd.append("style_midi", state.style.file);
        else fd.append("style_example", state.style.name);
      }
      fd.append("rhythm_scale", dom.rhythmScale.value);
      fd.append("rhythm_noise", dom.rhythmNoise.value);
      fd.append("pitch_scale", dom.pitchScale.value);
      fd.append("pitch_noise", dom.pitchNoise.value);
    }

    const modeLabel = buildModeLabel();
    setBusy(true);
    try {
      const res = await fetch("/api/generate", { method: "POST", body: fd });
      const data = await res.json();
      if (!res.ok) {
        showError(data.detail || `Generation failed (HTTP ${res.status}).`);
        return;
      }
      renderResult(data, modeLabel);
    } catch (_) {
      showError("Couldn't reach the server. Is it still running?");
    } finally {
      setBusy(false);
    }
  }

  function renderResult(data, modeLabel) {
    state.lastResult = data;
    state.lastModeLabel = modeLabel;

    dom.outputEmpty.hidden = true;
    dom.outputResult.hidden = false;

    dom.outputModeLabel.textContent = modeLabel;
    dom.statNotes.textContent = `${data.note_count} notes`;
    dom.statDensity.textContent = `density ${data.density.toFixed(3)}`;

    drawRoll(dom.rollCanvas, data.roll, { mode: "gradient", grid: true });
    drawLatentBars(dom.rhythmBarCanvas, data.z_rhythm, COLORS.rhythm);
    drawLatentBars(dom.pitchBarCanvas, data.z_pitch, COLORS.pitch);

    if (data.audio_base64) {
      dom.audioPlayer.src = `data:audio/wav;base64,${data.audio_base64}`;
      dom.audioPlayer.hidden = false;
      const method = data.audio_method || "";
      dom.audioCaption.textContent = method === "sine"
        ? "Sine-wave preview — FluidSynth wasn't found, so this is a rough approximation, not real piano tone. The downloaded MIDI is unaffected."
        : method.startsWith("fluidsynth:")
          ? `Soundfont: ${method.slice("fluidsynth:".length)}`
          : "";
    } else {
      dom.audioPlayer.hidden = true;
      dom.audioCaption.textContent = data.note_count > 0
        ? `Audio preview unavailable (${data.audio_method || "unknown error"}). Download the MIDI to listen.`
        : "No notes generated — lower the threshold and regenerate to hear audio.";
    }

    const bpm = dom.bpmInput.value;
    const seconds = Math.round(state.meta.seqLen / state.meta.fs);
    dom.outputFooter.textContent =
      `${seconds}s window · fs=${state.meta.fs} · BPM=${bpm} · ${data.note_count} notes · density ${data.density.toFixed(3)}`;
  }

  dom.downloadBtn.addEventListener("click", () => {
    if (!state.lastResult) return;
    const bytes = atob(state.lastResult.midi_base64);
    const buf = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
    const blob = new Blob([buf], { type: "audio/midi" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "generated.mid";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  });

  dom.generateBtn.addEventListener("click", generate);

  // ---------- resize: keep canvases crisp ----------

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (state.lastContentRoll) drawRoll(dom.contentPreviewCanvas, state.lastContentRoll, { mode: "mono", color: COLORS.rhythm });
      if (state.lastStyleRoll) drawRoll(dom.stylePreviewCanvas, state.lastStyleRoll, { mode: "mono", color: COLORS.pitch });
      if (state.lastResult) {
        drawRoll(dom.rollCanvas, state.lastResult.roll, { mode: "gradient", grid: true });
        drawLatentBars(dom.rhythmBarCanvas, state.lastResult.z_rhythm, COLORS.rhythm);
        drawLatentBars(dom.pitchBarCanvas, state.lastResult.z_pitch, COLORS.pitch);
      }
    }, 150);
  });

  // ---------- boot ----------

  checkHealth();
  setInterval(checkHealth, 5000);
  loadExamples();
})();
