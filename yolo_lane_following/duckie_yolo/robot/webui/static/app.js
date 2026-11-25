"use strict";
// Duckiebot web UI. Everything runs on the robot; this page only sends commands and shows state.

const $ = (id) => document.getElementById(id);
const S = { status: null, models: [], selectedModel: null, tab: "drive", held: new Set(), lastDriveZero: true };

// ------------------------------------------------------------------ helpers
async function api(method, path, body) {
  const opt = { method, headers: {} };
  if (body !== undefined) { opt.headers["Content-Type"] = "application/json"; opt.body = JSON.stringify(body); }
  const r = await fetch(path, opt);
  let data = {};
  try { data = await r.json(); } catch (_) { /* not JSON */ }
  if (!r.ok) throw new Error(data.error || `${r.status} ${r.statusText}`);
  return data;
}
let toastTimer = null;
function toast(msg, isError) {
  const t = $("toast");
  t.textContent = msg; t.className = "toast" + (isError ? " error" : ""); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), isError ? 6000 : 2500);
}
async function act(fn, okMsg) {
  try { const r = await fn(); if (okMsg) toast(okMsg); return r; } catch (e) { toast(e.message, true); return null; }
}
const fmtBytes = (b) => b > 1e9 ? (b / 1e9).toFixed(1) + " GB" : b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " kB";
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const typing = (el) => el && (el.tagName === "INPUT" && el.type !== "range" && el.type !== "checkbox" || el.tagName === "SELECT" || el.tagName === "TEXTAREA");

// ------------------------------------------------------------------ tabs & view
function showTab(name) {
  if (!document.getElementById("tab-" + name)) name = "drive";
  S.tab = name;
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  document.querySelectorAll(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== "tab-" + name));
  if (name === "model") loadModels();
  if (name === "data") loadSessions();
}
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
document.querySelectorAll("[data-goto]").forEach((a) => a.addEventListener("click", (e) => { e.preventDefault(); showTab(a.dataset.goto); }));
document.querySelectorAll(".seg button").forEach((b) =>
  b.addEventListener("click", () => act(() => api("POST", "/api/view", { view: b.dataset.view }))));

// keep the MJPEG stream alive across robot / server restarts
$("video").addEventListener("error", () => setTimeout(() => ($("video").src = "/stream.mjpg?" + Date.now()), 1500));

// ------------------------------------------------------------------ driving (hold to drive)
const bindRange = (id, outId, digits) => {
  const upd = () => ($(outId).textContent = Number($(id).value).toFixed(digits));
  $(id).addEventListener("input", upd); upd();
};
bindRange("speed", "speed-out", 2); bindRange("turn", "turn-out", 1);
bindRange("conf", "conf-out", 2); bindRange("maxspeed", "maxspeed-out", 2);

const KEYMAP = { w: "w", arrowup: "w", s: "s", arrowdown: "s", a: "a", arrowleft: "a", d: "d", arrowright: "d" };
function command() {
  const sp = Number($("speed").value), tr = Number($("turn").value), h = S.held;
  const v = (h.has("w") ? sp : 0) - (h.has("s") ? sp : 0);
  const w = (h.has("a") ? tr : 0) - (h.has("d") ? tr : 0);
  return { v, w };
}
function renderHeld() {
  document.querySelectorAll(".dpad [data-key]").forEach((b) => b.classList.toggle("held", S.held.has(b.dataset.key)));
}
function press(k) { if (!S.held.has(k)) { S.held.add(k); renderHeld(); sendDrive(); } }
function release(k) { if (S.held.delete(k)) { renderHeld(); sendDrive(); } }
function releaseAll() { if (S.held.size) { S.held.clear(); renderHeld(); sendDrive(); } }
function sendDrive() {
  const c = command();
  const zero = c.v === 0 && c.w === 0;
  if (zero && S.lastDriveZero) return;   // one zero is enough to end a drive
  S.lastDriveZero = zero;
  fetch("/api/drive", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(c) }).catch(() => {});
}
setInterval(() => { if (S.held.size) sendDrive(); }, 100);   // robot stops 0.3 s after the last command

async function stopAll(reason) {
  S.held.clear(); renderHeld(); S.lastDriveZero = true;
  $("autodrive").checked = false;
  try { await api("POST", "/api/stop", { reason: reason || "STOP pressed" }); } catch (e) { toast("STOP failed: " + e.message, true); }
}
$("btn-stop").addEventListener("click", () => stopAll());
$("dpad-stop").addEventListener("click", () => stopAll());

window.addEventListener("keydown", (e) => {
  if (typing(document.activeElement) || e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === " " || k === "escape") { e.preventDefault(); stopAll(); return; }
  if (k === "r" && !e.repeat) { e.preventDefault(); toggleRecording(); return; }
  if (KEYMAP[k]) { e.preventDefault(); press(KEYMAP[k]); }
});
window.addEventListener("keyup", (e) => { const k = KEYMAP[e.key.toLowerCase()]; if (k) release(k); });
window.addEventListener("blur", releaseAll);
document.addEventListener("visibilitychange", () => { if (document.hidden) releaseAll(); });

document.querySelectorAll(".dpad [data-key]").forEach((b) => {
  const k = b.dataset.key;
  b.addEventListener("pointerdown", (e) => { e.preventDefault(); b.setPointerCapture(e.pointerId); press(k); });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach((ev) => b.addEventListener(ev, () => release(k)));
  b.addEventListener("contextmenu", (e) => e.preventDefault());
});

// ------------------------------------------------------------------ recording
async function toggleRecording() {
  if (S.status && S.status.record.active) {
    await act(() => api("POST", "/api/record/stop"), "Recording stopped");
  } else {
    const r = await act(() => api("POST", "/api/record/start", { name: $("rec-name").value, hz: Number($("rec-hz").value) }));
    if (r) toast("Recording to " + r.session);
  }
  poll();
  if (S.tab === "data") loadSessions();
}
$("btn-record").addEventListener("click", toggleRecording);
// after choosing a value, give the keyboard back to driving (W A S D would otherwise go to the control)
document.querySelectorAll("select, input[type=range], input[type=checkbox]").forEach((el) =>
  el.addEventListener("change", () => el.blur()));

async function loadSessions() {
  const r = await act(() => api("GET", "/api/sessions"));
  if (!r) return;
  const box = $("sessions");
  if (!r.sessions.length) { box.innerHTML = '<p class="small muted">No recordings yet.</p>'; return; }
  box.innerHTML = r.sessions.map((s) => `
    <div class="item">
      <div class="grow">
        <div class="name">${esc(s.name)} ${s.recording ? '<span class="tag rec">recording</span>' : ""}</div>
        <div class="meta">${s.frames} frames · ${fmtBytes(s.bytes)}${s.meta.calibration && !s.meta.calibration.calibrated.intrinsics ? " · recorded before lens calibration" : ""}${s.meta.recorrected ? " · re-corrected" : ""}</div>
      </div>
      <div class="actions">
        <a href="/api/sessions/${encodeURIComponent(s.name)}.zip" download>Download</a>
        <button data-recorrect="${esc(s.name)}" ${s.recording ? "disabled" : ""} title="Rebuild rect/ from raw/ with the current calibration">Re-correct</button>
        <button data-delete="${esc(s.name)}" ${s.recording ? "disabled" : ""}>Delete</button>
      </div>
    </div>`).join("");
  box.querySelectorAll("[data-delete]").forEach((b) => b.addEventListener("click", async () => {
    if (!confirm(`Delete ${b.dataset.delete} and all its frames from the robot?`)) return;
    await act(() => api("DELETE", "/api/sessions/" + encodeURIComponent(b.dataset.delete)), "Deleted");
    loadSessions();
  }));
  box.querySelectorAll("[data-recorrect]").forEach((b) => b.addEventListener("click", async () => {
    b.disabled = true; b.textContent = "Working…";
    const r2 = await act(() => api("POST", `/api/sessions/${encodeURIComponent(b.dataset.recorrect)}/recorrect`));
    if (r2) toast(`Re-corrected ${r2.frames} frames`);
    loadSessions();
  }));
}

// ------------------------------------------------------------------ models
async function loadModels() {
  const r = await act(() => api("GET", "/api/models"));
  if (!r) return;
  S.models = r.models;
  $("no-yolo").hidden = r.ultralytics;
  const running = S.status ? S.status.model : {};
  if (!S.selectedModel || !S.models.some((m) => m.name === S.selectedModel)) {
    const pick = S.models.find((m) => m.kind === "engine") || S.models.find((m) => m.kind === "pt") || S.models[0];
    S.selectedModel = pick ? pick.name : null;
  }
  $("model-list").innerHTML = S.models.map((m) => `
    <div class="item ${m.name === S.selectedModel ? "selected" : ""}" data-model="${esc(m.name)}" role="button" tabindex="0">
      <div class="grow">
        <div class="name">${esc(m.label)}</div>
        <div class="meta">${m.kind === "color" ? "no neural network, works anywhere" : `${m.kind.toUpperCase()} · ${fmtBytes(m.bytes)} · ${new Date(m.mtime * 1000).toLocaleString()}`}${m.usable ? "" : " · needs the Docker container"}</div>
      </div>
      ${running.name === m.name && running.state !== "idle" ? `<span class="tag live">${esc(running.state)}</span>` : `<span class="tag">${esc(m.kind)}</span>`}
    </div>`).join("");
  $("model-list").querySelectorAll("[data-model]").forEach((el) => {
    const pick = () => { S.selectedModel = el.dataset.model; loadModels(); };
    el.addEventListener("click", pick);
    el.addEventListener("keydown", (e) => { if (e.key === "Enter") pick(); });
  });
}
$("btn-model-start").addEventListener("click", async () => {
  if (!S.selectedModel) return toast("Select a model first", true);
  await act(() => api("POST", "/api/model/start", { name: S.selectedModel, conf: Number($("conf").value) }), "Loading model…");
  await act(() => api("POST", "/api/view", { view: "model" }));
  poll(); loadModels();
});
$("btn-model-stop").addEventListener("click", async () => { await act(() => api("POST", "/api/model/stop"), "Model stopped"); poll(); loadModels(); });
$("btn-model-delete").addEventListener("click", async () => {
  const m = S.selectedModel;
  if (!m || m === "__color__") return toast("Select a model file first", true);
  if (!confirm(`Delete ${m} from the robot?`)) return;
  await act(() => api("DELETE", "/api/models/" + encodeURIComponent(m)), "Deleted");
  S.selectedModel = null; loadModels();
});
$("conf").addEventListener("change", () => {
  if (S.status && S.status.model.state === "running") act(() => api("POST", "/api/model/conf", { conf: Number($("conf").value) }));
});

// autodrive + heartbeat
$("autodrive").addEventListener("change", async (e) => {
  const on = e.target.checked;
  const r = await act(() => api("POST", "/api/autodrive", { enabled: on, max_speed: Number($("maxspeed").value) }), on ? "Model is driving" : "Autodrive off");
  if (!r) e.target.checked = false;
  poll();
});
$("maxspeed").addEventListener("change", () => {
  if ($("autodrive").checked) act(() => api("POST", "/api/autodrive", { enabled: true, max_speed: Number($("maxspeed").value) }));
});
setInterval(() => {
  if (S.status && S.status.autodrive.enabled && !document.hidden)
    fetch("/api/heartbeat", { method: "POST" }).catch(() => {});
}, 250);

// upload
$("btn-upload").addEventListener("click", () => {
  const f = $("upload-file").files[0];
  if (!f) return toast("Choose a .pt / .onnx / .engine file first", true);
  const name = f.name.replace(/[^A-Za-z0-9_.-]/g, "_");
  const xhr = new XMLHttpRequest();
  const bar = $("upload-progress");
  bar.hidden = false; bar.value = 0;
  xhr.upload.onprogress = (e) => { if (e.lengthComputable) bar.value = (100 * e.loaded) / e.total; };
  xhr.onload = () => {
    bar.hidden = true;
    let d = {}; try { d = JSON.parse(xhr.responseText); } catch (_) {}
    if (xhr.status === 200) { toast("Uploaded " + d.name); S.selectedModel = d.name; loadModels(); }
    else toast(d.error || "Upload failed", true);
  };
  xhr.onerror = () => { bar.hidden = true; toast("Upload failed", true); };
  xhr.open("PUT", "/api/models/upload?name=" + encodeURIComponent(name));
  xhr.send(f);
});

// TensorRT build job
$("btn-engine").addEventListener("click", async () => {
  const m = S.selectedModel;
  if (!m || !/\.(pt|onnx)$/i.test(m)) return toast("Select a .pt model first", true);
  if (!confirm(`Build a TensorRT engine from ${m}? This takes several minutes and stops the running model.`)) return;
  await act(() => api("POST", "/api/engine/build", { name: m }), "Engine build started");
  pollJob();
});
$("btn-job-cancel").addEventListener("click", () => act(() => api("POST", "/api/job/cancel"), "Cancelling…"));
let jobWasRunning = false;
async function pollJob() {
  let j;
  try { j = await api("GET", "/api/job"); } catch (_) { return; }
  const running = j.state === "running";
  $("btn-job-cancel").hidden = !running;
  $("btn-engine").disabled = running;
  $("job-state").textContent = j.state === "idle" ? "" :
    `${j.name}: ${j.state}${running ? ` (${Math.round(j.elapsed)} s)` : j.returncode !== null ? ` (exit ${j.returncode})` : ""}`;
  const log = $("job-log");
  if (j.log && j.log.length) {
    log.hidden = false;
    const atBottom = log.scrollTop + log.clientHeight >= log.scrollHeight - 8;
    log.textContent = j.log.join("\n");
    if (atBottom) log.scrollTop = log.scrollHeight;
  }
  if (jobWasRunning && !running) loadModels();
  jobWasRunning = running;
}
setInterval(() => { if (S.tab === "model" || jobWasRunning) pollJob(); }, 1000);

// ------------------------------------------------------------------ calibration
const num = (id) => { const v = $(id).value; return v === "" ? null : Number(v); };
$("btn-intr-start").addEventListener("click", () => act(() => api("POST", "/api/calib/intrinsics/start", {}), "Capturing: move the board slowly"));
$("btn-intr-stop").addEventListener("click", () => act(() => api("POST", "/api/calib/intrinsics/stop", {})));
$("btn-intr-compute").addEventListener("click", async () => {
  $("btn-intr-compute").disabled = true; $("intr-state").textContent = "Calibrating… (can take a minute)";
  const r = await act(() => api("POST", "/api/calib/intrinsics/compute", {}));
  $("btn-intr-compute").disabled = false;
  if (r) {
    $("intr-result").innerHTML = `<p><b>Done.</b> RMS error ${r.rms} px from ${r.views} views
      ${r.rms > 1 ? '<span class="warn small">high: capture again with more varied views</span>' : "(under ~0.5 px is good)"}.
      Left: raw · right: corrected (straight edges should look straight). Now redo step 2.</p>
      <img src="/calib/${encodeURIComponent(r.image)}?${Date.now()}" alt="Raw and corrected camera image">`;
    poll();
  }
});
$("btn-extr").addEventListener("click", async () => {
  try {
    const r = await api("POST", "/api/calib/extrinsics", { x0: num("x0"), y0: num("y0") });
    $("extr-result").innerHTML = `<p><b>Done.</b> Error on the floor: mean ${r.mean_mm} mm, max ${r.max_mm} mm.
      The green 10 cm grid should lie on the floor squares and the red line point straight ahead.
      ${r.intrinsics_calibrated ? "" : '<span class="warn small">Lens not calibrated yet: do step 1, then this again.</span>'}</p>
      <img src="/calib/${encodeURIComponent(r.image)}?${Date.now()}" alt="Floor grid check">`;
    poll();
  } catch (e) {
    toast(e.message, true);
    const st = S.status && S.status.calib.results.extrinsics;
    $("extr-result").innerHTML = `<p class="warn small">${esc(e.message)}. Is the whole board visible, flat and evenly lit?
      This is what the camera saw:</p>` + (st && st.image ? `<img src="/calib/${encodeURIComponent(st.image)}?${Date.now()}" alt="Camera view">` : "");
  }
});
$("btn-w-straight").addEventListener("click", () =>
  act(() => api("POST", "/api/calib/wheels/run", { kind: "straight", v: num("w-v"), duration: num("w-dur") }), "Driving straight in 2 s…"));
$("btn-w-apply").addEventListener("click", async () => {
  const r = await act(() => api("POST", "/api/calib/wheels/apply", { kind: "straight", v: num("w-v"), duration: num("w-dur"), drift: num("w-drift"), distance: num("w-dist") }));
  if (r) { toast(`Saved: gain ${r.gain}, trim ${r.trim}`); $("w-drift").value = ""; $("w-dist").value = ""; poll(); }
});
$("btn-s-run").addEventListener("click", () =>
  act(() => api("POST", "/api/calib/wheels/run", { kind: "spin", omega: num("s-omega"), duration: num("s-dur") }), "Spinning in 2 s…"));
$("btn-s-apply").addEventListener("click", async () => {
  const r = await act(() => api("POST", "/api/calib/wheels/apply", { kind: "spin", omega: num("s-omega"), duration: num("s-dur"), turned_deg: num("s-deg") }));
  if (r) { toast(`Saved: omega gain ${r.omega_gain}`); $("s-deg").value = ""; poll(); }
});
const expected = () => { const o = num("s-omega"), d = num("s-dur"); return o && d ? Math.round((o * d * 180) / Math.PI) : null; };

// ------------------------------------------------------------------ status polling
function setPill(id, text, cls) { const p = $(id); p.textContent = text; p.className = "pill" + (cls ? " " + cls : ""); }
async function poll() {
  let s;
  try { s = await api("GET", "/api/status"); } catch (_) {
    setPill("pill-robot", "web UI offline", "bad"); return;
  }
  S.status = s;
  setPill("pill-robot", s.robot.connected ? "robot connected" : "robot_io offline", s.robot.connected ? "ok" : "bad");
  setPill("pill-hz", `${s.loop.hz.toFixed(0)} Hz`);
  const m = s.model;
  setPill("pill-model", m.state === "running" ? `${m.name === "__color__" ? "colour" : m.name} · ${s.loop.infer_ms.toFixed(0)} ms` :
    m.state === "loading" ? `loading ${m.name}…` : m.state === "error" ? "model error" : "no model",
    m.state === "running" ? "live" : m.state === "error" ? "bad" : "");
  setPill("pill-disk", `${s.disk_free_gb} GB free`, s.disk_free_gb >= 0 && s.disk_free_gb < 1 ? "bad" : "");

  const ban = $("banner");
  const msg = !s.robot.connected ? (s.robot.error || "robot_io not reachable") + ". Is robot_io running on the robot?" : s.event;
  ban.hidden = !msg; ban.textContent = msg || "";

  document.querySelectorAll(".seg button").forEach((b) => b.classList.toggle("active", b.dataset.view === s.view));
  const src = { idle: "idle", teleop: "manual", auto: "MODEL DRIVING", calib: "calibration run", "calib-wait": "calibration…" }[s.loop.source] || s.loop.source;
  $("src-badge").textContent = src;
  $("cmd-readout").textContent = `v ${s.loop.cmd[0].toFixed(2)} m/s · ω ${s.loop.cmd[1].toFixed(1)} rad/s`;

  const mo = s.motion, mb = $("motion-badge");
  mb.hidden = !mo;
  if (mo) mb.textContent = mo.starts_in > 0 ? `starting in ${mo.starts_in.toFixed(1)} s` : `running… ${mo.remaining.toFixed(1)} s`;

  // recording
  const rec = s.record;
  $("rec-badge").hidden = !rec.active;
  $("rec-count").textContent = rec.count;
  const br = $("btn-record");
  br.classList.toggle("on", rec.active);
  br.textContent = rec.active ? "■ Stop recording" : "● Start recording";
  $("rec-status").textContent = rec.active ? `Recording ${rec.session}: ${rec.count} frames at ${rec.hz} Hz${rec.dropped ? ` (${rec.dropped} dropped)` : ""}`
    : "Not recording. Frames are saved on the robot (raw + corrected).";
  $("rec-calib-warn").hidden = s.calibration.calibrated.intrinsics;

  // model tab
  $("model-state").textContent = m.state === "running" ? `Running ${m.name === "__color__" ? "colour thresholds" : m.name} on ${m.device} · inference ${s.loop.infer_ms.toFixed(0)} ms · loop ${s.loop.hz.toFixed(1)} Hz`
    : m.state === "loading" ? `Loading ${m.name}… (first start of a model can take a minute)`
    : m.state === "error" ? `Error: ${m.error}` : "No model running.";
  $("autodrive").checked = s.autodrive.enabled;
  $("autodrive").disabled = m.state !== "running" && !s.autodrive.enabled;

  // calibration tab
  const c = s.calibration;
  $("calib-file").textContent = c.file;
  $("calib-chips").innerHTML = [["Lens", c.calibrated.intrinsics], ["Floor", c.calibrated.extrinsics], ["Wheels", c.calibrated.wheels]]
    .map(([n, ok]) => `<span class="chip ${ok ? "ok" : "todo"}">${ok ? "✓" : "○"} ${n}</span>`).join("");
  const cap = s.calib.capture;
  $("intr-progress").max = cap ? cap.target : 30;
  $("intr-progress").value = cap ? cap.count : 0;
  $("intr-state").textContent = cap ? `${cap.count}/${cap.target} views${cap.active ? (cap.found ? " · board found" : " · looking for the board…") : " · capture stopped"}` : "";
  $("btn-intr-compute").disabled = !cap || cap.count < 8;
  const exp = expected();
  $("wheels-state").textContent = `Current: gain ${c.wheels.gain}, trim ${c.wheels.trim}, omega gain ${c.omega_gain}` +
    (exp ? ` · a spin of ${num("s-omega")} rad/s for ${num("s-dur")} s should turn ${exp}°` : "");
  if (S.tab === "model" && S.models.length) {
    const live = document.querySelector("#model-list .tag.live");
    if ((live && m.state === "idle") || (!live && m.state !== "idle")) loadModels();
  }
}
setInterval(poll, 500);
poll().then(() => showTab(location.hash.slice(1) || "drive"));
