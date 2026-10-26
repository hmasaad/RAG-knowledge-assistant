const userField = document.getElementById("user-id");
const healthEl = document.getElementById("health");
const documentsEl = document.getElementById("documents");
const sessionsEl = document.getElementById("sessions");
const askResult = document.getElementById("ask-result");
const researchResult = document.getElementById("research-result");
const uploadStatus = document.getElementById("upload-status");
const researchStatus = document.getElementById("research-status");

const storedUser = localStorage.getItem("rag-user-id");
if (storedUser) userField.value = storedUser;

userField.addEventListener("change", () => {
  localStorage.setItem("rag-user-id", userField.value.trim() || "local");
  refreshLibrary();
  refreshSessions();
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => showTab(tab.dataset.tab));
});

document.getElementById("refresh-docs").addEventListener("click", refreshLibrary);
document.getElementById("refresh-sessions").addEventListener("click", refreshSessions);
document.getElementById("upload-form").addEventListener("submit", uploadDocument);
document.getElementById("ask-form").addEventListener("submit", askQuestion);
document.getElementById("research-form").addEventListener("submit", runResearch);

function showTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    const active = panel.id === `panel-${name}`;
    panel.classList.toggle("is-active", active);
    panel.hidden = !active;
  });
}

function headers(extra = {}) {
  const user = userField.value.trim() || "local";
  return { "X-User-Id": user, ...extra };
}

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: headers(options.headers || {}),
  });
  if (response.status === 204) return null;
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!response.ok) {
    throw new Error(body?.detail || `Request failed (${response.status})`);
  }
  return body;
}

async function refreshHealth() {
  try {
    const body = await request("/health");
    healthEl.textContent = `${body.documents} documents · ${body.chunks} chunks · ${body.indexing_mode}`;
    healthEl.className = "health is-ok";
  } catch (error) {
    healthEl.textContent = error.message;
    healthEl.className = "health is-bad";
  }
}

async function refreshLibrary() {
  const body = await request("/api/documents");
  documentsEl.replaceChildren();
  if (!body.documents.length) {
    documentsEl.append(item("No documents yet."));
    return;
  }
  for (const doc of body.documents) {
    const row = document.createElement("li");
    const info = document.createElement("div");
    info.innerHTML = `<strong>${escapeHtml(doc.filename)}</strong><div class="meta">${escapeHtml(doc.status)} · v${doc.version} · ${escapeHtml(doc.visibility)}</div>`;
    const remove = document.createElement("button");
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
      await request(`/api/documents/${doc.id}`, { method: "DELETE" });
      refreshLibrary();
      refreshHealth();
    });
    row.append(info, remove);
    documentsEl.append(row);
  }
}

async function uploadDocument(event) {
  event.preventDefault();
  const file = document.getElementById("file").files[0];
  if (!file) return;
  const data = new FormData();
  data.append("file", file);
  data.append("visibility", document.getElementById("visibility").value);
  uploadStatus.textContent = "Indexing…";
  try {
    const created = await fetch("/api/documents", {
      method: "POST",
      headers: headers(),
      body: data,
    });
    const body = await created.json();
    if (!created.ok) throw new Error(body.detail || "Upload failed.");
    uploadStatus.textContent = body.already_indexed
      ? `${body.filename} was already indexed.`
      : `${body.filename} is ${body.status}.`;
    event.target.reset();
    refreshLibrary();
    refreshHealth();
  } catch (error) {
    uploadStatus.textContent = error.message;
  }
}

async function askQuestion(event) {
  event.preventDefault();
  askResult.hidden = false;
  askResult.textContent = "Searching…";
  try {
    const body = await request("/api/answer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        question: document.getElementById("ask-question").value,
        top_k: Number(document.getElementById("ask-top-k").value) || 4,
      }),
    });
    renderAnswer(askResult, body);
  } catch (error) {
    askResult.textContent = error.message;
  }
}

async function runResearch(event) {
  event.preventDefault();
  researchStatus.textContent = "Working…";
  researchResult.hidden = false;
  researchResult.textContent = "Investigating…";
  const mode = document.getElementById("research-mode").value;
  const path = mode === "plan" ? "/api/research" : "/api/research/cycle";
  try {
    const body = await request(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        question: document.getElementById("research-question").value,
        top_k: Number(document.getElementById("research-top-k").value) || 4,
        halt_after: document.getElementById("halt-after").checked ? 1 : null,
      }),
    });
    await showResearch(body);
    refreshSessions();
  } catch (error) {
    researchStatus.textContent = error.message;
    researchResult.textContent = error.message;
  }
}

async function showResearch(body) {
  if (body.status === "queued" || body.status === "running") {
    researchStatus.textContent = `Session ${body.session_id} is ${body.status}. Waiting for the worker…`;
    renderResearch(researchResult, body);
    await waitForSession(body.session_id);
    return;
  }
  researchStatus.textContent = body.session_id ? `Session ${body.session_id}` : body.action;
  renderResearch(researchResult, body);
}

async function waitForSession(sessionId) {
  for (let attempt = 0; attempt < 40; attempt += 1) {
    await sleep(1500);
    const body = await request(`/api/research/sessions/${sessionId}`);
    renderResearch(researchResult, body);
    if (body.status === "completed" || body.status === "failed") {
      researchStatus.textContent = `Session ${sessionId} is ${body.status}.`;
      return;
    }
  }
  researchStatus.textContent = "Still running. Refresh sessions to check again.";
}

async function refreshSessions() {
  const body = await request("/api/research/sessions");
  sessionsEl.replaceChildren();
  if (!body.sessions.length) {
    sessionsEl.append(item("No research sessions yet."));
    return;
  }
  for (const session of body.sessions) {
    const row = document.createElement("li");
    const info = document.createElement("div");
    info.innerHTML = `<strong>${escapeHtml(session.question)}</strong><div class="meta">${escapeHtml(session.status)} · ${escapeHtml(session.session_id.slice(0, 8))}</div>`;
    const actions = document.createElement("div");
    const open = document.createElement("button");
    open.className = "ghost";
    open.textContent = "Open";
    open.addEventListener("click", async () => {
      const detail = await request(`/api/research/sessions/${session.session_id}`);
      showTab("research");
      await showResearch(detail);
    });
    const resume = document.createElement("button");
    resume.textContent = "Resume";
    resume.addEventListener("click", async () => {
      const detail = await request(`/api/research/sessions/${session.session_id}/resume`, {
        method: "POST",
      });
      showTab("research");
      await showResearch(detail);
      refreshSessions();
    });
    actions.append(open, resume);
    row.append(info, actions);
    sessionsEl.append(row);
  }
}

function renderAnswer(target, body) {
  target.innerHTML = `
    <h2>Answer</h2>
    <p class="meta">${escapeHtml(body.action)} · ${body.sufficient_evidence ? "enough evidence" : "not enough evidence"}${body.conflict ? " · conflict" : ""}</p>
    <p class="report">${escapeHtml(body.answer)}</p>
    ${renderCitations(body.citations)}
  `;
}

function renderResearch(target, body) {
  const steps = (body.decisions || []).map((step) => `<div class="step">${escapeHtml(step)}</div>`).join("");
  const gaps = (body.gaps || []).map((gap) => `<div class="gap">${escapeHtml(gap)}</div>`).join("");
  target.innerHTML = `
    <h2>Research report</h2>
    <p class="meta">${escapeHtml(body.status || "completed")} · ${escapeHtml(body.action || "")} · ${body.sufficient_evidence ? "enough evidence" : "open gaps"}</p>
    <p class="report">${escapeHtml(body.report || "No report yet.")}</p>
    <h3>Steps</h3>
    ${steps || "<p class='hint'>No steps yet.</p>"}
    <h3>Gaps</h3>
    ${gaps || "<p class='hint'>None.</p>"}
    ${renderCitations(body.citations)}
    ${renderEffects(body.side_effects)}
  `;
}

function renderEffects(effects) {
  if (!effects || !effects.length) return "";
  const items = effects
    .map((item) => `<div class="step">${escapeHtml(item.action)} ran ${item.runs} time${item.runs === 1 ? "" : "s"}</div>`)
    .join("");
  return `<h3>Adapter runs</h3>${items}`;
}

function renderCitations(citations) {
  if (!citations || !citations.length) return "";
  const items = citations
    .map(
      (item) =>
        `<div class="citation"><strong>${escapeHtml(item.marker)}</strong> ${escapeHtml(item.filename)}${item.heading ? " · " + escapeHtml(item.heading) : ""}<div>${escapeHtml(item.quote)}</div></div>`,
    )
    .join("");
  return `<h3>Citations</h3>${items}`;
}

function item(text) {
  const row = document.createElement("li");
  row.textContent = text;
  return row;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

refreshHealth();
refreshLibrary().catch((error) => {
  documentsEl.replaceChildren(item(error.message));
});
refreshSessions().catch((error) => {
  sessionsEl.replaceChildren(item(error.message));
});
