const SVG_NS = "http://www.w3.org/2000/svg";

const messagesEl = document.getElementById("messages");
const inspectorBody = document.getElementById("inspector-body");
const inspectorTitle = document.getElementById("inspector-title");
const graph = document.getElementById("graph");
const assetBar = document.getElementById("asset-bar");
const filterMenu = document.getElementById("filter-menu");
const askInput = document.getElementById("ask-input");

let scene = null;
let messages = [];
let seeded = false;
let selectedClaim = "";
let selectedAgent = "";
let selectedChunk = "";
let expanded = false;
let query = "";
let showChunks = true;
let showContradictions = true;
let showEntities = false;
let busy = false;

document.getElementById("trace-ask-form").addEventListener("submit", ask);
document.getElementById("clear-chat").addEventListener("click", clearChat);
document.getElementById("clear-selection").addEventListener("click", clearSelection);
document.getElementById("close-inspector").addEventListener("click", () => {
  document.getElementById("trace").classList.toggle("is-inspector-closed");
});
document.getElementById("download-btn").addEventListener("click", downloadScene);
document.getElementById("expand-btn").addEventListener("click", () => {
  const shell = document.getElementById("trace");
  if (document.fullscreenElement) document.exitFullscreen();
  else shell.requestFullscreen?.();
});
document.getElementById("filter-btn").addEventListener("click", () => {
  filterMenu.hidden = !filterMenu.hidden;
  document.getElementById("filter-btn").setAttribute("aria-expanded", String(!filterMenu.hidden));
});
document.getElementById("node-search").addEventListener("input", (event) => {
  query = event.target.value.trim().toLowerCase();
  renderGraph();
});
document.getElementById("show-chunks").addEventListener("change", (event) => {
  showChunks = event.target.checked;
  renderGraph();
});
document.getElementById("show-contradictions").addEventListener("change", (event) => {
  showContradictions = event.target.checked;
  renderGraph();
});
document.getElementById("show-entities").addEventListener("change", (event) => {
  showEntities = event.target.checked;
  renderGraph();
});
document.addEventListener("click", (event) => {
  if (!event.target.closest("#filter-menu") && !event.target.closest("#filter-btn")) {
    filterMenu.hidden = true;
  }
});

async function load(asset, options = {}) {
  scene = await api(`/api/trace?asset=${encodeURIComponent(asset)}`);
  selectedAgent = scene.selected_agent;
  selectedClaim = scene.selected_claim;
  selectedChunk = "";
  expanded = false;
  if (!seeded && options.seed !== false) {
    messages = [
      { role: "user", text: scene.question, markers: [], asset: scene.asset },
      { role: "assistant", text: scene.answer, markers: scene.markers, asset: scene.asset },
    ];
    seeded = true;
  }
  render();
}

function render() {
  renderMessages();
  renderAssets();
  renderGraph();
  renderInspector();
}

function renderMessages() {
  messagesEl.replaceChildren();
  for (const message of messages) {
    const bubble = document.createElement("div");
    bubble.className = message.role === "user" ? "msg-user" : "msg-assistant";
    if (message.role === "user") {
      bubble.textContent = message.text;
    } else {
      appendAnswer(bubble, message);
    }
    messagesEl.append(bubble);
  }
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function appendAnswer(bubble, message) {
  const parts = String(message.text || "").split(/\[\[(\d+)\]\]/);
  const byNumber = new Map((message.markers || []).map((item) => [item.n, item.claim_id]));
  parts.forEach((part, index) => {
    if (index % 2 === 0) {
      bubble.append(document.createTextNode(part));
      return;
    }
    const n = Number(part);
    const button = document.createElement("button");
    button.type = "button";
    button.className = "cite";
    button.textContent = String(n);
    button.addEventListener("click", () => focusCitation(message.asset, byNumber.get(n)));
    bubble.append(button);
  });
}

async function focusCitation(asset, claimId) {
  if (!claimId) return;
  if (scene.asset !== asset) await load(asset, { seed: false });
  const claim = scene.claims.find((item) => item.id === claimId);
  if (!claim) return;
  selectedClaim = claim.id;
  selectedAgent = claim.agent_id;
  selectedChunk = "";
  expanded = true;
  renderGraph();
  renderInspector();
}

function renderAssets() {
  assetBar.replaceChildren();
  for (const asset of scene.assets) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "asset";
    button.style.borderColor = asset.id === scene.asset ? "#4c8dff" : asset.color;
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = asset.color;
    const ticker = document.createElement("strong");
    ticker.textContent = asset.id;
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = asset.name;
    button.append(dot, ticker, name);
    button.addEventListener("click", () => load(asset.id, { seed: false }));
    assetBar.append(button);
  }
}

function renderGraph() {
  const positions = layout(scene);
  const claimById = new Map(scene.claims.map((claim) => [claim.id, claim]));
  const active = claimById.get(selectedClaim);
  const upstream = new Set(active ? active.upstream : []);
  graph.replaceChildren();
  drawEdges(positions, claimById, upstream);
  for (const [id, box] of Object.entries(positions)) {
    drawNode(id, box, claimById, upstream);
  }
}

function layout(current) {
  const positions = {};
  const byRole = Object.fromEntries(current.agents.map((agent) => [agent.role, agent]));
  const paired = byRole.news && byRole.social;
  if (byRole.aggregator) {
    positions[byRole.aggregator.id] = box(paired ? 468 : 460, 575, paired ? 168 : 180, 52, "agent");
  }
  if (paired) {
    positions[byRole.news.id] = box(248, 268, 156, 52, "agent");
    positions[byRole.social.id] = box(700, 248, 156, 52, "agent");
    placeBand(claimsFor(current, byRole.news.id, "above"), 78, 42, positions);
    placeBand(claimsFor(current, byRole.news.id, "below"), 78, 360, positions);
    placeBand(claimsFor(current, byRole.social.id, "above"), 630, 36, positions);
    placeBand(claimsFor(current, byRole.social.id, "below"), 630, 348, positions);
    if (showChunks) {
      placeChunks(current.chunks.filter((chunk) => chunk.agent_id === byRole.news.id), 96, 470, positions);
      placeChunks(current.chunks.filter((chunk) => chunk.agent_id === byRole.social.id), 748, 455, positions);
    }
  } else {
    const source = byRole.news || byRole.social;
    if (source) {
      positions[source.id] = box(460, 250, 180, 52, "agent");
      placeBand(claimsFor(current, source.id, "above"), 400, 48, positions);
      placeBand(claimsFor(current, source.id, "below"), 400, 360, positions);
      if (showChunks) {
        placeChunks(current.chunks.filter((chunk) => chunk.agent_id === source.id), 470, 450, positions);
      }
    }
  }
  if (showEntities) {
    current.entities.forEach((entity, index) => {
      positions[entity.id] = box(430 + (index % 2) * 130, 150 + Math.floor(index / 2) * 36, 120, 26, "entity");
    });
  }
  return positions;
}

function claimsFor(current, agentId, band) {
  return current.claims.filter((claim) => claim.agent_id === agentId && claim.band === band);
}

function placeBand(claims, x, y, positions) {
  const width = 96;
  const height = 30;
  const gap = 8;
  claims.forEach((claim, index) => {
    const column = index % 3;
    const row = Math.floor(index / 3);
    positions[claim.id] = box(x + column * (width + gap), y + row * (height + gap), width, height, "claim");
  });
}

function placeChunks(chunks, x, y, positions) {
  chunks.slice(0, 4).forEach((chunk, index) => {
    const column = index % 2;
    const row = Math.floor(index / 2);
    positions[chunk.id] = box(x + column * 64, y + row * 40, 56, 32, "chunk");
  });
}

function box(x, y, w, h, kind) {
  return { x, y, w, h, kind };
}

function drawEdges(positions, claimById, upstream) {
  const drawn = new Set();
  for (const edge of scene.edges) {
    if (edge.relation === "contradicts" && !showContradictions) continue;
    if (edge.relation === "mentions" && !showEntities) continue;
    if (!["member_of", "feeds", "supports", "contradicts", "mentions"].includes(edge.relation)) continue;
    const from = positions[edge.source];
    const to = resolveEnd(edge.target, positions, claimById);
    if (!from || !to) continue;
    const hot =
      (edge.relation === "supports" && edge.target === selectedClaim) ||
      (edge.relation === "member_of" && upstream.has(edge.source));
    const x1 = from.x + from.w / 2;
    const y1 = from.y + from.h / 2;
    const x2 = to.x + to.w / 2;
    const y2 = to.y + to.h / 2;
    const kind = edge.relation === "contradicts" ? "edge is-contra" : hot ? "edge is-hot" : "edge";
    const key = `${kind}:${x1}:${y1}:${x2}:${y2}`;
    if (drawn.has(key)) continue;
    drawn.add(key);
    graph.append(svg("line", { x1, y1, x2, y2, class: kind }));
  }
}

function resolveEnd(id, positions, claimById) {
  if (positions[id]) return positions[id];
  const claim = claimById.get(id);
  if (claim && positions[claim.agent_id]) return positions[claim.agent_id];
  return null;
}

function drawNode(id, item, claimById, upstream) {
  const claim = claimById.get(id);
  const entity = scene.entities.find((row) => row.id === id);
  const chunk = scene.chunks.find((row) => row.id === id);
  const label = claim ? claim.short : entity ? entity.label : chunk ? "" : agentLabel(id);
  const dim = query && !`${label} ${claim ? claim.text : ""} ${chunk ? chunk.filename : ""}`.toLowerCase().includes(query);
  const group = svg("g", { class: nodeClass(id, item, upstream, dim), tabindex: "0", role: "button" });
  group.append(svg("rect", { x: item.x, y: item.y, width: item.w, height: item.h, rx: item.kind === "entity" ? 12 : 6 }));
  if (label) {
    const text = svg("text", {
      x: item.x + item.w / 2,
      y: item.y + item.h / 2 + 4,
      "text-anchor": "middle",
      fill: item.kind === "claim" || item.kind === "agent" ? "#1c1404" : "#dbe4f0",
      "font-size": item.kind === "agent" ? "14" : "11",
    });
    text.textContent = clip(label, item.kind === "agent" ? 16 : 14);
    group.append(text);
  }
  const title = svg("title");
  title.textContent = claim ? claim.text : chunk ? `${chunk.filename} v${chunk.version}` : label;
  group.append(title);
  group.addEventListener("click", () => selectNode(id, item.kind));
  group.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectNode(id, item.kind);
    }
  });
  graph.append(group);
}

function nodeClass(id, item, upstream, dim) {
  const classes = ["node", `is-${item.kind}`];
  if (id === selectedClaim || id === selectedAgent || id === selectedChunk) classes.push("is-selected");
  if (upstream.has(id)) classes.push("is-upstream");
  if (dim && item.kind !== "agent") classes.push("is-dim");
  if (isLoser(id) && id !== selectedClaim) classes.push("is-loser");
  return classes.join(" ");
}

function isLoser(id) {
  return (scene.conflicts || []).some((item) => item.loser_id === id && item.status !== "pending");
}

function agentLabel(id) {
  return scene.agents.find((agent) => agent.id === id)?.label || "";
}

function selectNode(id, kind) {
  if (kind === "claim") {
    const claim = scene.claims.find((item) => item.id === id);
    selectedClaim = id;
    selectedAgent = claim.agent_id;
    selectedChunk = "";
    expanded = true;
  } else if (kind === "agent") {
    selectedAgent = id;
    selectedClaim = "";
    selectedChunk = "";
    expanded = false;
  } else if (kind === "chunk") {
    selectedChunk = id;
    selectedClaim = "";
    const chunk = scene.chunks.find((item) => item.id === id);
    selectedAgent = chunk ? chunk.agent_id : selectedAgent;
    expanded = false;
  } else if (kind === "entity") {
    selectedChunk = "";
    selectedClaim = "";
  }
  renderGraph();
  renderInspector();
}

function renderInspector() {
  const agent = scene.agents.find((item) => item.id === selectedAgent) || scene.agents.find((item) => item.role === "aggregator");
  inspectorTitle.textContent = agent ? agent.label : "Aggregator";
  inspectorBody.replaceChildren();
  if (scene.decisions && scene.decisions.length) {
    const latest = scene.decisions[scene.decisions.length - 1];
    const decision = document.createElement("p");
    decision.className = "origin";
    decision.textContent = `Decision: ${latest.decision} · ${latest.recorded_at} · ${latest.detail}`;
    inspectorBody.append(decision);
  }
  renderResearchPlan();
  renderConflictQueue();
  if (selectedChunk) {
    renderChunk(scene.chunks.find((item) => item.id === selectedChunk));
    return;
  }
  const claims = scene.claims.filter((claim) => claim.agent_id === agent.id);
  for (const claim of claims) {
    inspectorBody.append(claimCard(claim));
  }
}

function claimCard(claim) {
  const card = document.createElement("article");
  card.className = "claim-card";
  card.tabIndex = 0;
  if (claim.id === selectedClaim) card.classList.add("is-selected");
  const text = document.createElement("p");
  text.textContent = claim.text;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = claim.upstream_count
    ? `Connected to: ${claim.upstream_count} upstream claims`
    : `${claim.provenance.filename} · v${claim.provenance.version}`;
  card.append(text, meta);
  if (expanded && claim.id === selectedClaim) card.append(provenanceBlock(claim));
  const choose = () => {
    if (selectedClaim === claim.id) expanded = !expanded;
    else expanded = true;
    selectedClaim = claim.id;
    selectedAgent = claim.agent_id;
    selectedChunk = "";
    renderGraph();
    renderInspector();
  };
  card.addEventListener("click", choose);
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      choose();
    }
  });
  return card;
}

function provenanceBlock(claim) {
  const wrap = document.createElement("div");
  const origin = document.createElement("p");
  origin.className = "origin";
  const checksum = claim.provenance.checksum.slice(0, 8);
  const recorded = claim.provenance.recorded_at ? ` · ${claim.provenance.recorded_at}` : "";
  origin.textContent = `Origin: ${claim.provenance.filename} v${claim.provenance.version} · ${claim.provenance.chunk_id} · ${checksum} · chars ${claim.provenance.start_char}–${claim.provenance.end_char}${recorded}`;
  wrap.append(origin);
  if (claim.provenance.lineage && claim.provenance.lineage.length) {
    const lineage = document.createElement("p");
    lineage.className = "history";
    lineage.textContent = `Lineage: ${claim.provenance.lineage.map(lineageLabel).join(" → ")}`;
    wrap.append(lineage);
  }
  if (claim.citation) {
    const cite = document.createElement("p");
    cite.className = "origin";
    const marker = claim.citation.marker ? `[[${claim.citation.marker}]] ` : "";
    cite.textContent = `Citation: ${marker}${claim.citation.filename} v${claim.citation.version} · chars ${claim.citation.start_char}–${claim.citation.end_char}`;
    wrap.append(cite);
  }
  if (claim.provenance.history.length > 1) {
    const history = document.createElement("p");
    history.className = "history";
    history.textContent = `Versions: ${claim.provenance.history.map((item) => {
      const day = item.recorded_at ? ` ${item.recorded_at.slice(0, 10)}` : "";
      return `v${item.version}${day}`;
    }).join(" → ")}`;
    wrap.append(history);
  }
  if (claim.audit && claim.audit.length) {
    const audit = document.createElement("ul");
    audit.className = "audit";
    for (const event of claim.audit) {
      const item = document.createElement("li");
      item.textContent = `${event.decision} · ${event.recorded_at} · ${event.detail}`;
      audit.append(item);
    }
    wrap.append(audit);
  }
  if (claim.upstream.length) {
    const list = document.createElement("ul");
    list.className = "upstream";
    for (const id of claim.upstream) {
      const source = scene.claims.find((item) => item.id === id);
      if (!source) continue;
      const item = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.textContent = `${source.short} · ${source.provenance.filename} v${source.provenance.version}`;
      button.addEventListener("click", (event) => {
        event.stopPropagation();
        selectedClaim = source.id;
        selectedAgent = source.agent_id;
        expanded = true;
        renderGraph();
        renderInspector();
      });
      item.append(button);
      list.append(item);
    }
    wrap.append(list);
  }
  if (claim.contradicts.length) {
    const note = document.createElement("p");
    note.className = "contra";
    const names = claim.contradicts
      .map((id) => scene.claims.find((item) => item.id === id)?.short)
      .filter(Boolean);
    note.textContent = `Contradicts: ${names.join(", ")}`;
    wrap.append(note);
  }
  for (const conflict of conflictsFor(claim.id)) {
    wrap.append(conflictNote(conflict, claim.id));
  }
  if (claim.entities.length) {
    const names = document.createElement("p");
    names.className = "meta";
    names.textContent = `Entities: ${claim.entities.join(", ")}`;
    wrap.append(names);
  }
  return wrap;
}

function conflictsFor(claimId) {
  return (scene.conflicts || []).filter((item) => item.left_id === claimId || item.right_id === claimId);
}

function renderResearchPlan() {
  const plan = scene.plan;
  if (!plan) return;
  const box = document.createElement("article");
  box.className = "claim-card plan";
  const title = document.createElement("p");
  title.textContent = plan.items && plan.items.length ? "Research plan" : "Evidence is sufficient";
  box.append(title);
  if (plan.summary) {
    const summary = document.createElement("p");
    summary.className = "origin";
    summary.textContent = plan.summary;
    box.append(summary);
  }
  for (const item of plan.items || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "follow-up";
    button.textContent = `${item.priority}. ${item.question}`;
    button.addEventListener("click", () => askFollowUp(item.question));
    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = `${item.strategy} · value ${Number(item.value).toFixed(2)} · ${item.missing}`;
    box.append(button, meta);
  }
  inspectorBody.append(box);
}

function askFollowUp(question) {
  if (busy) return;
  askInput.value = question;
  ask({ preventDefault() {} });
}

function renderConflictQueue() {
  const pending = (scene.conflicts || []).filter((item) => item.status === "pending");
  if (!pending.length) return;
  const box = document.createElement("article");
  box.className = "claim-card";
  const title = document.createElement("p");
  title.textContent = pending.length === 1 ? "1 conflict needs approval" : `${pending.length} conflicts need approval`;
  box.append(title);
  for (const conflict of pending) box.append(conflictNote(conflict));
  inspectorBody.append(box);
}

function conflictNote(conflict, currentId) {
  const wrap = document.createElement("div");
  const note = document.createElement("p");
  note.className = conflict.status === "pending" ? "contra" : "origin";
  const left = scene.claims.find((item) => item.id === conflict.left_id);
  const right = scene.claims.find((item) => item.id === conflict.right_id);
  const pair = `${left ? left.short : "Claim"} vs ${right ? right.short : "Claim"}`;
  note.textContent = `${labelStatus(conflict.status)}: ${pair}. ${conflict.reason}`;
  wrap.append(note);
  if (conflict.status === "pending" || conflict.status === "approved") {
    const actions = document.createElement("div");
    actions.className = "approval";
    actions.append(approveButton(conflict, conflict.left_id, left ? left.short : "Left", currentId));
    actions.append(approveButton(conflict, conflict.right_id, right ? right.short : "Right", currentId));
    wrap.append(actions);
  }
  return wrap;
}

function labelStatus(status) {
  if (status === "superseded") return "Superseded";
  if (status === "authority") return "Authority";
  if (status === "approved") return "Approved";
  return "Unresolved";
}

function approveButton(conflict, winnerId, name, currentId) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = conflict.winner_id === winnerId ? `Approved ${name}` : `Approve ${name}`;
  button.disabled = conflict.winner_id === winnerId;
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    approveConflict(conflict, winnerId, currentId);
  });
  return button;
}

async function approveConflict(conflict, winnerId, currentId) {
  if (busy) return;
  busy = true;
  try {
    scene = await api("/api/trace/conflicts/approve", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        asset: scene.asset,
        conflict_id: conflict.id,
        winner_id: winnerId,
      }),
    });
    if (currentId) {
      selectedClaim = currentId;
      const claim = scene.claims.find((item) => item.id === currentId);
      if (claim) selectedAgent = claim.agent_id;
      expanded = true;
    }
    render();
  } catch (error) {
    messages.push({ role: "assistant", text: error.message, markers: [], asset: scene.asset });
    renderMessages();
  } finally {
    busy = false;
  }
}

function lineageLabel(step) {
  if (step.kind === "version") {
    const day = (step.recorded_at || "").slice(0, 10);
    return day ? `v${step.version} ${day}` : `v${step.version}`;
  }
  if (step.kind === "claim") return "claim";
  if (step.kind === "passage") return "passage";
  if (step.kind === "source") return `source ${step.label}`;
  if (step.kind === "document") return `document ${step.label}`;
  return step.label || step.kind;
}

function renderChunk(chunk) {
  if (!chunk) return;
  inspectorTitle.textContent = chunk.filename;
  const card = document.createElement("article");
  card.className = "claim-card";
  const text = document.createElement("p");
  text.textContent = chunk.label;
  const origin = document.createElement("p");
  origin.className = "origin";
  origin.textContent = `Chunk ${chunk.chunk_id} · v${chunk.version} · ${chunk.checksum.slice(0, 8)}`;
  card.append(text, origin);
  inspectorBody.append(card);
}

async function ask(event) {
  event.preventDefault();
  const question = askInput.value.trim();
  if (!question || busy) return;
  busy = true;
  askInput.value = "";
  messages.push({ role: "user", text: question, markers: [], asset: scene.asset });
  renderMessages();
  try {
    const body = await api("/api/trace/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, asset: scene.asset }),
    });
    scene = body.scene;
    selectedAgent = scene.selected_agent;
    selectedClaim = scene.selected_claim;
    selectedChunk = "";
    expanded = body.markers.length > 0;
    messages.push({ role: "assistant", text: body.answer, markers: body.markers, asset: body.asset });
  } catch (error) {
    messages.push({ role: "assistant", text: error.message, markers: [], asset: scene.asset });
  } finally {
    busy = false;
    render();
  }
}

function clearChat() {
  messages = [];
  seeded = true;
  renderMessages();
}

function clearSelection() {
  selectedClaim = "";
  selectedChunk = "";
  expanded = false;
  renderGraph();
  renderInspector();
}

function downloadScene() {
  const blob = new Blob([JSON.stringify(scene, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `source-trace-${scene.asset}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function clip(value, length) {
  const text = String(value || "");
  return text.length > length ? `${text.slice(0, length - 1)}…` : text;
}

function svg(name, attrs = {}) {
  const element = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) element.setAttribute(key, value);
  return element;
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      "X-User-Id": localStorage.getItem("rag-user-id") || "local",
      ...(options.headers || {}),
    },
  });
  const text = await response.text();
  let body = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = { detail: text };
  }
  if (!response.ok) {
    const detail = body?.detail;
    throw new Error(typeof detail === "string" && detail ? detail : `Request failed (${response.status})`);
  }
  return body;
}

load("NVDA").catch((error) => {
  messagesEl.textContent = error.message;
});

window.showTraceScene = function showTraceScene(next, question) {
  if (!next) return;
  scene = next;
  selectedAgent = scene.selected_agent || "";
  selectedClaim = scene.selected_claim || "";
  selectedChunk = "";
  expanded = false;
  query = "";
  messages = [];
  if (question) messages.push({ role: "user", text: question, markers: [], asset: scene.asset });
  messages.push({ role: "assistant", text: scene.answer || "", markers: scene.markers || [], asset: scene.asset });
  seeded = true;
  render();
  document.querySelector('.tab[data-tab="trace"]')?.click();
};
