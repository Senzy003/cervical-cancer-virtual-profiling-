// Virtual Protein Profiling: demo UI.
// Reads precomputed held-out results from the FastAPI backend:
//   GET /demo-cases              case list
//   GET /demo-cases/<id>         one case (predictions, lab values, image paths)
//   GET /static/model_card.json  metrics for the model card

const MEANINGS = {
  "PD-L1": {
    High: "Predicted high PD-L1: prioritise confirmatory PD-L1 (CPS) testing for immunotherapy eligibility.",
    Low: "Predicted low PD-L1: confirmatory testing is lower priority.",
  },
  "phospho-Rb": {
    High: "Active cell-cycle signalling, consistent with HPV E7 disabling the Rb tumour suppressor.",
    Low: "Less Rb phosphorylation predicted: cell-cycle signalling appears less active.",
  },
  "E-cadherin": {
    High: "Cells predicted to remain cohesive.",
    Low: "Low E-cadherin predicted: loss of cohesion can indicate invasive behaviour.",
  },
  "p16": {
    High: "Pattern consistent with HPV-driven disease.",
    Low: "Lower p16 predicted: may indicate less typical, HPV-independent disease.",
  },
  "Cyclin B1": {
    High: "Higher proliferation predicted.",
    Low: "Lower proliferation predicted.",
  },
};

const state = { cases: [], caseData: null, protein: null, view: "heatmap", zoom: 1, x: 0, y: 0 };
const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

// ------------------------------------------------------------------ cases
async function loadCases() {
  const list = $("case-list");
  try {
    const { cases } = await getJSON("/demo-cases");
    state.cases = cases;
    if (!cases.length) {
      list.innerHTML = '<li class="muted">No demo cases found. Run scripts/make_demo_cases.py.</li>';
      return;
    }
    list.innerHTML = "";
    cases.forEach((c) => {
      const li = document.createElement("li");
      li.innerHTML = `
        <button class="case" data-id="${c.case_id}">
          <img src="${c.thumbnail}" alt="">
          <div><strong>${c.case_id}</strong><span>${c.summary.replace("Held-out case: ", "")}</span></div>
        </button>`;
      li.querySelector("button").addEventListener("click", () => selectCase(c.case_id));
      list.appendChild(li);
    });
    selectCase(cases[0].case_id);
  } catch (err) {
    list.innerHTML = `<li class="muted">Could not reach the API. Is it running?<br><small>${err.message}</small></li>`;
  }
}

async function selectCase(id) {
  document.querySelectorAll(".case").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  const data = await getJSON(`/demo-cases/${encodeURIComponent(id)}`);
  state.caseData = data;
  $("case-title").textContent = data.case_id;
  $("case-summary").textContent = data.summary;
  if (data.disclaimer) $("disclaimer").textContent = data.disclaimer;

  $("viewer-empty").hidden = true;
  $("img-slide").src = data.thumbnail || `/static/${data.case_id}/thumbnail.png`;
  $("img-slide").onload = resetZoom;

  renderCards(data.predictions);
  renderTabs(data.predictions);
  selectProtein(data.predictions[0].protein);
}

// ------------------------------------------------------------------ proteins
function renderTabs(predictions) {
  const tabs = $("protein-tabs");
  tabs.innerHTML = "";
  predictions.forEach((p) => {
    const b = document.createElement("button");
    b.className = p.evidence === "stronger" ? "tab" : "tab tab-explore";
    b.dataset.protein = p.protein;
    b.setAttribute("role", "tab");
    b.innerHTML = `<span class="dot"></span>${p.protein}`;
    b.addEventListener("click", () => selectProtein(p.protein));
    tabs.appendChild(b);
  });
}

function renderCards(predictions) {
  const strong = $("cards-strong");
  const explore = $("cards-exploratory");
  strong.innerHTML = explore.innerHTML = "";
  let exploratoryCount = 0;

  predictions.forEach((p) => {
    const card = document.createElement("article");
    card.className = "card";
    card.dataset.protein = p.protein;
    const high = p.call === "High";
    const pct = Math.round(p.probability_high * 100);

    let lab = '<span class="lab">Lab value: not clear-cut</span>';
    if (p.true_value) {
      const ok = p.true_value === p.call;
      lab = `<span class="lab ${ok ? "match" : "miss"}">${ok ? "✓" : "✗"} Lab value: ${p.true_value}</span>`;
    }
    const meaning = (MEANINGS[p.protein] || {})[p.call] || "";
    const evidence = p.evidence === "stronger"
      ? '<span class="badge badge-strong">Stronger evidence</span>'
      : '<span class="badge badge-explore">Exploratory</span>';

    card.innerHTML = `
      <div class="card-head">
        <h3>${p.protein}</h3>
        <span class="call ${high ? "high" : "low"}">${p.call}</span>
      </div>
      <div class="prob">
        <div class="prob-label"><span>Probability of high expression</span><strong>${pct}%</strong></div>
        <div class="prob-track">
          <div class="prob-fill ${high ? "" : "low"}" style="width:${pct}%"></div>
          <div class="prob-mid" title="Decision threshold (50%)"></div>
        </div>
      </div>
      <p class="meaning">${meaning}</p>
      <div class="card-foot">${lab}<span>${evidence}</span></div>
      ${p.model_auc != null ? `<div class="card-foot" style="margin-top:6px"><span>Model AUC ${p.model_auc.toFixed(2)} on held-out patients</span></div>` : ""}`;
    card.addEventListener("click", () => selectProtein(p.protein));

    if (p.evidence === "stronger") strong.appendChild(card);
    else { explore.appendChild(card); exploratoryCount++; }
  });

  $("exploratory-wrap").hidden = exploratoryCount === 0;
}

function selectProtein(name) {
  state.protein = name;
  const p = state.caseData.predictions.find((x) => x.protein === name);
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.protein === name));
  document.querySelectorAll(".card").forEach((c) => c.classList.toggle("selected", c.dataset.protein === name));
  // Open the exploratory section if an exploratory protein is chosen
  if (p.evidence !== "stronger") $("exploratory-wrap").open = true;

  $("img-heat").src = p.heatmap;
  applyView();

  $("evidence-title").textContent = `What drove the ${name} call`;
  $("patches").innerHTML = (p.top_patches || []).map((src, i) => `
    <figure><img src="${src}" alt="High-attention patch ${i + 1} for ${name}">
    <figcaption>Region ${i + 1}</figcaption></figure>`).join("");
}

// ------------------------------------------------------------------ viewer
function applyView() {
  const heat = $("img-heat");
  heat.style.opacity = state.view === "heatmap" ? $("opacity").value / 100 : 0;
  document.querySelectorAll(".segmented button").forEach((b) =>
    b.classList.toggle("active", b.dataset.view === state.view));
  $("stage").style.transform = `translate(${state.x}px, ${state.y}px) scale(${state.zoom})`;
}

function resetZoom() {
  const viewer = $("viewer").getBoundingClientRect();
  const img = $("img-slide");
  if (!img.naturalWidth) return;
  state.zoom = Math.min(viewer.width / img.naturalWidth, viewer.height / img.naturalHeight);
  state.x = (viewer.width - img.naturalWidth * state.zoom) / 2;
  state.y = (viewer.height - img.naturalHeight * state.zoom) / 2;
  applyView();
}

function zoomAt(factor, cx, cy) {
  const next = Math.min(Math.max(state.zoom * factor, 0.05), 8);
  const k = next / state.zoom;
  state.x = cx - (cx - state.x) * k;
  state.y = cy - (cy - state.y) * k;
  state.zoom = next;
  applyView();
}

function setupViewer() {
  const viewer = $("viewer");
  let drag = null;
  viewer.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX - state.x, y: e.clientY - state.y };
    viewer.classList.add("dragging");
    viewer.setPointerCapture(e.pointerId);
  });
  viewer.addEventListener("pointermove", (e) => {
    if (!drag) return;
    state.x = e.clientX - drag.x;
    state.y = e.clientY - drag.y;
    applyView();
  });
  const end = () => { drag = null; viewer.classList.remove("dragging"); };
  viewer.addEventListener("pointerup", end);
  viewer.addEventListener("pointercancel", end);
  viewer.addEventListener("wheel", (e) => {
    e.preventDefault();
    const r = viewer.getBoundingClientRect();
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });

  const centre = () => { const r = viewer.getBoundingClientRect(); return [r.width / 2, r.height / 2]; };
  $("zoom-in").addEventListener("click", () => zoomAt(1.4, ...centre()));
  $("zoom-out").addEventListener("click", () => zoomAt(1 / 1.4, ...centre()));
  $("zoom-reset").addEventListener("click", resetZoom);
  window.addEventListener("resize", resetZoom);

  document.querySelectorAll(".segmented button").forEach((b) =>
    b.addEventListener("click", () => { state.view = b.dataset.view; applyView(); }));
  $("opacity").addEventListener("input", () => { state.view = "heatmap"; applyView(); });
}

// ------------------------------------------------------------------ model card
async function openModelCard() {
  const dialog = $("model-card");
  dialog.showModal();
  const body = $("model-card-body");
  try {
    const card = await getJSON("/static/model_card.json");
    const rows = {};
    card.metrics.forEach((m) => { (rows[m.protein] ||= {})[m.model] = m; });
    const fmt = (m) => m && m.auc != null
      ? `${m.auc.toFixed(2)} <span class="muted">(${m.ci_low.toFixed(2)}–${m.ci_high.toFixed(2)})</span>` : "–";
    const splitText = card.split === "tertile"
      ? "top third vs bottom third of patients for each protein"
      : "above vs below the cohort median for each protein";

    body.innerHTML = `
      <div class="mc-section">
        <p><strong>What it does:</strong> predicts whether five cancer-related proteins are high or low
        from a routine H&amp;E slide, and shows which tissue regions drove each prediction.</p>
        <h3>Held-out performance (AUC, 95% CI)</h3>
        <table>
          <thead><tr><th>Protein</th><th>Attention model</th><th>Baseline</th></tr></thead>
          <tbody>${Object.entries(rows)
            .sort((a, b) => (b[1].abmil?.auc || 0) - (a[1].abmil?.auc || 0))
            .map(([p, m]) => `<tr><td>${p}</td><td>${fmt(m.abmil)}</td><td>${fmt(m.baseline)}</td></tr>`)
            .join("")}</tbody>
        </table>
        <p class="muted">AUC 0.5 = chance, 1.0 = perfect. ${card.folds}-fold cross-validation by patient.</p>
        <h3>Data</h3>
        <p>${card.n_patients || 145} TCGA cervical cancer patients with both a diagnostic H&amp;E slide and
        reverse-phase protein array (RPPA) data. Labels: ${splitText}.
        Patch features from the ${card.encoder} pathology foundation model.</p>
        <h3>Limitations</h3>
        <ul>
          <li>Trained on TCGA slides, mostly from US hospitals; not yet tested on South African slides.</li>
          <li>RPPA measures total protein in tissue, not clinical IHC scores such as PD-L1 CPS.</li>
          <li>Results are relative to this cohort, not clinical cut-offs.</li>
          <li>Decision support only: flags patients for confirmatory testing, never replaces it.</li>
        </ul>
      </div>`;
  } catch (err) {
    body.innerHTML = `<p class="muted">Model card not found (${err.message}).</p>`;
  }
}

$("model-card-btn").addEventListener("click", openModelCard);
$("model-card-close").addEventListener("click", () => $("model-card").close());
setupViewer();
loadCases();
