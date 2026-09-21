/* Invoice pipeline console. Vanilla — no build step, no framework. */

const $ = (sel) => document.querySelector(sel);
const api = async (path, opts) => {
  const res = await fetch(path, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
};

/* Escape everything interpolated into innerHTML. Invoice text is adversarial by
   premise — one of the sample documents is literally from "Fraudster LLC". */
const esc = (v) =>
  String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );

const money = (v, ccy) =>
  v == null ? "—" : `${Number(v).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}${ccy ? " " + ccy : ""}`;

const DECISION = {
  approved:     { cls: "ok",   mark: "✓", word: "APPROVED" },
  rejected:     { cls: "bad",  mark: "✗", word: "REJECTED" },
  needs_review: { cls: "warn", mark: "!", word: "NEEDS REVIEW" },
  error:        { cls: "bad",  mark: "✗", word: "ERROR" },
};

const SEVERITY = { critical: "✗", warning: "!", info: "·" };

let inbox = [];
let selected = null;

/* ───────────────────────────────── tabs ───────────────────────────────── */

const LOADERS = {
  inbox: loadInbox,
  inventory: loadInventory,
  vendors: loadVendors,
  ledger: loadLedger,
};
const loaded = new Set();

$("#tabs").addEventListener("click", (e) => {
  const tab = e.target.closest(".tab");
  if (!tab) return;
  const name = tab.dataset.tab;

  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  document.querySelectorAll(".panel").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${name}`)
  );

  if (!loaded.has(name)) {
    loaded.add(name);
    LOADERS[name]();
  }
});

/* Number keys jump between tabs. */
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  const tabs = [...document.querySelectorAll(".tab")];
  const i = Number(e.key) - 1;
  if (i >= 0 && i < tabs.length) tabs[i].click();
});

/* ─────────────────────────────── overview ─────────────────────────────── */

async function loadOverview() {
  try {
    const o = await api("/api/overview");
    const live = o.llm_mode === "grok";
    const failed = o.decisions?.error ?? 0;
    $("#pills").innerHTML = `
      <span class="pill"><span class="dot ${live ? "live" : "replay"}"></span>
        ${live ? "GROK" : "OFFLINE"} <b>${esc(o.model)}</b></span>
      <span class="pill"><span class="dot ${o.db_ready ? "live" : "warn"}"></span>
        DB <b>${esc(o.db)}</b></span>
      <span class="pill">DOCS <b>${o.documents}</b></span>
      <span class="pill">SKUS <b>${o.items ?? "—"}</b></span>
      <span class="pill">VENDORS <b>${o.vendors ?? "—"}</b></span>
      <span class="pill">PROCESSED <b>${o.processed ?? 0}</b></span>
      ${failed ? `<span class="pill"><span class="dot warn"></span>FAILED <b>${failed}</b></span>` : ""}`;
    $("#doc-count").textContent = o.documents;
    $("#inventory-note").textContent = `${o.formats?.join(" · ") ?? ""}`;
  } catch (err) {
    $("#pills").innerHTML = `<span class="pill"><span class="dot warn"></span>${esc(err.message)}</span>`;
  }
}

/* ──────────────────────────────── inbox ───────────────────────────────── */

async function loadInbox() {
  try {
    inbox = await api("/api/invoices");
  } catch (err) {
    $("#inbox-body").innerHTML = `<tr><td colspan="4" class="dim">${esc(err.message)}</td></tr>`;
    return;
  }
  renderInbox();
}

function renderInbox() {
  const q = $("#filter").value.trim().toLowerCase();
  const rows = inbox.filter(
    (r) =>
      !q ||
      r.filename.toLowerCase().includes(q) ||
      (r.vendor || "").toLowerCase().includes(q) ||
      (r.invoice_number || "").toLowerCase().includes(q)
  );

  $("#inbox-body").innerHTML =
    rows
      .map((r) => {
        const d = DECISION[r.decision];
        const last = d
          ? `<span class="badge ${d.cls}">${d.word}</span>`
          : `<span class="dim">unprocessed</span>`;
        return `<tr data-file="${esc(r.filename)}" class="${selected === r.filename ? "selected" : ""}">
          <td class="nowrap">${esc(r.filename)}</td>
          <td><span class="fmt">${esc(r.format)}</span></td>
          <td class="num">${r.total == null ? "<span class='dim'>—</span>" : esc(money(r.total))}</td>
          <td class="nowrap">${last}</td>
        </tr>`;
      })
      .join("") || `<tr><td colspan="4" class="dim">no documents match</td></tr>`;
}

$("#filter").addEventListener("input", renderInbox);

$("#inbox-body").addEventListener("click", (e) => {
  const tr = e.target.closest("tr[data-file]");
  if (tr) selectInvoice(tr.dataset.file);
});

async function selectInvoice(filename) {
  selected = filename;
  renderInbox();

  const detail = $("#detail");
  detail.className = "detail";
  detail.innerHTML = `<div class="section-body dim">reading ${esc(filename)}…</div>`;

  let doc;
  try {
    doc = await api(`/api/invoices/${encodeURIComponent(filename)}/raw`);
  } catch (err) {
    detail.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
    return;
  }

  const meta = inbox.find((r) => r.filename === filename) || {};
  detail.innerHTML = `
    <div class="detail-head">
      <div>
        <div class="detail-title">${esc(filename)}</div>
        <div class="detail-sub">
          <span class="fmt">${esc(doc.format)}</span>
          &nbsp;${doc.lines} lines · ${meta.bytes ?? 0} bytes
          ${meta.processed_at ? ` · last run ${esc(meta.processed_at)}` : ""}
        </div>
      </div>
      <button class="btn" id="run-btn">▶ RUN PIPELINE</button>
    </div>
    <div class="detail-body">
      ${priorRun(meta)}
      <div id="result"></div>
      <div class="section">
        <div class="section-head"><span>SOURCE DOCUMENT</span><span>${esc(doc.format).toUpperCase()}</span></div>
        <pre class="doc"></pre>
      </div>
    </div>`;

  detail.querySelector("pre.doc").textContent = doc.raw_text;
  $("#run-btn").addEventListener("click", () => runPipeline(filename));
}

/* The ledger already holds what the last run decided and why. Showing it here means
   reading a past verdict costs nothing, instead of a fresh 90-second model run. */
function priorRun(meta) {
  if (!meta.decision) return "";
  const d = DECISION[meta.decision] || DECISION.error;
  return `
    <div class="section" id="prior">
      <div class="section-head">
        <span>LAST RUN</span>
        <span>${esc(meta.processed_at || "")}</span>
      </div>
      <div class="section-body">
        <div class="verdict ${d.cls}">
          <span class="mark">${d.mark}</span>
          <span class="word">${d.word}</span>
          ${meta.invoice_number ? `<span class="badge info">${esc(meta.invoice_number)}</span>` : ""}
          ${meta.total != null ? `<span class="badge info">${esc(money(meta.total, meta.currency))}</span>` : ""}
        </div>
        <pre class="rationale ${d.cls}">${esc(meta.reason || "(no reasoning recorded)")}</pre>
        <div class="dim" style="margin-top:8px;font-size:11px">
          Recorded reasoning from the ledger. Run again for the full findings and audit trail.
        </div>
      </div>
    </div>`;
}

/* ─────────────────────────────── pipeline ─────────────────────────────── */

async function runPipeline(filename) {
  const btn = $("#run-btn");
  const target = $("#result");
  btn.disabled = true;
  btn.textContent = "RUNNING…";

  const started = Date.now();
  document.getElementById("prior")?.remove();
  target.innerHTML = `
    <div class="section"><div class="section-body">
      <div class="running">
        <span>EXECUTING GRAPH</span><span class="bar"><i></i></span><span id="elapsed">0.0s</span>
      </div>
      <div class="dim" style="margin-top:8px">
        extract → reconcile → validate → approve → critique → finalize
      </div>
    </div></div>`;

  const tick = setInterval(() => {
    const el = $("#elapsed");
    if (el) el.textContent = ((Date.now() - started) / 1000).toFixed(1) + "s";
  }, 100);

  try {
    const result = await api(`/api/invoices/${encodeURIComponent(filename)}/process`, {
      method: "POST",
    });
    renderResult(target, result, (Date.now() - started) / 1000);
    await loadInbox();
    await loadOverview();
    loaded.delete("ledger");
  } catch (err) {
    target.innerHTML = `<div class="error-box">${esc(err.message)}</div>`;
  } finally {
    clearInterval(tick);
    btn.disabled = false;
    btn.textContent = "▶ RUN PIPELINE";
  }
}

function renderResult(target, r, seconds) {
  const dec = r.decision ? DECISION[r.decision.decision] : DECISION.error;
  const inv = r.invoice;
  const rec = r.reconciliation;
  const c = r.counts;

  const counts = `<span class="counts">
      ${c.critical ? `<span class="badge bad">${c.critical} CRITICAL</span>` : ""}
      ${c.warning ? `<span class="badge warn">${c.warning} WARNING</span>` : ""}
      ${c.info ? `<span class="badge info">${c.info} INFO</span>` : ""}
      ${!c.total ? `<span class="badge ok">CLEAN</span>` : ""}
    </span>`;

  const findings = r.findings.length
    ? r.findings
        .slice()
        .sort((a, b) => rank(a.severity) - rank(b.severity))
        .map(
          (f) => `<div class="finding ${esc(f.severity)}">
            <span class="mark">${SEVERITY[f.severity] || "·"}</span>
            <div><div class="code">${esc(f.code)}</div><div class="msg">${esc(f.message)}</div></div>
          </div>`
        )
        .join("")
    : `<div class="dim">nothing flagged</div>`;

  const lineRows = (inv?.line_items || [])
    .map(
      (l, i) => `<tr>
        <td class="num dim">${i + 1}</td>
        <td>${esc(l.item)}${l.note ? ` <span class="dim">[${esc(l.note)}]</span>` : ""}</td>
        <td class="num">${esc(l.quantity)}</td>
        <td class="num">${esc(money(l.unit_price))}</td>
        <td class="num">${l.amount == null ? "<span class='dim'>—</span>" : esc(money(l.amount))}</td>
      </tr>`
    )
    .join("");

  const trail = r.audit_log
    .map(
      (e) => `<div class="trail-row ${esc(e.severity)}">
        <span class="t">${e.elapsed_ms ? Math.round(e.elapsed_ms) + "ms" : ""}</span>
        <span class="n">${esc(e.node)}.${esc(e.event)}</span>
        <span class="d">${esc(e.detail || "")}</span>
      </div>`
    )
    .join("");

  const pay = r.payment;
  const payLine =
    pay?.status === "success"
      ? `<span class="badge ok">PAID ${esc(money(pay.amount, pay.currency))}</span>
         <span class="dim">&nbsp;${esc(pay.reference)}</span>`
      : `<span class="badge warn">PAYMENT ${esc((pay?.status || "none").toUpperCase())}</span>`;

  target.innerHTML = `
    <div class="section">
      <div class="section-head"><span>DECISION</span><span>${seconds.toFixed(1)}s · ${r.extraction_attempts} extraction pass(es) · ${r.critique_rounds} critique round(s)</span></div>
      <div class="section-body">
        <div class="verdict ${dec.cls}">
          <span class="mark">${dec.mark}</span>
          <span class="word">${dec.word}</span>
          ${r.decision?.revised ? `<span class="badge violet">REVISED AFTER CRITIQUE</span>` : ""}
          ${counts}
        </div>
        <pre class="rationale ${dec.cls}">${esc(r.decision?.rationale || "No decision was reached.")}</pre>
        ${
          r.critique?.concerns?.length
            ? `<div style="margin-top:10px"><div class="dim" style="font-size:10px;letter-spacing:.14em">CRITIQUE</div>
               ${r.critique.concerns.map((x) => `<div class="dim">· ${esc(x)}</div>`).join("")}</div>`
            : ""
        }
        <div style="margin-top:12px">${payLine}</div>
      </div>
    </div>

    ${
      rec
        ? `<div class="section">
            <div class="section-head"><span>ARITHMETIC</span>
              <span class="badge ${rec.is_consistent ? "ok" : "bad"}">${rec.is_consistent ? "RECONCILES" : "DOES NOT RECONCILE"}</span></div>
            <div class="section-body">
              <dl class="kv">
                <dt>subtotal</dt><dd>computed ${esc(money(rec.computed_subtotal))} · stated ${esc(money(rec.stated_subtotal))}</dd>
                <dt>tax</dt><dd>computed ${esc(money(rec.computed_tax))} · stated ${esc(money(rec.stated_tax))}</dd>
                <dt>total</dt><dd>computed ${esc(money(rec.computed_total))} · stated ${esc(money(rec.stated_total))}</dd>
                <dt>per-SKU totals</dt><dd>${esc(Object.entries(rec.aggregated_quantities || {}).map(([k, v]) => `${k} ×${v}`).join("  ·  ") || "—")}</dd>
              </dl>
              ${rec.discrepancies.map((d) => `<div style="margin-top:8px;color:var(--red)">· ${esc(d)}</div>`).join("")}
            </div>
          </div>`
        : ""
    }

    <div class="section">
      <div class="section-head"><span>FINDINGS</span><span>${c.total}</span></div>
      <div class="section-body">${findings}</div>
    </div>

    ${
      inv
        ? `<div class="section">
            <div class="section-head"><span>EXTRACTED</span><span>${esc(inv.invoice_number || "unnumbered")}</span></div>
            <div class="section-body">
              <dl class="kv">
                <dt>vendor</dt><dd>${esc(inv.vendor_name || "MISSING")}</dd>
                <dt>issued</dt><dd>${esc(inv.invoice_date || inv.invoice_date_raw || "—")}</dd>
                <dt>due</dt><dd>${esc(inv.due_date || inv.due_date_raw || "—")}</dd>
                <dt>terms</dt><dd>${esc(inv.payment_terms || "—")}</dd>
                <dt>total</dt><dd>${esc(money(inv.total, inv.currency))}</dd>
                ${inv.notes ? `<dt>notes</dt><dd>${esc(inv.notes)}</dd>` : ""}
              </dl>
              <table class="grid wide" style="margin-top:12px">
                <thead><tr><th class="num">#</th><th>ITEM</th><th class="num">QTY</th><th class="num">UNIT</th><th class="num">STATED</th></tr></thead>
                <tbody>${lineRows}</tbody>
              </table>
            </div>
          </div>`
        : ""
    }

    ${r.errors.length ? `<div class="error-box">${r.errors.map(esc).join("<br>")}</div>` : ""}

    <div class="section">
      <div class="section-head"><span>AUDIT TRAIL</span><span>${r.audit_log.length} events</span></div>
      <div class="section-body trail">${trail}</div>
    </div>`;
}

const rank = (s) => ({ critical: 0, warning: 1, info: 2 })[s] ?? 3;

/* ─────────────────────────── reference tables ─────────────────────────── */

async function loadInventory() {
  const rows = await api("/api/inventory").catch((e) => e);
  if (rows instanceof Error) return fail("#inventory-body", 8, rows);

  $("#inventory-body").innerHTML = rows
    .map((r) => {
      const cls = r.status === "discontinued" ? "bad" : r.stock === 0 ? "warn" : "ok";
      return `<tr>
        <td><b>${esc(r.item)}</b></td>
        <td class="dim">${esc(r.display_name)}</td>
        <td class="num"><span class="badge ${r.stock === 0 ? "bad" : r.stock < 10 ? "warn" : "ok"}">${r.stock}</span></td>
        <td class="num">${esc(money(r.unit_price))}</td>
        <td class="dim">${esc(r.currency)}</td>
        <td class="dim">${esc(r.category || "—")}</td>
        <td><span class="badge ${cls}">${esc(r.status)}</span></td>
        <td class="dim">${esc(r.notes || "")}</td>
      </tr>`;
    })
    .join("");
}

async function loadVendors() {
  const rows = await api("/api/vendors").catch((e) => e);
  if (rows instanceof Error) return fail("#vendors-body", 6, rows);

  const cls = { approved: "ok", unverified: "warn", blocked: "bad" };
  $("#vendors-body").innerHTML = rows
    .map(
      (r) => `<tr>
        <td><b>${esc(r.name)}</b></td>
        <td><span class="badge ${cls[r.status] || "info"}">${esc(r.status)}</span></td>
        <td class="dim">${esc(r.payment_terms || "—")}</td>
        <td class="dim">${esc(r.currency)}</td>
        <td class="dim">${esc(r.also_known_as || "—")}</td>
        <td class="dim">${esc(r.notes || "")}</td>
      </tr>`
    )
    .join("");
}

async function loadLedger() {
  const rows = await api("/api/ledger").catch((e) => e);
  if (rows instanceof Error) return fail("#ledger-body", 7, rows);

  $("#ledger-body").innerHTML = rows.length
    ? rows
        .map((r) => {
          const d = DECISION[r.decision] || { cls: "info", word: r.decision || "—" };
          return `<tr data-ledger-id="${r.id}" class="clickable">
            <td class="num dim"><span class="caret">▸</span> ${r.id}</td>
            <td>${
              r.invoice_number
                ? `<b>${esc(r.invoice_number)}</b>${r.revision ? ` <span class="badge violet">${esc(r.revision)}</span>` : ""}`
                : `<span class="dim">${esc((r.source_path || "").split(/[\\/]/).pop() || "—")}</span>`
            }</td>
            <td class="dim">${esc(r.vendor_name || "—")}</td>
            <td class="num">${esc(money(r.total, r.currency))}</td>
            <td><span class="badge ${d.cls}">${esc(d.word)}</span></td>
            <td class="dim" style="font-size:10px">${
              r.decision === "error"
                ? `<span style="color:var(--red)">${esc((r.reason || "").slice(0, 70))}</span>`
                : esc((r.line_hash || "").slice(0, 12))
            }</td>
            <td class="dim nowrap">${esc(r.processed_at)}</td>
          </tr>
          <tr class="reason-row" hidden data-reason-for="${r.id}">
            <td colspan="7">
              <pre class="rationale ${d.cls}">${esc(r.reason || "(no reasoning recorded)")}</pre>
            </td>
          </tr>`;
        })
        .join("")
    : `<tr><td colspan="7" class="dim">nothing processed yet — run an invoice from the inbox</td></tr>`;
}

$("#ledger-body").addEventListener("click", (e) => {
  const row = e.target.closest("tr[data-ledger-id]");
  if (!row) return;
  const detail = document.querySelector(`tr[data-reason-for="${row.dataset.ledgerId}"]`);
  if (!detail) return;
  detail.hidden = !detail.hidden;
  row.classList.toggle("expanded", !detail.hidden);
});

$("#clear-ledger").addEventListener("click", async () => {
  await api("/api/ledger", { method: "DELETE" });
  await Promise.all([loadLedger(), loadInbox(), loadOverview()]);
});

function fail(sel, cols, err) {
  $(sel).innerHTML = `<tr><td colspan="${cols}" class="dim">${esc(err.message)}</td></tr>`;
}

/* ──────────────────────────────── boot ────────────────────────────────── */

loaded.add("inbox");
loadOverview();
loadInbox();
