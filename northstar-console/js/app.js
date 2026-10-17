import { AGENT, PROCEDURES, resolveCase, defaultPath } from "./data.js";

const params = new URLSearchParams(location.search);
const procedureId = PROCEDURES[params.get("procedure")] ? params.get("procedure") : "warranty";
const pathId = params.get("path") || params.get("scenario") || defaultPath(procedureId);

const state = {
  procedureId,
  pathId,
  view: PROCEDURES[procedureId].nav[0][0],
  caseId: params.get("case") || null,
  case: null,
  proc: null,
  path: null,
  query: "",
  customerFound: false,
  ticketOpen: false,
  deviceSelected: false,
  warrantyChecked: false,
  diagRan: false,
  repairCreated: false,
  paidRecorded: false,
  stockChecked: false,
  shipped: false,
  backordered: false,
  severityAssessed: false,
  escalated: false,
  ticketUpdated: false,
  repairNotes: "",
  ticketStatus: "",
  ticketNote: "",
  autoplay: params.get("autoplay") === "1",
  hud: params.get("hud") !== "0",
  demoRunning: false,
  currentOp: null,
};

function bootCase() {
  const { proc, path, case: c, pathId: pid } = resolveCase(state.procedureId, state.pathId, state.caseId);
  state.proc = proc;
  state.path = path;
  state.pathId = pid;
  state.case = { ...c };
  state.caseId = c.customer_id;
  if (state.procedureId === "warranty") {
    state.case.warranty_status = state.pathId === "expired" ? "EXPIRED" : "ACTIVE";
  }
  if (state.procedureId === "parts") {
    state.case.stock = state.pathId === "backorder" ? "OUT_OF_STOCK" : "IN_STOCK";
  }
  if (state.procedureId === "escalate") {
    state.case.severity = state.pathId === "p1" ? "P1" : state.case.ticket_priority || "P2";
  }
}

bootCase();

const app = document.getElementById("app");
const cursorEl = document.getElementById("cursor");

function toast(msg) {
  const t = document.getElementById("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 2200);
}

function badge(text, kind) {
  return `<span class="badge ${kind}">${text}</span>`;
}

function setView(view) {
  state.view = view;
  render();
}

function resetFlow() {
  Object.assign(state, {
    view: state.proc.nav[0][0],
    query: "",
    customerFound: false,
    ticketOpen: false,
    deviceSelected: false,
    warrantyChecked: false,
    diagRan: false,
    repairCreated: false,
    paidRecorded: false,
    stockChecked: false,
    shipped: false,
    backordered: false,
    severityAssessed: false,
    escalated: false,
    ticketUpdated: false,
    repairNotes: "",
    ticketStatus: "",
    ticketNote: "",
    currentOp: null,
    demoRunning: false,
  });
  bootCase();
  render();
}

function doneFlags() {
  if (state.procedureId === "warranty")
    return state.ticketUpdated && (state.repairCreated || state.paidRecorded);
  if (state.procedureId === "parts") return state.ticketUpdated && (state.shipped || state.backordered);
  if (state.procedureId === "escalate") return state.ticketUpdated && state.escalated;
  return false;
}

/* ---------- shared bits ---------- */

function customersView() {
  const c = state.case;
  return `
    <h1 class="page-title">Customers — search & open record</h1>
    <p class="page-sub">${state.proc.title}</p>
    <div class="card">
      <h3>Customer search</h3>
      <div class="row">
        <div class="field" style="flex:1">
          <label>Query</label>
          <input data-testid="customer-query" value="${esc(state.query)}" placeholder="Customer ID or name" />
        </div>
        <button class="btn btn-primary" data-testid="btn-search-customer" data-demo-target="search-customer">Search</button>
      </div>
    </div>
    ${
      state.customerFound
        ? `<div class="card"><h3>Results</h3>
        <table class="table"><thead><tr><th>ID</th><th>Name</th><th>Tier</th></tr></thead>
        <tbody><tr class="clickable selected" data-testid="row-customer"><td>${c.customer_id}</td><td>${c.name}</td><td>${c.tier}</td></tr></tbody></table>
        <div class="toolbar"><button class="btn btn-primary" data-testid="btn-open-tickets" data-demo-target="goto-tickets">Open tickets</button></div></div>`
        : ""
    }`;
}

function ticketsView() {
  const c = state.case;
  if (!state.ticketOpen) {
    return `
      <h1 class="page-title">Tickets — queue</h1>
      <p class="page-sub">Open the support case</p>
      <div class="card">
        <table class="table"><thead><tr><th>Ticket</th><th>Subject</th><th>Priority</th></tr></thead>
        <tbody>
          <tr class="clickable" data-testid="row-ticket" data-demo-target="open-ticket">
            <td>${c.ticket_id}</td><td>${c.ticket_subject}</td><td>${c.ticket_priority}</td>
          </tr>
        </tbody></table>
      </div>`;
  }
  const statusDefault =
    state.procedureId === "parts"
      ? state.shipped
        ? "Parts shipped"
        : "Parts backordered"
      : state.procedureId === "escalate"
        ? state.pathId === "p1"
          ? "Escalated to on-call"
          : "Assigned L2"
        : state.pathId === "expired"
          ? "Paid options offered"
          : "Repair scheduled";
  return `
    <h1 class="page-title">Tickets — update</h1>
    <div class="card">
      <h3>Update ${c.ticket_id}</h3>
      <div class="field" style="margin-bottom:10px"><label>Status</label>
        <input data-testid="ticket-status" value="${esc(state.ticketStatus || statusDefault)}" /></div>
      <div class="field" style="margin-bottom:10px"><label>Internal note</label>
        <input data-testid="ticket-note" value="${esc(state.ticketNote || "Processed via Northstar console")}" /></div>
      ${state.ticketUpdated ? badge("Saved", "ok") : ""}
      <div class="toolbar">
        <button class="btn btn-primary" data-testid="btn-save-ticket" data-demo-target="save-ticket">Save ticket</button>
        ${
          state.procedureId !== "escalate"
            ? `<button class="btn btn-secondary" data-testid="btn-goto-devices" data-demo-target="goto-devices">Open linked device</button>`
            : `<button class="btn btn-secondary" data-testid="btn-goto-customers" data-demo-target="goto-customers">View customer</button>
               <button class="btn btn-secondary" data-testid="btn-goto-escalate" data-demo-target="goto-escalate">Escalation desk</button>`
        }
      </div>
    </div>`;
}

function devicesView() {
  const c = state.case;
  return `
    <h1 class="page-title">Devices — registry</h1>
    <div class="card">
      <table class="table"><thead><tr><th>Device</th><th>Serial</th><th>Model</th><th>Health</th></tr></thead>
      <tbody>
        <tr class="clickable ${state.deviceSelected ? "selected" : ""}" data-testid="row-device" data-demo-target="select-device">
          <td>${c.device_id}</td><td>${c.serial}</td><td>${c.model}</td><td>${badge(c.health, c.health === "OK" ? "ok" : "warn")}</td>
        </tr>
      </tbody></table>
    </div>
    ${
      state.deviceSelected
        ? `<div class="card"><h3>Device record</h3>
        <dl class="kv"><dt>Owner</dt><dd>${c.name} (${c.customer_id})</dd>
        <dt>Purchase</dt><dd>${c.purchase}</dd></dl>
        <div class="toolbar">
          ${
            state.procedureId === "warranty"
              ? `<button class="btn btn-primary" data-testid="btn-check-warranty" data-demo-target="check-warranty">Check warranty</button>
                 <button class="btn btn-secondary" data-testid="btn-run-diag" data-demo-target="run-diag">Run diagnostics</button>`
              : `<button class="btn btn-primary" data-testid="btn-goto-inventory" data-demo-target="goto-inventory">Check parts stock</button>`
          }
        </div></div>`
        : ""
    }`;
}

function warrantyView() {
  const c = state.case;
  return `
    <h1 class="page-title">Warranty — coverage</h1>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <h3 style="margin:0">Warranty panel</h3>
        ${state.warrantyChecked ? badge(c.warranty_status, c.warranty_status === "ACTIVE" ? "ok" : "danger") : badge("Not checked", "neutral")}
      </div>
      ${
        state.warrantyChecked
          ? `<dl class="kv" style="margin-top:12px">
            <dt>ID</dt><dd>${c.warranty_id}</dd><dt>End</dt><dd>${c.warranty_end}</dd>
            <dt>Status</dt><dd>${badge(c.warranty_status, c.warranty_status === "ACTIVE" ? "ok" : "danger")}</dd></dl>
            <div class="toolbar"><button class="btn btn-primary" data-testid="btn-goto-diag" data-demo-target="goto-diag">Continue to diagnostics</button></div>`
          : `<div class="toolbar"><button class="btn btn-primary" data-testid="btn-check-warranty-page" data-demo-target="check-warranty" ${state.deviceSelected ? "" : "disabled"}>Check warranty</button></div>`
      }
    </div>`;
}

function diagnosticsView() {
  const c = state.case;
  return `
    <h1 class="page-title">Diagnostics — run</h1>
    <div class="card">
      ${
        state.diagRan
          ? `<table class="table"><thead><tr><th>Check</th><th>Result</th><th>Code</th></tr></thead>
            <tbody><tr><td>Primary</td><td>${badge(c.diag_result, "danger")}</td><td>${c.diag_code}</td></tr></tbody></table>
            <div class="toolbar"><button class="btn btn-primary" data-testid="btn-goto-repair" data-demo-target="goto-repair">${
              state.pathId === "expired" ? "Offer paid options" : "Create repair order"
            }</button></div>`
          : `<div class="toolbar"><button class="btn btn-primary" data-testid="btn-run-diag-page" data-demo-target="run-diag" ${
              state.deviceSelected ? "" : "disabled"
            }>Run diagnostics</button></div>`
      }
    </div>`;
}

function repairView() {
  const c = state.case;
  if (state.pathId === "expired") {
    return `
      <h1 class="page-title">Repair — paid options</h1>
      <div class="card">
        <p style="color:var(--muted)">Warranty ended ${c.warranty_end}.</p>
        <div class="field"><label>Option</label>
          <select data-testid="paid-choice"><option>Depot repair — paid</option><option>Onsite — paid</option></select></div>
        ${state.paidRecorded ? badge("Recorded", "ok") : ""}
        <div class="toolbar">
          <button class="btn btn-primary" data-testid="btn-record-paid" data-demo-target="record-paid">Record selection</button>
          <button class="btn btn-secondary" data-testid="btn-goto-ticket-update" data-demo-target="goto-ticket-update" ${state.paidRecorded ? "" : "disabled"}>Update ticket</button>
        </div>
      </div>`;
  }
  return `
    <h1 class="page-title">Repair — create order</h1>
    <div class="card">
      <div class="field" style="margin-bottom:8px"><label>Device</label><input class="readonly" readonly value="${c.device_id}" /></div>
      <div class="field" style="margin-bottom:8px"><label>Fault</label><input class="readonly" readonly value="${c.diag_code}" /></div>
      <div class="field" style="margin-bottom:8px"><label>Notes</label>
        <input data-testid="repair-notes" data-demo-target="repair-notes" value="${esc(state.repairNotes)}" /></div>
      ${state.repairCreated ? badge(`Created ${c.repair_id}`, "ok") : ""}
      <div class="toolbar">
        <button class="btn btn-primary" data-testid="btn-submit-repair" data-demo-target="submit-repair">Submit repair</button>
        <button class="btn btn-secondary" data-testid="btn-goto-ticket-update" data-demo-target="goto-ticket-update" ${state.repairCreated ? "" : "disabled"}>Update ticket</button>
      </div>
    </div>`;
}

function inventoryView() {
  const c = state.case;
  return `
    <h1 class="page-title">Inventory — spare parts</h1>
    <p class="page-sub">Check depot stock for the failed FRU</p>
    <div class="card">
      <h3>${c.part_name} · ${c.part_sku}</h3>
      ${
        state.stockChecked
          ? `<dl class="kv"><dt>Stock</dt><dd>${
              c.stock === "IN_STOCK" ? badge("IN_STOCK", "ok") : badge("OUT_OF_STOCK", "danger")
            }</dd>
            <dt>Device</dt><dd>${c.device_id}</dd></dl>
            <div class="toolbar">
              <button class="btn btn-primary" data-testid="btn-goto-shipping" data-demo-target="goto-shipping">${
                c.stock === "IN_STOCK" ? "Create shipment" : "Create backorder"
              }</button>
            </div>`
          : `<div class="toolbar"><button class="btn btn-primary" data-testid="btn-check-stock" data-demo-target="check-stock" ${
              state.deviceSelected ? "" : "disabled"
            }>Check parts stock</button></div>`
      }
    </div>`;
}

function shippingView() {
  const c = state.case;
  if (state.pathId === "backorder" || c.stock === "OUT_OF_STOCK") {
    return `
      <h1 class="page-title">Shipping — backorder</h1>
      <div class="card">
        <h3>Create backorder</h3>
        <dl class="kv"><dt>SKU</dt><dd>${c.part_sku}</dd><dt>Ticket</dt><dd>${c.ticket_id}</dd></dl>
        ${state.backordered ? badge(`Backorder ${c.backorder_id}`, "warn") : ""}
        <div class="toolbar">
          <button class="btn btn-primary" data-testid="btn-create-backorder" data-demo-target="create-backorder">Submit backorder</button>
          <button class="btn btn-secondary" data-testid="btn-goto-ticket-update" data-demo-target="goto-ticket-update" ${state.backordered ? "" : "disabled"}>Update ticket</button>
        </div>
      </div>`;
  }
  return `
    <h1 class="page-title">Shipping — create shipment</h1>
    <div class="card">
      <h3>Ship spare to customer</h3>
      <dl class="kv"><dt>SKU</dt><dd>${c.part_sku}</dd><dt>Qty</dt><dd>1</dd><dt>Ship to</dt><dd>${c.name}</dd></dl>
      ${state.shipped ? badge(`Shipment ${c.shipment_id}`, "ok") : ""}
      <div class="toolbar">
        <button class="btn btn-primary" data-testid="btn-create-shipment" data-demo-target="create-shipment">Create shipment</button>
        <button class="btn btn-secondary" data-testid="btn-goto-ticket-update" data-demo-target="goto-ticket-update" ${state.shipped ? "" : "disabled"}>Update ticket</button>
      </div>
    </div>`;
}

function escalateView() {
  const c = state.case;
  return `
    <h1 class="page-title">Escalation — severity desk</h1>
    <p class="page-sub">Route P1 to on-call vs standard L2</p>
    <div class="card">
      <h3>Ticket ${c.ticket_id}</h3>
      <dl class="kv"><dt>Reported priority</dt><dd>${c.ticket_priority}</dd>
        <dt>Assessed</dt><dd>${state.severityAssessed ? badge(c.severity, c.severity === "P1" ? "danger" : "warn") : badge("Pending", "neutral")}</dd></dl>
      <div class="toolbar">
        <button class="btn btn-primary" data-testid="btn-assess-severity" data-demo-target="assess-severity" ${state.ticketOpen && state.customerFound ? "" : "disabled"}>Assess severity</button>
      </div>
      ${
        state.severityAssessed
          ? `<div class="toolbar" style="margin-top:12px">
              ${
                state.pathId === "p1"
                  ? `<button class="btn btn-primary" data-testid="btn-page-oncall" data-demo-target="page-oncall">Page on-call (${c.oncall})</button>`
                  : `<button class="btn btn-primary" data-testid="btn-assign-l2" data-demo-target="assign-l2">Assign ${c.l2_queue}</button>`
              }
            </div>
            ${state.escalated ? badge("Escalation recorded", "ok") : ""}
            <div class="toolbar">
              <button class="btn btn-secondary" data-testid="btn-goto-ticket-update" data-demo-target="goto-ticket-update" ${state.escalated ? "" : "disabled"}>Update ticket</button>
            </div>`
          : ""
      }
    </div>`;
}

function page() {
  switch (state.view) {
    case "tickets":
      return ticketsView();
    case "devices":
      return devicesView();
    case "warranty":
      return warrantyView();
    case "diagnostics":
      return diagnosticsView();
    case "repair":
      return repairView();
    case "inventory":
      return inventoryView();
    case "shipping":
      return shippingView();
    case "escalate":
      return escalateView();
    case "customers":
    default:
      return customersView();
  }
}

function shell(inner) {
  const pathOptions = Object.entries(state.proc.paths)
    .map(([id, p]) => `<option value="${id}" ${state.pathId === id ? "selected" : ""}>${p.label}</option>`)
    .join("");
  const procOptions = Object.values(PROCEDURES)
    .map((p) => `<option value="${p.id}" ${state.procedureId === p.id ? "selected" : ""}>${p.title}</option>`)
    .join("");
  return `
    <header class="topbar">
      <div class="brand">Northstar <span>Support Console</span></div>
      <input class="top-search" readonly placeholder="Search customers, tickets, devices..." />
      <div class="top-right">
        <span class="pill env">${AGENT.env}</span>
        <span class="pill user">${AGENT.name}</span>
      </div>
    </header>
    <div class="shell">
      <nav class="sidebar">
        ${state.proc.nav
          .map(
            ([id, label]) =>
              `<button class="nav-item ${state.view === id ? "active" : ""}" data-nav="${id}" data-testid="nav-${id}">${label}</button>`
          )
          .join("")}
      </nav>
      <main class="main">${inner}</main>
    </div>
    ${
      state.hud
        ? `<div class="demo-bar">
      <strong>Agent</strong>
      <label>procedure <select id="procedure-select" data-testid="procedure-select">${procOptions}</select></label>
      <label>path <select id="path-select" data-testid="path-select">${pathOptions}</select></label>
      <span class="op">${state.currentOp || "idle"}</span>
      <button type="button" id="btn-demo" data-testid="btn-demo">Play flow</button>
      <button type="button" id="btn-reset">Reset</button>
    </div>`
        : ""
    }
    <div id="toast" class="toast"></div>`;
}

function esc(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/"/g, "&quot;")
    .replace(/</g, "&lt;");
}

function render() {
  app.innerHTML = shell(page());
  bind();
}

/* ---------- mutations ---------- */

function searchCustomer() {
  const el = document.querySelector('[data-testid="customer-query"]');
  state.query = el?.value?.trim() || state.case.customer_id;
  state.customerFound = true;
  state.currentOp = "LOOKUP_CUSTOMER";
  render();
  toast(`Customer ${state.case.customer_id}`);
}

function openTicket() {
  state.ticketOpen = true;
  state.currentOp = "GET_TICKET";
  state.view = "tickets";
  render();
  toast(`Opened ${state.case.ticket_id}`);
}

function selectDevice() {
  state.deviceSelected = true;
  state.currentOp = "GET_DEVICE";
  state.view = "devices";
  render();
}

function checkWarranty() {
  state.deviceSelected = true;
  state.warrantyChecked = true;
  state.currentOp = "CHECK_WARRANTY";
  state.view = "warranty";
  render();
  toast(`Warranty ${state.case.warranty_status}`);
}

function runDiag() {
  state.diagRan = true;
  state.currentOp = "GET_DIAGNOSTICS";
  state.view = "diagnostics";
  render();
}

function submitRepair() {
  const n = document.querySelector('[data-testid="repair-notes"]');
  if (n) state.repairNotes = n.value || "Advance RMA";
  state.repairCreated = true;
  state.currentOp = "CREATE_REPAIR";
  render();
  toast(`Repair ${state.case.repair_id}`);
}

function recordPaid() {
  state.paidRecorded = true;
  state.currentOp = "RECORD_PAID_OPTIONS";
  render();
}

function checkStock() {
  state.stockChecked = true;
  state.currentOp = "CHECK_PARTS_STOCK";
  state.view = "inventory";
  render();
  toast(state.case.stock);
}

function createShipment() {
  state.shipped = true;
  state.currentOp = "CREATE_PARTS_SHIPMENT";
  state.view = "shipping";
  render();
  toast(state.case.shipment_id);
}

function createBackorder() {
  state.backordered = true;
  state.currentOp = "CREATE_BACKORDER";
  state.view = "shipping";
  render();
  toast(state.case.backorder_id);
}

function assessSeverity() {
  state.severityAssessed = true;
  state.currentOp = "ASSESS_SEVERITY";
  state.view = "escalate";
  render();
  toast(`Severity ${state.case.severity}`);
}

function pageOncall() {
  state.escalated = true;
  state.currentOp = "PAGE_ONCALL";
  render();
  toast(`Paged ${state.case.oncall}`);
}

function assignL2() {
  state.escalated = true;
  state.currentOp = "ASSIGN_L2_QUEUE";
  render();
  toast(`Assigned ${state.case.l2_queue}`);
}

function saveTicket() {
  const st = document.querySelector('[data-testid="ticket-status"]');
  const note = document.querySelector('[data-testid="ticket-note"]');
  if (st) state.ticketStatus = st.value;
  if (note) state.ticketNote = note.value;
  state.ticketUpdated = true;
  state.currentOp = "UPDATE_TICKET";
  state.view = "tickets";
  state.ticketOpen = true;
  render();
  toast("Ticket saved");
}

function on(testid, fn) {
  document.querySelector(`[data-testid="${testid}"]`)?.addEventListener("click", fn);
}

function bind() {
  document.querySelectorAll("[data-nav]").forEach((el) =>
    el.addEventListener("click", () => setView(el.getAttribute("data-nav")))
  );
  on("btn-search-customer", searchCustomer);
  on("btn-open-tickets", () => setView("tickets"));
  on("row-ticket", openTicket);
  on("btn-goto-devices", () => setView("devices"));
  on("btn-goto-customers", () => setView("customers"));
  on("row-device", selectDevice);
  on("btn-check-warranty", checkWarranty);
  on("btn-check-warranty-page", checkWarranty);
  on("btn-run-diag", runDiag);
  on("btn-run-diag-page", runDiag);
  on("btn-goto-diag", () => setView("diagnostics"));
  on("btn-goto-repair", () => setView("repair"));
  on("btn-submit-repair", submitRepair);
  on("btn-record-paid", recordPaid);
  on("btn-goto-inventory", () => setView("inventory"));
  on("btn-check-stock", checkStock);
  on("btn-goto-shipping", () => setView("shipping"));
  on("btn-create-shipment", createShipment);
  on("btn-create-backorder", createBackorder);
  on("btn-goto-escalate", () => setView("escalate"));
  on("btn-assess-severity", assessSeverity);
  on("btn-page-oncall", pageOncall);
  on("btn-assign-l2", assignL2);
  on("btn-goto-ticket-update", () => {
    state.ticketOpen = true;
    setView("tickets");
  });
  on("btn-save-ticket", saveTicket);

  document.getElementById("procedure-select")?.addEventListener("change", (e) => {
    state.procedureId = e.target.value;
    state.pathId = defaultPath(state.procedureId);
    state.caseId = null;
    resetFlow();
    history.replaceState(null, "", `?procedure=${state.procedureId}&path=${state.pathId}${state.hud ? "" : "&hud=0"}`);
  });
  document.getElementById("path-select")?.addEventListener("change", (e) => {
    state.pathId = e.target.value;
    state.caseId = null;
    resetFlow();
    history.replaceState(null, "", `?procedure=${state.procedureId}&path=${state.pathId}${state.hud ? "" : "&hud=0"}`);
  });
  document.getElementById("btn-demo")?.addEventListener("click", () => startDemo());
  document.getElementById("btn-reset")?.addEventListener("click", () => resetFlow());
}

/* ---------- agent API ---------- */

function snapshot() {
  return {
    app: "northstar-support-console",
    procedure: state.procedureId,
    family: state.proc.family,
    path: state.pathId,
    view: state.view,
    currentOp: state.currentOp,
    case: {
      customer_id: state.case.customer_id,
      ticket_id: state.case.ticket_id,
      device_id: state.case.device_id,
      warranty_status: state.case.warranty_status,
      stock: state.case.stock,
      severity: state.case.severity,
    },
    flags: {
      customerFound: state.customerFound,
      ticketOpen: state.ticketOpen,
      deviceSelected: state.deviceSelected,
      warrantyChecked: state.warrantyChecked,
      diagRan: state.diagRan,
      repairCreated: state.repairCreated,
      paidRecorded: state.paidRecorded,
      stockChecked: state.stockChecked,
      shipped: state.shipped,
      backordered: state.backordered,
      severityAssessed: state.severityAssessed,
      escalated: state.escalated,
      ticketUpdated: state.ticketUpdated,
      done: doneFlags(),
    },
    targetOps: state.path.ops,
  };
}

function targets() {
  return [...document.querySelectorAll("[data-testid]")].map((el) => {
    const r = el.getBoundingClientRect();
    return {
      testid: el.getAttribute("data-testid"),
      text: (el.innerText || el.value || "").trim().slice(0, 80),
      disabled: !!el.disabled,
      visible: r.width > 0 && r.height > 0,
      rect: { x: r.x, y: r.y, width: r.width, height: r.height },
    };
  });
}

function availableActions() {
  const out = [];
  const add = (action, testid, why) => {
    const el = testid ? document.querySelector(`[data-testid="${testid}"]`) : null;
    if (testid && (!el || el.disabled)) return;
    out.push({ action, testid, why });
  };

  if (state.procedureId === "escalate") {
    if (!state.ticketOpen) add("click", "row-ticket", "GET_TICKET");
    else if (!state.customerFound) {
      add("click", "btn-goto-customers", "go customers");
      add("type", "customer-query", "LOOKUP_CUSTOMER");
      add("click", "btn-search-customer", "LOOKUP_CUSTOMER");
    } else if (!state.severityAssessed) {
      add("click", "btn-goto-escalate", "escalation desk");
      add("click", "btn-assess-severity", "ASSESS_SEVERITY");
    } else if (!state.escalated) {
      add("click", state.pathId === "p1" ? "btn-page-oncall" : "btn-assign-l2", state.pathId === "p1" ? "PAGE_ONCALL" : "ASSIGN_L2_QUEUE");
    } else if (!state.ticketUpdated) {
      add("click", "btn-goto-ticket-update", "ticket");
      add("click", "btn-save-ticket", "UPDATE_TICKET");
    }
    return out;
  }

  // warranty + parts share CRM front
  if (!state.customerFound) {
    add("type", "customer-query", `type ${state.case.customer_id}`);
    add("click", "btn-search-customer", "LOOKUP_CUSTOMER");
  } else if (state.view === "customers") add("click", "btn-open-tickets", "tickets");
  else if (!state.ticketOpen) add("click", "row-ticket", "GET_TICKET");
  else if (!state.deviceSelected) {
    add("click", "btn-goto-devices", "devices");
    add("click", "row-device", "GET_DEVICE");
  } else if (state.procedureId === "warranty") {
    if (!state.warrantyChecked) add("click", "btn-check-warranty", "CHECK_WARRANTY");
    else if (!state.diagRan) {
      add("click", "btn-goto-diag", "diagnostics");
      add("click", "btn-run-diag-page", "GET_DIAGNOSTICS");
    } else if (state.pathId === "active" && !state.repairCreated) {
      add("click", "btn-goto-repair", "repair");
      add("click", "btn-submit-repair", "CREATE_REPAIR");
    } else if (state.pathId === "expired" && !state.paidRecorded) {
      add("click", "btn-goto-repair", "paid");
      add("click", "btn-record-paid", "RECORD_PAID_OPTIONS");
    } else if (!state.ticketUpdated) {
      add("click", "btn-goto-ticket-update", "ticket");
      add("click", "btn-save-ticket", "UPDATE_TICKET");
    }
  } else if (state.procedureId === "parts") {
    if (!state.stockChecked) {
      add("click", "btn-goto-inventory", "inventory");
      add("click", "btn-check-stock", "CHECK_PARTS_STOCK");
    } else if (state.pathId === "in_stock" && !state.shipped) {
      add("click", "btn-goto-shipping", "shipping");
      add("click", "btn-create-shipment", "CREATE_PARTS_SHIPMENT");
    } else if (state.pathId === "backorder" && !state.backordered) {
      add("click", "btn-goto-shipping", "shipping");
      add("click", "btn-create-backorder", "CREATE_BACKORDER");
    } else if (!state.ticketUpdated) {
      add("click", "btn-goto-ticket-update", "ticket");
      add("click", "btn-save-ticket", "UPDATE_TICKET");
    }
  }
  return out;
}

function observe() {
  return {
    ...snapshot(),
    availableActions: availableActions(),
    targets: targets(),
    instructions: "Mimic a human via data-testid click/type. Follow availableActions or playbook ops.",
  };
}

function act(action, payload = {}) {
  const a = String(action).toLowerCase();
  if (a === "observe") return observe();
  if (a === "reset") {
    resetFlow();
    return observe();
  }
  if (a === "set_procedure") {
    state.procedureId = PROCEDURES[payload.procedure] ? payload.procedure : "warranty";
    state.pathId = payload.path || defaultPath(state.procedureId);
    state.caseId = payload.caseId || null;
    resetFlow();
    return observe();
  }
  if (a === "goto") {
    setView(payload.view || payload.tab);
    return observe();
  }
  if (a === "type") {
    const el = document.querySelector(`[data-testid="${payload.testid}"]`);
    if (!el) throw new Error("missing testid");
    el.focus();
    el.value = payload.text ?? "";
    el.dispatchEvent(new Event("input", { bubbles: true }));
    return observe();
  }
  if (a === "click") {
    const el = document.querySelector(`[data-testid="${payload.testid}"]`);
    if (!el || el.disabled) throw new Error(`click failed ${payload.testid}`);
    el.click();
    return observe();
  }
  if (a === "playbook_step") {
    const op = payload.op;
    const run = {
      LOOKUP_CUSTOMER: () => {
        setView("customers");
        state.query = state.case.customer_id;
        searchCustomer();
      },
      GET_TICKET: () => openTicket(),
      GET_DEVICE: () => {
        setView("devices");
        selectDevice();
      },
      CHECK_WARRANTY: () => checkWarranty(),
      GET_DIAGNOSTICS: () => runDiag(),
      CREATE_REPAIR: () => {
        setView("repair");
        submitRepair();
      },
      RECORD_PAID_OPTIONS: () => {
        setView("repair");
        recordPaid();
      },
      CHECK_PARTS_STOCK: () => {
        state.deviceSelected = true;
        checkStock();
      },
      CREATE_PARTS_SHIPMENT: () => {
        setView("shipping");
        createShipment();
      },
      CREATE_BACKORDER: () => {
        setView("shipping");
        createBackorder();
      },
      ASSESS_SEVERITY: () => {
        state.ticketOpen = true;
        state.customerFound = true;
        assessSeverity();
      },
      PAGE_ONCALL: () => pageOncall(),
      ASSIGN_L2_QUEUE: () => assignL2(),
      UPDATE_TICKET: () => {
        state.ticketOpen = true;
        saveTicket();
      },
    };
    if (!run[op]) throw new Error(`unknown op ${op}`);
    run[op]();
    return observe();
  }
  throw new Error(`unknown action ${action}`);
}

/* canned demo */
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}
async function moveCursorTo(el) {
  if (!el || !cursorEl) return;
  cursorEl.classList.add("show");
  const r = el.getBoundingClientRect();
  cursorEl.style.transition = "left .45s ease, top .45s ease";
  cursorEl.style.left = `${r.left + r.width / 2}px`;
  cursorEl.style.top = `${r.top + 8}px`;
  await sleep(500);
}
async function clickDemo(target) {
  const el = document.querySelector(`[data-demo-target="${target}"], [data-testid="${target}"]`);
  await moveCursorTo(el);
  el?.click();
  await sleep(700);
}
async function startDemo() {
  if (state.demoRunning) return;
  state.demoRunning = true;
  resetFlow();
  await sleep(300);
  for (const op of state.path.ops) {
    state.currentOp = op;
    act("playbook_step", { op });
    await sleep(900);
  }
  state.demoRunning = false;
  cursorEl?.classList.remove("show");
  toast("Flow complete");
  render();
}

render();
if (state.autoplay) setTimeout(() => startDemo(), 500);

window.Northstar = {
  observe,
  act,
  snapshot,
  targets,
  availableActions,
  startDemo,
  resetFlow,
  procedures: () =>
    Object.values(PROCEDURES).map((p) => ({
      id: p.id,
      title: p.title,
      family: p.family,
      paths: Object.keys(p.paths),
    })),
  playbook() {
    return {
      procedure: state.procedureId,
      family: state.proc.family,
      path: state.pathId,
      caseId: state.case.customer_id,
      steps: state.path.ops.map((op, i) => ({ i: i + 1, op })),
    };
  },
  setProcedure(procedure, path, caseId) {
    return act("set_procedure", { procedure, path, caseId });
  },
};

window.dispatchEvent(new CustomEvent("northstar:ready", { detail: observe() }));
console.info("[Northstar] procedures:", Object.keys(PROCEDURES).join(", "));
