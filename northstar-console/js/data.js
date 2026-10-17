/** Seeded Northstar world + three procedures (distinct DAG topologies). */
export const AGENT = { name: "Jordan Lee", env: "PROD" };

/** @type {Record<string, object>} */
export const CASES = {
  "C-1042": {
    customer_id: "C-1042",
    name: "Ava Chen",
    email: "ava.chen@example.com",
    tier: "Enterprise",
    ticket_id: "CS-48213",
    ticket_subject: "Laptop storage failure — boot loops",
    ticket_priority: "P2",
    device_id: "D-2021",
    serial: "SN-9X2K-771",
    model: "Northstar Pro 15",
    purchase: "2024-03-14",
    health: "Degraded",
    alt_device: { device_id: "D-1888", serial: "SN-1A9C-002", model: "Northstar Air 13", health: "OK" },
    warranty_id: "W-9001",
    warranty_status: "ACTIVE",
    warranty_end: "2027-03-14",
    diag_result: "HDD failure",
    diag_code: "STOR-HDD-SMART",
    repair_id: "SR-88412",
    part_sku: "SSD-1TB-NS",
    part_name: "1TB NVMe SSD",
    stock: "IN_STOCK",
    shipment_id: "SH-22011",
    backorder_id: null,
    severity: "P2",
    oncall: "Riley Cho",
    l2_queue: "Hardware-L2",
  },
  "C-2109": {
    customer_id: "C-2109",
    name: "Marcus Webb",
    email: "m.webb@example.com",
    tier: "Standard",
    ticket_id: "CS-49102",
    ticket_subject: "Battery swell — out of coverage check",
    ticket_priority: "P3",
    device_id: "D-3301",
    serial: "SN-7Q4M-119",
    model: "Northstar Pro 14",
    purchase: "2022-01-08",
    health: "Degraded",
    alt_device: { device_id: "D-3302", serial: "SN-7Q4M-220", model: "Northstar Dock", health: "OK" },
    warranty_id: "W-4410",
    warranty_status: "EXPIRED",
    warranty_end: "2025-01-08",
    diag_result: "Battery swell detected",
    diag_code: "PWR-BATT-SWELL",
    repair_id: null,
    part_sku: "BATT-14-NS",
    part_name: "Pro 14 battery pack",
    stock: "OUT_OF_STOCK",
    shipment_id: null,
    backorder_id: "BO-9014",
    severity: "P3",
    oncall: "Riley Cho",
    l2_queue: "Hardware-L2",
  },
  "C-0881": {
    customer_id: "C-0881",
    name: "Priya Nair",
    email: "priya.nair@example.com",
    tier: "Enterprise",
    ticket_id: "CS-47550",
    ticket_subject: "Keyboard ghosting under warranty",
    ticket_priority: "P2",
    device_id: "D-1555",
    serial: "SN-2K8P-440",
    model: "Northstar Pro 16",
    purchase: "2024-11-02",
    health: "Degraded",
    alt_device: { device_id: "D-1556", serial: "SN-2K8P-441", model: "Northstar Air 15", health: "OK" },
    warranty_id: "W-8122",
    warranty_status: "ACTIVE",
    warranty_end: "2027-11-02",
    diag_result: "Keyboard matrix fault",
    diag_code: "INP-KB-MATRIX",
    repair_id: "SR-89001",
    part_sku: "KB-16-US",
    part_name: "Pro 16 US keyboard",
    stock: "IN_STOCK",
    shipment_id: "SH-22088",
    backorder_id: null,
    severity: "P2",
    oncall: "Riley Cho",
    l2_queue: "Hardware-L2",
  },
  "C-5010": {
    customer_id: "C-5010",
    name: "Elena Ortiz",
    email: "e.ortiz@example.com",
    tier: "Enterprise",
    ticket_id: "CS-50001",
    ticket_subject: "Production fleet — unit offline (P1)",
    ticket_priority: "P1",
    device_id: "D-9001",
    serial: "SN-P1-001",
    model: "Northstar Rack Node",
    purchase: "2025-06-01",
    health: "Down",
    alt_device: { device_id: "D-9002", serial: "SN-P1-002", model: "Northstar Rack Node", health: "OK" },
    warranty_id: "W-9900",
    warranty_status: "ACTIVE",
    warranty_end: "2028-06-01",
    diag_result: "No POST",
    diag_code: "SYS-NOPOST",
    repair_id: "SR-99010",
    part_sku: "MB-RACK-A",
    part_name: "Rack motherboard A",
    stock: "IN_STOCK",
    shipment_id: "SH-33001",
    backorder_id: null,
    severity: "P1",
    oncall: "Sam Okonkwo",
    l2_queue: "Critical-Oncall",
  },
};

/**
 * Three procedures → three DAG families for demo diversity.
 * path variants produce different branches within a family.
 */
export const PROCEDURES = {
  warranty: {
    id: "warranty",
    title: "Warranty resolution",
    family: "warranty_resolution",
    nav: [
      ["customers", "Customers"],
      ["tickets", "Tickets"],
      ["devices", "Devices"],
      ["warranty", "Warranty"],
      ["diagnostics", "Diagnostics"],
      ["repair", "Repair"],
    ],
    paths: {
      active: {
        label: "Active → repair",
        defaultCase: "C-1042",
        ops: [
          "LOOKUP_CUSTOMER",
          "GET_TICKET",
          "GET_DEVICE",
          "CHECK_WARRANTY",
          "GET_DIAGNOSTICS",
          "CREATE_REPAIR",
          "UPDATE_TICKET",
        ],
      },
      expired: {
        label: "Expired → paid",
        defaultCase: "C-2109",
        ops: [
          "LOOKUP_CUSTOMER",
          "GET_TICKET",
          "GET_DEVICE",
          "CHECK_WARRANTY",
          "GET_DIAGNOSTICS",
          "RECORD_PAID_OPTIONS",
          "UPDATE_TICKET",
        ],
      },
    },
  },
  parts: {
    id: "parts",
    title: "Spare parts fulfillment",
    family: "parts_fulfillment",
    nav: [
      ["customers", "Customers"],
      ["tickets", "Tickets"],
      ["devices", "Devices"],
      ["inventory", "Inventory"],
      ["shipping", "Shipping"],
    ],
    paths: {
      in_stock: {
        label: "In stock → ship",
        defaultCase: "C-1042",
        ops: [
          "LOOKUP_CUSTOMER",
          "GET_TICKET",
          "GET_DEVICE",
          "CHECK_PARTS_STOCK",
          "CREATE_PARTS_SHIPMENT",
          "UPDATE_TICKET",
        ],
      },
      backorder: {
        label: "Out of stock → backorder",
        defaultCase: "C-2109",
        ops: [
          "LOOKUP_CUSTOMER",
          "GET_TICKET",
          "GET_DEVICE",
          "CHECK_PARTS_STOCK",
          "CREATE_BACKORDER",
          "UPDATE_TICKET",
        ],
      },
    },
  },
  escalate: {
    id: "escalate",
    title: "Severity escalation",
    family: "severity_escalation",
    nav: [
      ["tickets", "Tickets"],
      ["customers", "Customers"],
      ["escalate", "Escalation"],
    ],
    paths: {
      p1: {
        label: "P1 → page on-call",
        defaultCase: "C-5010",
        ops: ["GET_TICKET", "LOOKUP_CUSTOMER", "ASSESS_SEVERITY", "PAGE_ONCALL", "UPDATE_TICKET"],
      },
      standard: {
        label: "P2/P3 → L2 queue",
        defaultCase: "C-0881",
        ops: ["GET_TICKET", "LOOKUP_CUSTOMER", "ASSESS_SEVERITY", "ASSIGN_L2_QUEUE", "UPDATE_TICKET"],
      },
    },
  },
};

export function resolveCase(procedureId, pathId, caseId) {
  const proc = PROCEDURES[procedureId] || PROCEDURES.warranty;
  const path = proc.paths[pathId] || Object.values(proc.paths)[0];
  if (caseId && CASES[caseId]) return { proc, path, case: CASES[caseId], pathId };
  return { proc, path, case: CASES[path.defaultCase], pathId: pathId in proc.paths ? pathId : Object.keys(proc.paths)[0] };
}

export function defaultPath(procedureId) {
  const proc = PROCEDURES[procedureId] || PROCEDURES.warranty;
  return Object.keys(proc.paths)[0];
}
