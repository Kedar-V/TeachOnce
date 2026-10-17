#!/usr/bin/env python3
"""One Stage-B sample: realistic Northstar SaaS console + meta + expected DAG.

Not a step-trainer. Cursor, typing, tables, forms, status chips. Ops only in meta.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.tool_registry_adapter import adapt_record, validate_registry_dag  # noqa: E402

OUT = ROOT / "data" / "generated" / "demonstrations" / "videos" / "warranty_01"
W, H, FPS = 1280, 720, 24

CASE = {
    "customer_id": "C-1042",
    "customer_name": "Ava Chen",
    "email": "ava.chen@example.com",
    "device_id": "D-2021",
    "serial": "SN-9X2K-771",
    "model": "Northstar Pro 15",
    "ticket_id": "CS-48213",
    "ticket_subject": "Laptop fails to boot — HDD click",
    "warranty_id": "W-9001",
    "warranty_status": "ACTIVE",
    "warranty_end": "2027-03-14",
    "repair_id": "SR-88412",
    "agent": "Jordan Lee",
}

NAV = ["Customers", "Tickets", "Devices", "Warranty", "Diagnostics", "Repair"]
# map scene → highlighted nav index
NAV_FOR = {
    "lookup": 0,
    "ticket": 1,
    "device": 2,
    "warranty": 3,
    "diag": 4,
    "repair": 5,
    "update": 1,
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cands = (
        [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
        ]
        if bold
        else [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )
    for p in cands:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


F12, F13, F14, F16, F18, F20, F22 = (font(s) for s in (12, 13, 14, 16, 18, 20, 22))
FB16, FB18, FB20, FB24 = (font(s, True) for s in (16, 18, 20, 24))

# palette — light SaaS
BG = "#e8edf2"
TOP = "#0f1b2d"
NAV_BG = "#152238"
NAV_ACTIVE = "#1e3a5f"
PANEL = "#ffffff"
BORDER = "#c9d4e0"
TEXT = "#1a2332"
MUTED = "#5b6b7c"
ACCENT = "#2563eb"
GREEN = "#15803d"
GREEN_BG = "#dcfce7"
RED = "#b91c1c"
ROW_HOVER = "#eff6ff"
INPUT_BG = "#f8fafc"
TOAST_BG = "#052e16"


def rr(d: ImageDraw.ImageDraw, xy, fill=None, outline=None, width=1, radius=6):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def chip(d, xy, text, fg, bg):
    x0, y0 = xy
    tw = d.textlength(text, font=F13)
    rr(d, [x0, y0, x0 + tw + 16, y0 + 22], fill=bg, radius=11)
    d.text((x0 + 8, y0 + 4), text, font=F13, fill=fg)


def draw_cursor(img: Image.Image, x: float, y: float, click: bool = False):
    d = ImageDraw.Draw(img)
    pts = [(x, y), (x, y + 18), (x + 5, y + 14), (x + 10, y + 22), (x + 13, y + 20), (x + 7, y + 12), (x + 14, y + 12)]
    d.polygon(pts, fill="#111827", outline="#ffffff")
    if click:
        d.ellipse([x - 10, y - 10, x + 10, y + 10], outline="#2563eb", width=2)


def chrome(nav_idx: int, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # top bar
    d.rectangle([0, 0, W, 48], fill=TOP)
    d.text((16, 12), "Northstar", font=FB20, fill="#ffffff")
    d.text((118, 16), "Support Console", font=F14, fill="#94a3b8")
    rr(d, [360, 10, 720, 38], fill="#1e293b", outline="#334155", radius=6)
    d.text((372, 16), "Search customers, tickets, devices…", font=F13, fill="#64748b")
    chip(d, (980, 13), "PROD", "#fef3c7", "#92400e")
    chip(d, (1040, 13), CASE["agent"], "#e2e8f0", "#334155")

    # left nav
    d.rectangle([0, 48, 200, H], fill=NAV_BG)
    for i, name in enumerate(NAV):
        y = 70 + i * 44
        if i == nav_idx:
            d.rectangle([8, y - 8, 192, y + 28], fill=NAV_ACTIVE)
            d.rectangle([8, y - 8, 12, y + 28], fill=ACCENT)
            d.text((28, y), name, font=FB16, fill="#ffffff")
        else:
            d.text((28, y), name, font=F16, fill="#94a3b8")

    # workspace header strip
    d.rectangle([200, 48, W, 88], fill=PANEL)
    d.line([200, 88, W, 88], fill=BORDER)
    d.text((220, 58), subtitle or "Workspace", font=FB18, fill=TEXT)
    return img, d


def draw_table(d, x, y, cols, rows, highlight_row: int | None = None, col_w=None):
    col_w = col_w or [140] * len(cols)
    tw = sum(col_w)
    d.rectangle([x, y, x + tw, y + 28], fill="#f1f5f9")
    d.rectangle([x, y, x + tw, y + 28 + 32 * len(rows)], outline=BORDER)
    cx = x
    for i, c in enumerate(cols):
        d.text((cx + 8, y + 7), c, font=F13, fill=MUTED)
        cx += col_w[i]
    for ri, row in enumerate(rows):
        ry = y + 28 + ri * 32
        if ri == highlight_row:
            d.rectangle([x + 1, ry, x + tw - 1, ry + 31], fill=ROW_HOVER)
        d.line([x, ry + 32, x + tw, ry + 32], fill=BORDER)
        cx = x
        for i, cell in enumerate(row):
            d.text((cx + 8, ry + 8), str(cell), font=F14, fill=TEXT)
            cx += col_w[i]


def input_box(d, x, y, w, h, value: str, placeholder: str = "", focused: bool = False):
    rr(d, [x, y, x + w, y + h], fill=INPUT_BG, outline=ACCENT if focused else BORDER, radius=4)
    text = value if value else placeholder
    color = TEXT if value else "#94a3b8"
    d.text((x + 10, y + (h - 16) // 2), text, font=F14, fill=color)
    if focused and value is not None:
        # caret
        tw = d.textlength(value, font=F14)
        cx = x + 10 + tw + 1
        d.line([cx, y + 6, cx, y + h - 6], fill=ACCENT, width=2)


def btn(d, x, y, w, h, label, primary=True):
    fill = ACCENT if primary else "#ffffff"
    outline = ACCENT if primary else BORDER
    fg = "#ffffff" if primary else TEXT
    rr(d, [x, y, x + w, y + h], fill=fill, outline=outline, radius=5)
    tw = d.textlength(label, font=F14)
    d.text((x + (w - tw) / 2, y + (h - 16) / 2), label, font=F14, fill=fg)


def toast(d, msg: str):
    tw = d.textlength(msg, font=F14)
    rr(d, [W - tw - 60, H - 56, W - 20, H - 20], fill=TOAST_BG, radius=6)
    d.text((W - tw - 44, H - 48), msg, font=F14, fill="#bbf7d0")


# ---------- scene builders: return (image, cursor_target) ----------

def scene_lookup(typed: str, show_results: bool, highlight: int | None, detail: bool) -> Image.Image:
    img, d = chrome(NAV_FOR["lookup"], "Customers — search & open record")
    d.text((220, 104), "Customer search", font=FB18, fill=TEXT)
    d.text((220, 136), "Query", font=F13, fill=MUTED)
    input_box(d, 220, 156, 360, 36, typed, "Customer ID or name…", focused=not show_results or not detail)
    btn(d, 592, 156, 100, 36, "Search", primary=True)

    if show_results:
        draw_table(
            d, 220, 220,
            ["Customer ID", "Name", "Email", "Open tickets"],
            [
                ["C-0881", "Sam Ortiz", "sam.o@example.com", "0"],
                [CASE["customer_id"], CASE["customer_name"], CASE["email"], "1"],
                ["C-2109", "Riley Park", "riley.p@example.com", "2"],
            ],
            highlight_row=highlight,
            col_w=[130, 160, 220, 120],
        )
    if detail:
        rr(d, [220, 360, 980, 620], fill=PANEL, outline=BORDER, radius=8)
        d.text((240, 376), "Customer detail", font=FB16, fill=TEXT)
        chip(d, (400, 374), "Active account", GREEN, GREEN_BG)
        for i, (k, v) in enumerate(
            [
                ("ID", CASE["customer_id"]),
                ("Name", CASE["customer_name"]),
                ("Email", CASE["email"]),
                ("Phone", "+1 (555) 014-2201"),
                ("Region", "US-East"),
            ]
        ):
            d.text((240, 420 + i * 28), f"{k}", font=F13, fill=MUTED)
            d.text((340, 420 + i * 28), v, font=F14, fill=TEXT)
        btn(d, 240, 570, 160, 34, "View open tickets", primary=True)
    return img


def scene_ticket(queue: bool, open_detail: bool, highlight: int | None = 0) -> Image.Image:
    img, d = chrome(NAV_FOR["ticket"], "Tickets — queue")
    d.text((220, 104), "Ticket queue", font=FB18, fill=TEXT)
    if queue:
        draw_table(
            d, 220, 140,
            ["Ticket", "Subject", "Priority", "Status", "Customer"],
            [
                [CASE["ticket_id"], CASE["ticket_subject"][:28] + "…", "P2", "Open", CASE["customer_id"]],
                ["CS-48101", "Screen flicker under load", "P3", "Open", "C-2109"],
                ["CS-47990", "Battery swelling report", "P1", "Pending", "C-0881"],
            ],
            highlight_row=highlight,
            col_w=[110, 280, 80, 90, 100],
        )
    if open_detail:
        rr(d, [220, 280, 1220, 660], fill=PANEL, outline=BORDER, radius=8)
        d.text((240, 296), CASE["ticket_id"], font=FB20, fill=TEXT)
        chip(d, (360, 298), "Open", "#1e40af", "#dbeafe")
        chip(d, (430, 298), "P2", "#9a3412", "#ffedd5")
        d.text((240, 340), CASE["ticket_subject"], font=F16, fill=TEXT)
        d.text((240, 372), f"Customer  {CASE['customer_id']}  ·  {CASE['customer_name']}", font=F14, fill=MUTED)
        d.text((240, 404), "Description", font=F13, fill=MUTED)
        d.text(
            (240, 426),
            "Customer reports boot failure with clicking noise from drive bay. Repro: cold start.",
            font=F14,
            fill=TEXT,
        )
        d.text((240, 470), "Linked device", font=F13, fill=MUTED)
        rr(d, [240, 492, 520, 536], fill=INPUT_BG, outline=BORDER, radius=4)
        d.text((252, 504), f"{CASE['device_id']}  ·  {CASE['model']}", font=F14, fill=TEXT)
        btn(d, 240, 560, 140, 34, "Open device", primary=True)
        btn(d, 392, 560, 120, 34, "Add note", primary=False)
    return img


def scene_device(selected: bool) -> Image.Image:
    img, d = chrome(NAV_FOR["device"], "Devices — registry")
    d.text((220, 104), "Linked devices", font=FB18, fill=TEXT)
    draw_table(
        d, 220, 140,
        ["Device ID", "Serial", "Model", "Owner", "Health"],
        [
            [CASE["device_id"], CASE["serial"], CASE["model"], CASE["customer_id"], "Degraded"],
            ["D-1888", "SN-1A9C-002", "Northstar Air 13", CASE["customer_id"], "OK"],
        ],
        highlight_row=0 if selected else None,
        col_w=[120, 150, 180, 110, 100],
    )
    if selected:
        rr(d, [220, 260, 900, 560], fill=PANEL, outline=BORDER, radius=8)
        d.text((240, 276), "Device record", font=FB16, fill=TEXT)
        chip(d, (380, 274), "Degraded", "#9a3412", "#ffedd5")
        fields = [
            ("Device ID", CASE["device_id"]),
            ("Serial", CASE["serial"]),
            ("Model", CASE["model"]),
            ("Purchase", "2024-03-14"),
            ("Owner", f"{CASE['customer_name']} ({CASE['customer_id']})"),
        ]
        for i, (k, v) in enumerate(fields):
            d.text((240, 320 + i * 30), k, font=F13, fill=MUTED)
            d.text((360, 320 + i * 30), v, font=F14, fill=TEXT)
        btn(d, 240, 490, 150, 34, "Check warranty", primary=True)
        btn(d, 404, 490, 140, 34, "Run diagnostics", primary=False)
    return img


def scene_warranty(loaded: bool) -> Image.Image:
    img, d = chrome(NAV_FOR["warranty"], "Warranty — coverage")
    d.text((220, 104), "Warranty coverage", font=FB18, fill=TEXT)
    d.text((220, 136), f"Device {CASE['device_id']}  ·  {CASE['serial']}", font=F14, fill=MUTED)
    rr(d, [220, 180, 760, 420], fill=PANEL, outline=BORDER, radius=8)
    d.text((240, 200), "Coverage panel", font=FB16, fill=TEXT)
    status = CASE.get("warranty_status", "ACTIVE")
    if loaded:
        if status == "ACTIVE":
            chip(d, (240, 240), "ACTIVE", GREEN, GREEN_BG)
            d.text((340, 242), "In warranty — covered repair eligible", font=F14, fill=GREEN)
            entitlements = "Parts + labor · advance RMA"
        else:
            chip(d, (240, 240), "EXPIRED", RED, "#fee2e2")
            d.text((340, 242), "Out of warranty — paid service only", font=F14, fill=RED)
            entitlements = "None — paid depot available"
        for i, (k, v) in enumerate(
            [
                ("Warranty ID", CASE["warranty_id"]),
                ("Plan", "Premier Hardware — 3yr"),
                ("Start", "2024-03-14"),
                ("End", CASE["warranty_end"]),
                ("Entitlements", entitlements),
            ]
        ):
            d.text((240, 290 + i * 28), k, font=F13, fill=MUTED)
            d.text((380, 290 + i * 28), v, font=F14, fill=TEXT)
    else:
        d.text((240, 260), "Loading coverage…", font=F16, fill=MUTED)
        rr(d, [240, 300, 520, 308], fill="#e2e8f0", radius=4)
        rr(d, [240, 300, 360, 308], fill=ACCENT, radius=4)
    btn(d, 220, 460, 160, 34, "Open diagnostics", primary=True)
    return img


def scene_diag(running: bool, done: bool) -> Image.Image:
    img, d = chrome(NAV_FOR["diag"], "Diagnostics — results")
    d.text((220, 104), "Hardware diagnostics", font=FB18, fill=TEXT)
    d.text((220, 136), f"Target  {CASE['device_id']}  ·  remote probe", font=F14, fill=MUTED)
    if running and not done:
        rr(d, [220, 180, 700, 280], fill=PANEL, outline=BORDER, radius=8)
        d.text((240, 210), "Running SMART + storage self-test…", font=F16, fill=TEXT)
        rr(d, [240, 250, 640, 260], fill="#e2e8f0", radius=4)
        rr(d, [240, 250, 480, 260], fill=ACCENT, radius=4)
    if done:
        draw_table(
            d, 220, 180,
            ["Component", "Test", "Result", "Severity"],
            [
                ["Storage", "SMART long", "FAIL — reallocated sectors", "Critical"],
                ["Memory", "Memtest quick", "PASS", "OK"],
                ["Battery", "Cycle health", "PASS — 92%", "OK"],
                ["Thermal", "Idle / load", "PASS", "OK"],
            ],
            highlight_row=0,
            col_w=[120, 140, 280, 100],
        )
        chip(d, (220, 360), "HDD failure", RED, "#fee2e2")
        covered = CASE.get("warranty_status", "ACTIVE") == "ACTIVE"
        action = "depot repair under warranty" if covered else "paid depot quote"
        d.text((340, 362), f"Recommended action: {action}", font=F14, fill=TEXT)
        btn(d, 220, 420, 170, 34, "Create repair order" if covered else "Record paid options", primary=True)
    return img


def scene_repair(form_filled: bool, confirmed: bool, typed_notes: str) -> Image.Image:
    img, d = chrome(NAV_FOR["repair"], "Repair — create order")
    d.text((220, 104), "Create repair order", font=FB18, fill=TEXT)
    rr(d, [220, 150, 900, 560], fill=PANEL, outline=BORDER, radius=8)
    status = CASE.get("warranty_status", "ACTIVE")
    fields = [
        ("Device", CASE["device_id"]),
        ("Ticket", CASE["ticket_id"]),
        ("Warranty", f"{CASE['warranty_id']} · {status}"),
        ("Service type", "Depot — advance RMA"),
        ("Fault code", "STOR-HDD-SMART"),
    ]
    for i, (k, v) in enumerate(fields):
        d.text((240, 176 + i * 44), k, font=F13, fill=MUTED)
        if form_filled:
            input_box(d, 400, 168 + i * 44, 420, 34, v, focused=False)
        else:
            input_box(d, 400, 168 + i * 44, 420, 34, "", placeholder="—", focused=False)
    d.text((240, 400), "Technician notes", font=F13, fill=MUTED)
    input_box(d, 400, 392, 420, 34, typed_notes, "Add notes…", focused=not confirmed and form_filled)
    if not confirmed:
        btn(d, 400, 460, 160, 36, "Submit repair", primary=True)
        btn(d, 576, 460, 100, 36, "Cancel", primary=False)
    else:
        chip(d, (400, 460), f"Created {CASE['repair_id']}", GREEN, GREEN_BG)
        d.text((560, 464), "Advance RMA label queued", font=F14, fill=MUTED)
        toast(d, f"Repair {CASE['repair_id']} created")
    return img


def scene_paid(form_filled: bool, confirmed: bool, typed_notes: str) -> Image.Image:
    img, d = chrome(NAV_FOR["repair"], "Repair — paid options")
    d.text((220, 104), "Record paid service options", font=FB18, fill=TEXT)
    rr(d, [220, 150, 900, 560], fill=PANEL, outline=BORDER, radius=8)
    fields = [
        ("Device", CASE["device_id"]),
        ("Ticket", CASE["ticket_id"]),
        ("Warranty", f"{CASE['warranty_id']} · EXPIRED"),
        ("Quote", "$349 depot + parts"),
        ("Fault code", "STOR-HDD-SMART"),
    ]
    for i, (k, v) in enumerate(fields):
        d.text((240, 176 + i * 44), k, font=F13, fill=MUTED)
        if form_filled:
            input_box(d, 400, 168 + i * 44, 420, 34, v, focused=False)
        else:
            input_box(d, 400, 168 + i * 44, 420, 34, "", placeholder="—", focused=False)
    d.text((240, 400), "Customer choice", font=F13, fill=MUTED)
    input_box(d, 400, 392, 420, 34, typed_notes, "Add notes…", focused=not confirmed and form_filled)
    if not confirmed:
        btn(d, 400, 460, 180, 36, "Save paid options", primary=True)
        btn(d, 596, 460, 100, 36, "Cancel", primary=False)
    else:
        chip(d, (400, 460), "Paid options recorded", GREEN, GREEN_BG)
        d.text((600, 464), "Awaiting customer approval", font=F14, fill=MUTED)
        toast(d, "Paid options saved")
    return img


def scene_update(editing: bool, saved: bool, note: str) -> Image.Image:
    img, d = chrome(NAV_FOR["update"], "Tickets — update status")
    d.text((220, 104), f"Update {CASE['ticket_id']}", font=FB18, fill=TEXT)
    rr(d, [220, 150, 980, 520], fill=PANEL, outline=BORDER, radius=8)
    d.text((240, 176), "Status", font=F13, fill=MUTED)
    ticket_status = (
        "Repair scheduled"
        if CASE.get("warranty_status", "ACTIVE") == "ACTIVE"
        else "Awaiting paid approval"
    )
    status = ticket_status if (editing or saved) else "Open"
    input_box(d, 400, 168, 280, 34, status, focused=editing and not saved)
    chip(d, (700, 174), status, GREEN if saved else "#1e40af", GREEN_BG if saved else "#dbeafe")
    d.text((240, 230), "Internal note", font=F13, fill=MUTED)
    input_box(
        d, 400, 222, 500, 34,
        note,
        "Add note…",
        focused=editing and not saved,
    )
    d.text((240, 290), "Linked record", font=F13, fill=MUTED)
    linked = CASE["repair_id"] if CASE.get("warranty_status", "ACTIVE") == "ACTIVE" else "Paid quote"
    d.text((400, 290), linked, font=F14, fill=TEXT)
    if not saved:
        btn(d, 400, 360, 120, 36, "Save ticket", primary=True)
    else:
        toast(d, f"Ticket saved — {ticket_status}")
        chip(d, (400, 360), "Saved", GREEN, GREEN_BG)
    return img


def lerp(a, b, t):
    return a + (b - a) * t


def ease(t):
    return t * t * (3 - 2 * t)


def move_frames(n: int, x0, y0, x1, y1):
    for i in range(n):
        t = ease((i + 1) / n)
        yield lerp(x0, x1, t), lerp(y0, y1, t), False


def hold_frames(n: int, x, y, click=False):
    for i in range(n):
        yield x, y, click and i < max(1, n // 4)


def type_frames(text: str, cps: float = 8.0):
    """Yield progressive typed strings."""
    delay = max(1, int(FPS / cps))
    for i in range(len(text) + 1):
        for _ in range(delay if i < len(text) else int(0.4 * FPS)):
            yield text[:i]


class Recorder:
    def __init__(self, td: Path):
        self.td = td
        self.i = 0
        self.events: list[dict] = []
        self.t = 0.0

    def add(self, img: Image.Image, cx: float, cy: float, click: bool = False):
        frame = img.copy()
        draw_cursor(frame, cx, cy, click=click)
        frame.save(self.td / f"frame_{self.i:05d}.png")
        self.i += 1
        self.t = self.i / FPS

    def mark_event(self, op: str, label: str, observed: dict, t0: float):
        self.events.append(
            {"t_start": round(t0, 2), "t_end": round(self.t, 2), "op": op, "label": label, "observed": observed}
        )


def encode(frames_dir: Path, mp4: Path):
    # Flat UI compresses hard — force CBR so file isn't a ~100kbps slideshow.
    subprocess.run(
        [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-framerate", str(FPS), "-i", str(frames_dir / "frame_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
            "-b:v", "2500k", "-minrate", "2000k", "-maxrate", "2500k",
            "-bufsize", "2500k", "-x264-params", "nal-hrd=cbr",
            "-r", str(FPS), str(mp4),
        ],
        check=True,
    )


def film(rec: Recorder):
    # --- LOOKUP_CUSTOMER ---
    t0 = rec.t
    img = scene_lookup("", False, None, False)
    for x, y, c in hold_frames(int(0.6 * FPS), 400, 174):
        rec.add(img, x, y, c)
    # move to search box & type
    for x, y, c in move_frames(int(0.7 * FPS), 400, 174, 300, 174):
        rec.add(img, x, y, c)
    for typed in type_frames(CASE["customer_id"], cps=7):
        img = scene_lookup(typed, False, None, False)
        rec.add(img, 300 + len(typed) * 8, 174)
    # click Search
    for x, y, c in move_frames(int(0.5 * FPS), 300, 174, 640, 174):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.35 * FPS), 640, 174, click=True):
        rec.add(img, x, y, c)
    img = scene_lookup(CASE["customer_id"], True, None, False)
    for x, y, c in hold_frames(int(0.8 * FPS), 640, 174):
        rec.add(img, x, y, c)
    # click result row
    for x, y, c in move_frames(int(0.6 * FPS), 640, 174, 420, 280):
        rec.add(img, x, y, c)
    img = scene_lookup(CASE["customer_id"], True, 1, False)
    for x, y, c in hold_frames(int(0.3 * FPS), 420, 280, click=True):
        rec.add(img, x, y, c)
    img = scene_lookup(CASE["customer_id"], True, 1, True)
    for x, y, c in hold_frames(int(2.2 * FPS), 420, 280):
        rec.add(img, x, y, c)
    rec.mark_event("LOOKUP_CUSTOMER", "Search customer", {"customer_id": CASE["customer_id"]}, t0)

    # --- GET_TICKET ---
    t0 = rec.t
    for x, y, c in move_frames(int(0.5 * FPS), 420, 280, 320, 586):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.25 * FPS), 320, 586, click=True):
        rec.add(img, x, y, c)
    img = scene_ticket(True, False, None)
    for x, y, c in hold_frames(int(0.7 * FPS), 400, 200):
        rec.add(img, x, y, c)
    for x, y, c in move_frames(int(0.5 * FPS), 400, 200, 380, 180):
        rec.add(img, x, y, c)
    img = scene_ticket(True, False, 0)
    for x, y, c in hold_frames(int(0.3 * FPS), 380, 180, click=True):
        rec.add(img, x, y, c)
    img = scene_ticket(True, True, 0)
    for x, y, c in hold_frames(int(2.4 * FPS), 380, 180):
        rec.add(img, x, y, c)
    rec.mark_event("GET_TICKET", "Open ticket", {"ticket_id": CASE["ticket_id"]}, t0)

    # --- GET_DEVICE ---
    t0 = rec.t
    for x, y, c in move_frames(int(0.5 * FPS), 380, 180, 310, 576):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.25 * FPS), 310, 576, click=True):
        rec.add(img, x, y, c)
    img = scene_device(False)
    for x, y, c in hold_frames(int(0.5 * FPS), 400, 170):
        rec.add(img, x, y, c)
    for x, y, c in move_frames(int(0.45 * FPS), 400, 170, 360, 180):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.25 * FPS), 360, 180, click=True):
        rec.add(img, x, y, c)
    img = scene_device(True)
    for x, y, c in hold_frames(int(2.3 * FPS), 360, 180):
        rec.add(img, x, y, c)
    rec.mark_event("GET_DEVICE", "Open device", {"device_id": CASE["device_id"], "serial": CASE["serial"]}, t0)

    # --- CHECK_WARRANTY ---
    t0 = rec.t
    for x, y, c in move_frames(int(0.5 * FPS), 360, 180, 310, 506):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.25 * FPS), 310, 506, click=True):
        rec.add(img, x, y, c)
    img = scene_warranty(False)
    for x, y, c in hold_frames(int(0.9 * FPS), 400, 280):
        rec.add(img, x, y, c)
    img = scene_warranty(True)
    for x, y, c in hold_frames(int(2.8 * FPS), 400, 280):
        rec.add(img, x, y, c)
    status = CASE.get("warranty_status", "ACTIVE")
    rec.mark_event(
        "CHECK_WARRANTY",
        "Check warranty",
        {"warranty_id": CASE["warranty_id"], "status": status, "end": CASE["warranty_end"]},
        t0,
    )

    # --- GET_DIAGNOSTICS ---
    t0 = rec.t
    for x, y, c in move_frames(int(0.5 * FPS), 400, 280, 300, 476):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.25 * FPS), 300, 476, click=True):
        rec.add(img, x, y, c)
    img = scene_diag(True, False)
    for x, y, c in hold_frames(int(1.2 * FPS), 400, 230):
        rec.add(img, x, y, c)
    img = scene_diag(True, True)
    for x, y, c in hold_frames(int(2.8 * FPS), 400, 230):
        rec.add(img, x, y, c)
    rec.mark_event("GET_DIAGNOSTICS", "Run diagnostics", {"device_id": CASE["device_id"], "result": "HDD failure"}, t0)

    # --- CREATE_REPAIR or RECORD_PAID_OPTIONS ---
    t0 = rec.t
    for x, y, c in move_frames(int(0.5 * FPS), 400, 230, 300, 436):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.25 * FPS), 300, 436, click=True):
        rec.add(img, x, y, c)
    scene_fn = scene_repair if status == "ACTIVE" else scene_paid
    img = scene_fn(False, False, "")
    for x, y, c in hold_frames(int(0.5 * FPS), 500, 200):
        rec.add(img, x, y, c)
    img = scene_fn(True, False, "")
    for x, y, c in hold_frames(int(0.6 * FPS), 500, 200):
        rec.add(img, x, y, c)
    note = "SMART fail — advance RMA" if status == "ACTIVE" else "Customer wants paid depot quote"
    for typed in type_frames(note, cps=10):
        img = scene_fn(True, False, typed)
        rec.add(img, 520 + len(typed) * 7, 408)
    for x, y, c in move_frames(int(0.45 * FPS), 520, 408, 470, 476):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.3 * FPS), 470, 476, click=True):
        rec.add(img, x, y, c)
    img = scene_fn(True, True, note)
    for x, y, c in hold_frames(int(2.5 * FPS), 470, 476):
        rec.add(img, x, y, c)
    if status == "ACTIVE":
        rec.mark_event("CREATE_REPAIR", "Create repair order", {"repair_id": CASE["repair_id"]}, t0)
    else:
        rec.mark_event("RECORD_PAID_OPTIONS", "Record paid options", {"ticket_id": CASE["ticket_id"]}, t0)

    # --- UPDATE_TICKET ---
    t0 = rec.t
    for x, y, c in move_frames(int(0.55 * FPS), 470, 476, 80, 130):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.25 * FPS), 80, 130, click=True):
        rec.add(img, x, y, c)
    img = scene_update(False, False, "")
    for x, y, c in hold_frames(int(0.5 * FPS), 200, 200):
        rec.add(img, x, y, c)
    for x, y, c in move_frames(int(0.4 * FPS), 200, 200, 480, 184):
        rec.add(img, x, y, c)
    img = scene_update(True, False, "")
    for x, y, c in hold_frames(int(0.4 * FPS), 480, 184):
        rec.add(img, x, y, c)
    ticket_status = "Repair scheduled" if status == "ACTIVE" else "Awaiting paid approval"
    tnote = (
        f"Linked {CASE['repair_id']}; customer notified"
        if status == "ACTIVE"
        else "Paid options sent; awaiting approval"
    )
    for typed in type_frames(tnote, cps=11):
        img = scene_update(True, False, typed)
        rec.add(img, 520 + min(len(typed), 40) * 6, 238)
    for x, y, c in move_frames(int(0.4 * FPS), 520, 238, 460, 376):
        rec.add(img, x, y, c)
    for x, y, c in hold_frames(int(0.3 * FPS), 460, 376, click=True):
        rec.add(img, x, y, c)
    img = scene_update(True, True, tnote)
    for x, y, c in hold_frames(int(2.8 * FPS), 460, 376):
        rec.add(img, x, y, c)
    rec.mark_event(
        "UPDATE_TICKET",
        "Update ticket",
        {"ticket_id": CASE["ticket_id"], "status": ticket_status},
        t0,
    )


def generate_sample(spec: dict) -> dict:
    """Render one warranty video. `spec` keys: video_id, path_type, split, case."""
    global CASE
    video_id = spec["video_id"]
    path_type = spec["path_type"]  # active | expired
    split = spec.get("split", "train")
    case = dict(spec["case"])
    case["warranty_status"] = "ACTIVE" if path_type == "active" else "EXPIRED"
    CASE = case

    out = ROOT / "data" / "generated" / "demonstrations" / "videos" / video_id
    out.mkdir(parents=True, exist_ok=True)

    ops = [
        "LOOKUP_CUSTOMER",
        "GET_TICKET",
        "GET_DEVICE",
        "CHECK_WARRANTY",
        "GET_DIAGNOSTICS",
        "CREATE_REPAIR" if path_type == "active" else "RECORD_PAID_OPTIONS",
        "UPDATE_TICKET",
    ]

    with tempfile.TemporaryDirectory() as td:
        rec = Recorder(Path(td))
        film(rec)
        mp4 = out / "video.mp4"
        encode(rec.td, mp4)
        duration = round(rec.t, 2)
        events = rec.events

    goal = (
        "Resolve laptop hardware warranty and create repair if covered"
        if path_type == "active"
        else "Resolve expired warranty case and record paid service options"
    )
    sop = (
        "Search customer → open ticket → open device → check warranty → run diagnostics → create repair → update ticket"
        if path_type == "active"
        else "Search customer → open ticket → open device → check warranty → run diagnostics → record paid options → update ticket"
    )
    meta = {
        "video_id": video_id,
        "source_dataset": "teachonce_self_synthetic",
        "generation_method": "synthetic_screen_recording",
        "family": "warranty_resolution",
        "split": split,
        "path_type": path_type,
        "duration_sec": duration,
        "fps": FPS,
        "resolution": f"{W}x{H}",
        "goal": goal,
        "sop": sop,
        "case": {
            "customer_id": case["customer_id"],
            "device_id": case["device_id"],
            "ticket_id": case["ticket_id"],
            "warranty_id": case["warranty_id"],
            "warranty_status": case["warranty_status"],
        },
        "ops": ops,
        "events": events,
        "tools_ref": "mvp_registry_v1",
        "prompt_variant": "light theme baseline",
        "notes": f"PIL synthetic SaaS UI ({path_type}).",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    adapted = adapt_record(
        "teachonce_self_synthetic",
        [{"op": o} for o in ops],
        workflow_id=f"WF-{video_id}",
        name=f"Warranty resolution ({path_type})",
    )
    if not adapted["ok"]:
        raise RuntimeError(f"{video_id}: adapter failed: {adapted['errors']}")
    verrs = validate_registry_dag(adapted["dag"])
    if verrs:
        raise RuntimeError(f"{video_id}: dag invalid: {verrs}")
    (out / "expected_dag.json").write_text(json.dumps(adapted["dag"], indent=2) + "\n")

    train_row = {
        "id": f"{split}-{video_id}",
        "family": "warranty_resolution",
        "split": split,
        "path_type": path_type,
        "source_dataset": "teachonce_self_synthetic",
        "input": {
            "goal": goal,
            "video_id": video_id,
            "video_path": str(mp4.resolve()),
            "sop": sop,
            "policies": [],
            "demonstrations": [{"video_id": video_id, "events": events}],
            "tools_ref": "mvp_registry_v1",
        },
        "target_dag": adapted["dag"],
    }
    (out / "train_row.json").write_text(json.dumps(train_row, indent=2) + "\n")

    size_mb = mp4.stat().st_size / (1024 * 1024)
    return {
        "ok": True,
        "video_id": video_id,
        "path_type": path_type,
        "split": split,
        "video": str(mp4.resolve()),
        "duration_sec": duration,
        "size_mb": round(size_mb, 2),
        "ops": ops,
    }


def main() -> None:
    result = generate_sample(
        {
            "video_id": "warranty_01",
            "path_type": "active",
            "split": "train",
            "case": {
                "customer_id": "C-1042",
                "customer_name": "Ava Chen",
                "email": "ava.chen@example.com",
                "device_id": "D-2021",
                "serial": "SN-9X2K-771",
                "model": "Northstar Pro 15",
                "ticket_id": "CS-48213",
                "ticket_subject": "Laptop fails to boot — HDD click",
                "warranty_id": "W-9001",
                "warranty_status": "ACTIVE",
                "warranty_end": "2027-03-14",
                "repair_id": "SR-88412",
                "agent": "Jordan Lee",
            },
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
