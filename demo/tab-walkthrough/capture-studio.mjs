// Captures Teach Studio upload → analysis → compile → workflow editor for the Remotion walkthrough.
// Needs teachonce-web running (./run.sh). Resets the studio draft; compiling adds a workflow to the in-memory store.
import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";

const dir = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(dir, "../..");
const out = path.join(root, "motion-graphics-studio/public/walkthrough");
const APP = "http://localhost:8000";

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
const shot = async (name) => {
  await page.waitForTimeout(600);
  await page.screenshot({ path: path.join(out, `${name}.png`) });
  console.log("captured", name);
};

await page.request.post(`${APP}/api/studio/reset`);
await page.goto(`${APP}/#/studio`, { waitUntil: "networkidle" });
await page.addStyleTag({ content: "#toasts{display:none!important}" });

await page.click('[data-act="step"][data-id="1"]');
await page.waitForSelector('[data-act="pick-video"]');
await shot("studio-provide");

await page.setInputFiles("#vfile", path.join(root, "data/generated/demonstrations/videos/warranty_0001/video.mp4"));
await page.waitForSelector(".vidcard video");
await page.setInputFiles("#sfile", path.join(root, "demo/SOP-CS-214_Laptop_Warranty.md"));
await page.waitForSelector('[data-act="clear-sop"]');
await page.$eval(".vidcard video", (v) => new Promise((r) => { v.currentTime = 6; v.onseeked = r; }));
await shot("studio-provide-filled");

await page.click('[data-act="analyze"]');
await page.waitForSelector('text=Continue to compile', { timeout: 120000 });
await shot("studio-extract");

await page.click('[data-act="step"][data-id="3"]');
await page.waitForSelector('[data-act="compile"]');
await page.click('[data-act="compile"]');
await page.waitForTimeout(1500);
await page.screenshot({ path: path.join(out, "studio-compiling.png") });
console.log("captured studio-compiling");

await page.waitForURL(/workflows/, { timeout: 30000 });
await page.waitForLoadState("networkidle");
await page.waitForTimeout(1500);
await shot("workflow-editor");

await browser.close();
