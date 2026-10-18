import { chromium } from "playwright";
import path from "path";
import { fileURLToPath } from "url";

const dir = path.dirname(fileURLToPath(import.meta.url));
const tabs = [
  "overview",
  "studio",
  "workflows",
  "executions",
  "review",
  "integrations",
];

const browser = await chromium.launch();
const context = await browser.newContext({
  viewport: { width: 1440, height: 900 },
  deviceScaleFactor: 1,
  recordVideo: { dir, size: { width: 1440, height: 900 } },
});
const page = await context.newPage();

for (const tab of tabs) {
  await page.goto(`http://localhost:8000/#/${tab}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);
  await page.screenshot({
    path: path.join(dir, `${String(tabs.indexOf(tab) + 1).padStart(2, "0")}-${tab}.png`),
    fullPage: true,
  });
}

await context.close();
await browser.close();
console.log("done");
