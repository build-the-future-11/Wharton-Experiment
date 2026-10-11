// Local artifact QA only; never connects to a user's browser or to trading services.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { pathToFileURL } = require('url');
(async () => {
  const root = path.resolve(__dirname, '..');
  const browser = await chromium.launch({ headless: true, executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1100 } });
  const errors = [];
  page.on('pageerror', e => errors.push(String(e)));
  await page.goto(pathToFileURL(path.join(root, 'reports/competition_package/DECISION_ROOM.html')).href);
  const data = JSON.parse(fs.readFileSync(path.join(root, 'runs/competition_stress/RESULTS.json')));
  let checked = 0;
  for (const r of data.results) {
    await page.selectOption('#scenario', r.scenario);
    await page.selectOption('#policy', r.policy);
    await page.selectOption('#cost', String(r.cost_bps));
    const values = await page.locator('#metrics strong').allTextContents();
    if (values[0] !== r.terminal_wealth.toFixed(4) || values[2] !== r.unpaid_at_end.toFixed(4))
      throw new Error(`Displayed evidence mismatch: ${r.scenario}/${r.policy}/${r.cost_bps}`);
    checked++;
  }
  await page.selectOption('#scenario', 'funding_failure');
  await page.selectOption('#policy', 'liability_reserve');
  await page.selectOption('#cost', '10');
  await page.screenshot({ path: path.join(root, '.local_tmp/astra/decision-room-desktop.png'), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
  await page.screenshot({ path: path.join(root, '.local_tmp/astra/decision-room-mobile.png'), fullPage: true });
  await browser.close();
  const receipt = { checked, javascript_errors: errors, mobile_overflow: overflow };
  fs.writeFileSync(path.join(root, 'audit/astra_2026-09-27/BROWSER_QA.json'), JSON.stringify(receipt, null, 2)+'\n');
  console.log(JSON.stringify(receipt));
  if (errors.length || overflow) process.exit(1);
})().catch(e => { console.error(e); process.exit(1); });
