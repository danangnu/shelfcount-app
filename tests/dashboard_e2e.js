// End-to-end test of the ShelfCount dashboard with the final model.
const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');

const BASE = process.env.SHELFCOUNT_URL || 'http://localhost:8000/';
const IMG = (process.env.SHELF_IMAGES || 'datasets/grocery/ShelfImages') + '/';
const OUT = process.argv[2];
const photo = (n) => path.join(IMG, n);
const TRUE = { 'C1_P01_N1_S3_1.JPG': 15, 'C1_P05_N1_S3_1.JPG': 36, 'C2_P06_N3_S4_1.JPG': 52, 'C2_P09_N1_S7_1.JPG': 168, 'C1_P04_N1_S3_1.JPG': 39 };
const results = [];
const check = (name, ok, detail = '') => { results.push({ name, ok: !!ok, detail: String(detail) }); console.log(ok ? 'PASS' : 'FAIL', name, detail); };

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2, acceptDownloads: true });
  const page = await ctx.newPage();
  const errors = [];
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('pageerror', (e) => errors.push('pageerror: ' + e.message));

  // 1. Empty state
  await page.goto(BASE);
  await page.waitForSelector('#drop');
  check('page loads, empty state shown', await page.isVisible('#empty') && !(await page.isVisible('#last')));
  check('no stray "Load more" button or empty result card', !(await page.isVisible('#more')) && !(await page.isVisible('#progress')));
  await page.screenshot({ path: path.join(OUT, '01-empty.png') });

  // 2. Single upload via file picker
  let t0 = Date.now();
  await page.setInputFiles('#file', photo('C1_P05_N1_S3_1.JPG'));
  await page.waitForSelector('#last:not([hidden])');
  await page.waitForFunction(() => document.querySelectorAll('#history .item').length === 1);
  const dt = Date.now() - t0;
  const c1 = Number(await page.textContent('#lastCount'));
  check('single upload returns a count', c1 > 0, `count ${c1}, labelled 36, ${dt} ms from choosing the file to result on screen`);
  check('count shown next to thumbnail in history', (await page.textContent('#history .item .n')) === String(c1));
  const thumbOk = await page.evaluate(() => { const i = document.querySelector('#history .item img'); return i.complete && i.naturalWidth > 0; });
  check('thumbnail image loads', thumbOk);
  check('stats update', (await page.textContent('#sImages')) === '1' && (await page.textContent('#sProducts')) === String(c1));
  await page.waitForFunction(() => { const i = document.querySelector('#lastImg'); return i.complete && i.naturalWidth > 0; });
  await page.waitForTimeout(2700);
  await page.screenshot({ path: path.join(OUT, '02-single-result.png') });

  // 3. Batch upload (4 photos at once)
  await page.setInputFiles('#file', ['C1_P01_N1_S3_1.JPG', 'C2_P06_N3_S4_1.JPG', 'C2_P09_N1_S7_1.JPG', 'C1_P04_N1_S3_1.JPG'].map(photo));
  await page.waitForFunction(() => document.querySelector('#sImages').textContent === '5', null, { timeout: 60000 });
  const hist = await page.$$eval('#history .item', (els) => els.map((e) => ({ fn: e.querySelector('.fn').textContent, n: Number(e.querySelector('.n').textContent) })));
  check('batch of 4 adds 4 history rows', hist.length === 5, JSON.stringify(hist));
  for (const h of hist) check(`count for ${h.fn}`, Math.abs(h.n - TRUE[h.fn]) / TRUE[h.fn] <= 0.1, `counted ${h.n}, labelled ${TRUE[h.fn]}`);
  check('history is newest first', hist[0].fn === 'C1_P04_N1_S3_1.JPG', hist.map((h) => h.fn).join(', '));
  await page.waitForTimeout(2700);
  await page.waitForFunction(() => [...document.querySelectorAll('#history img')].every((i) => i.complete && i.naturalWidth > 0));
  await page.screenshot({ path: path.join(OUT, '03-history.png'), fullPage: true });

  // 4. Drag and drop
  const buf = fs.readFileSync(photo('C1_P01_N1_S3_1.JPG')).toString('base64');
  const dtHandle = await page.evaluateHandle((b64) => {
    const bin = atob(b64); const arr = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    const d = new DataTransfer(); d.items.add(new File([arr], 'dropped-shelf.jpg', { type: 'image/jpeg' })); return d;
  }, buf);
  await page.dispatchEvent('#drop', 'dragover', { dataTransfer: dtHandle });
  check('drop zone highlights on drag-over', await page.$eval('#drop', (e) => e.classList.contains('over')));
  await page.dispatchEvent('#drop', 'drop', { dataTransfer: dtHandle });
  await page.waitForFunction(() => document.querySelector('#sImages').textContent === '6', null, { timeout: 30000 });
  check('drag-and-drop upload works', (await page.textContent('#history .item .fn')) === 'dropped-shelf.jpg', `count ${await page.textContent('#history .item .n')}`);

  // 5. Viewer with overlay toggle
  await page.click('#history .item >> nth=1');
  await page.waitForSelector('#viewer[open]');
  await page.waitForFunction(() => { const i = document.querySelector('#vImg'); return i.complete && i.naturalWidth > 0; });
  check('viewer opens with box overlay', (await page.getAttribute('#vImg', 'src')).includes('/overlay'), await page.textContent('#vCount'));
  await page.screenshot({ path: path.join(OUT, '04-viewer-overlay.png') });
  await page.uncheck('#vOverlay');
  check('Boxes toggle switches to plain photo', (await page.getAttribute('#vImg', 'src')).endsWith('/image'));
  await page.check('#vOverlay');

  // 6. Delete from viewer
  page.once('dialog', (d) => d.accept());
  await page.click('#vDelete');
  await page.waitForFunction(() => document.querySelector('#sImages').textContent === '5');
  check('delete removes the result', (await page.$$('#history .item')).length === 5);

  // 7. Settings: lower confidence changes the request
  await page.click('#settingsBtn');
  await page.waitForSelector('#settings[open]');
  check('settings shows model info', (await page.textContent('#modelInfo')).includes('shelfcount.onnx'), await page.textContent('#modelInfo'));
  check('slider starts at server default 0.65', (await page.inputValue('#conf')) === '0.65', await page.inputValue('#conf'));
  await page.fill('#conf', '0.2');
  await page.screenshot({ path: path.join(OUT, '05-settings.png') });
  await page.click('#settings .btn');
  await page.setInputFiles('#file', photo('C1_P04_N1_S3_1.JPG'));
  await page.waitForFunction(() => document.querySelector('#sImages').textContent === '6', null, { timeout: 30000 });
  const lowC = Number(await page.textContent('#history .item .n'));
  const newest = (await (await page.request.get(BASE + 'api/results?limit=1')).json()).items[0];
  check('lower confidence is sent and stored with the result', newest.conf === 0.2, 'stored conf ' + newest.conf);
  check('lower confidence finds at least as many boxes', lowC >= 39, `conf 0.2 counted ${lowC}; conf 0.65 counted ${hist.find((h) => h.fn === 'C1_P04_N1_S3_1.JPG').n}`);
  await page.reload();
  await page.click('#settingsBtn');
  check('confidence setting is remembered after reload', (await page.inputValue('#conf')) === '0.2');
  await page.fill('#conf', '0.65');
  await page.click('#settings .btn');

  // 8. CSV export
  const [download] = await Promise.all([page.waitForEvent('download'), page.click('#exportBtn')]);
  const csvPath = path.join(OUT, 'export.csv');
  await download.saveAs(csvPath);
  const lines = fs.readFileSync(csvPath, 'utf8').trim().split('\n');
  check('CSV export has header + one row per result', lines.length === 7 && lines[0].startsWith('id,created_at_utc,filename,product_count'), `${lines.length} lines; file ${download.suggestedFilename()}`);

  // 9. Bad inputs
  const bad = path.join(OUT, 'not-an-image.jpg');
  fs.writeFileSync(bad, 'this is not a jpeg');
  await page.setInputFiles('#file', bad);
  await page.waitForFunction(() => document.querySelector('#toast').classList.contains('show'));
  check('corrupt image gives a clear error, no crash', (await page.textContent('#toast')).includes('Not a readable image'), await page.textContent('#toast'));
  check('corrupt image adds nothing to history', (await page.textContent('#sImages')) === '6');
  const txt = path.join(OUT, 'notes.txt');
  fs.writeFileSync(txt, 'hello');
  await page.setInputFiles('#file', txt);
  await page.waitForFunction(() => document.querySelector('#toast').textContent.includes('Please choose image files'));
  check('non-image file is refused in the browser', true, await page.textContent('#toast'));

  // 10. Keyboard: history rows and drop zone focusable
  await page.waitForTimeout(3200);
  await page.focus('#history .item');
  await page.keyboard.press('Enter');
  check('history row opens with Enter key', await page.isVisible('#viewer[open]'));
  await page.keyboard.press('Escape');

  // 10b. Paging: more than 48 results shows "Load more"
  const names = fs.readdirSync(IMG).filter((n) => /_P0[4568]_/.test(n) && n.endsWith('.JPG')).slice(0, 48);
  for (let i = 0; i < names.length; i += 12) {
    const multipart = {}; // Playwright sends one field per key, so post in sequence instead
    for (const n of names.slice(i, i + 12)) {
      await page.request.post(BASE + 'api/count', { multipart: { file: { name: n, mimeType: 'image/jpeg', buffer: fs.readFileSync(photo(n)) }, source: 'api' } });
    }
  }
  await page.reload();
  await page.waitForFunction(() => document.querySelectorAll('#history .item').length === 48);
  check('"Load more" appears when history is longer than one page', await page.isVisible('#more'), (await page.textContent('#sImages')) + ' results');
  await page.click('#more');
  await page.waitForFunction(() => document.querySelectorAll('#history .item').length === 54);
  check('"Load more" loads the rest and then hides', !(await page.isVisible('#more')));
  check('results posted through the API show an "api" badge', (await page.$$('#history .badge')).length === 48);

  check('no browser console errors', errors.filter((e) => !e.includes('415')).length === 0, errors.join(' | '));

  // 11. Phone width and dark mode screenshots
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
  const pp = await phone.newPage();
  await pp.goto(BASE);
  await pp.waitForFunction(() => document.querySelectorAll('#history .item').length > 0);
  const overflow = await pp.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  check('no sideways scroll at phone width (390 px)', overflow <= 0, `overflow ${overflow} px`);
  await pp.setInputFiles('#file', photo('C1_P05_N1_S3_1.JPG'));
  await pp.waitForSelector('#last:not([hidden])');
  await pp.waitForFunction(() => { const i = document.querySelector('#lastImg'); return i.complete && i.naturalWidth > 0; });
  await pp.waitForTimeout(2800);
  await pp.screenshot({ path: path.join(OUT, '06-phone.png') });
  const dark = await browser.newContext({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 2, colorScheme: 'dark' });
  const dp = await dark.newPage();
  await dp.goto(BASE);
  await dp.waitForFunction(() => document.querySelectorAll('#history .item').length > 0);
  await dp.waitForFunction(() => [...document.querySelectorAll('#history img')].every((i) => i.complete && i.naturalWidth > 0));
  await dp.screenshot({ path: path.join(OUT, '07-dark.png') });

  fs.writeFileSync(path.join(OUT, 'results.json'), JSON.stringify(results, null, 1));
  console.log(`\n${results.filter((r) => r.ok).length} passed, ${results.filter((r) => !r.ok).length} failed`);
  await browser.close();
})().catch((e) => { console.error('SCRIPT ERROR', e); process.exit(1); });
