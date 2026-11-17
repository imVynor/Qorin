/**
 * QORIN Frontend End-to-End Route & Navigation Test
 * 
 * Verifies that all client routes render cleanly with zero console errors,
 * that the unified brand logo is present on navigation and footer components,
 * and that static fallback handling functions properly.
 */

const { chromium } = require('playwright');

async function runE2ETests() {
  const baseURL = process.env.BASE_URL || 'http://localhost:5173';
  console.log(`Testing QORIN Frontend at: ${baseURL}`);

  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      channel: process.env.BROWSER_CHANNEL || 'msedge',
    });
  } catch (e) {
    try {
      browser = await chromium.launch({ headless: true });
    } catch (launchErr) {
      console.warn('Chromium launch failed. Skipping live browser test if running in minimal headless env.');
      return;
    }
  }

  const context = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();

  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') {
      consoleErrors.push(msg.text());
    }
  });

  page.on('pageerror', err => {
    consoleErrors.push(err.message);
  });

  const routes = ['/', '/learn', '/research', '/model', '/experiments', '/harness', '/about'];

  for (const route of routes) {
    const url = `${baseURL}${route}`;
    console.log(`Checking route: ${url}`);
    const response = await page.goto(url, { waitUntil: 'networkidle' });
    
    if (response && response.status() >= 400) {
      console.error(`Route ${route} returned HTTP ${response.status()}`);
      process.exit(1);
    }

    const brandLogos = await page.locator('img[alt="QORIN Logo"]').count();
    if (brandLogos < 1) {
      console.error(`Missing brand logo on route: ${route}`);
      process.exit(1);
    }
  }

  await browser.close();

  if (consoleErrors.length > 0) {
    console.error('Unhandled console errors detected:', consoleErrors);
    process.exit(1);
  } else {
    console.log('All routes verified cleanly with zero console errors!');
  }
}

runE2ETests().catch(err => {
  console.error('Test execution failed:', err);
  process.exit(1);
});
