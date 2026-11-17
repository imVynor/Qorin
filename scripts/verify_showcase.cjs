#!/usr/bin/env node

/**
 * QORIN Frontend Showcase — Verification & Reproducibility Script
 * 
 * Validates that all static registries, required assets, and client routes
 * are intact and properly formatted before publication or deployment.
 */

const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..');

console.log('====================================================');
console.log('QORIN Research Showcase — Reproducibility Audit');
console.log('====================================================\n');

let failed = false;

function checkFile(relPath, description) {
  const fullPath = path.join(ROOT_DIR, relPath);
  if (!fs.existsSync(fullPath)) {
    console.error(`[FAIL] Missing required file: ${relPath} (${description})`);
    failed = true;
    return false;
  }
  console.log(`[PASS] Found ${relPath}`);
  return true;
}

function checkJsonRegistry(relPath, expectedKeys = []) {
  const fullPath = path.join(ROOT_DIR, relPath);
  if (!checkFile(relPath, 'JSON Registry')) return;

  try {
    const raw = fs.readFileSync(fullPath, 'utf8');
    const data = JSON.parse(raw);
    console.log(`  -> Valid JSON (${Array.isArray(data) ? data.length + ' entries' : Object.keys(data).length + ' keys'})`);
  } catch (err) {
    console.error(`[FAIL] Could not parse JSON registry: ${relPath} - ${err.message}`);
    failed = true;
  }
}

console.log('1. Auditing Research Registries...');
checkJsonRegistry('dataset_registry.json');
checkJsonRegistry('experiment_registry.json');
checkJsonRegistry('model_registry.json');
checkJsonRegistry('references_registry.json');

console.log('\n2. Auditing Static Brand Assets & Deployment Routes...');
checkFile('web/public/qorin.svg', 'Brand SVG Favicon & Navbar Logo');
checkFile('web/public/404.html', 'GitHub Pages SPA Router Decoder');
checkFile('web/src/assets/hero.png', 'Homepage Visual Asset');
checkFile('.github/workflows/deploy-pages.yml', 'GitHub Actions Deployment Workflow');

console.log('\n3. Auditing Research Pages...');
const pages = [
  'Home.tsx',
  'Learn.tsx',
  'Research.tsx',
  'Model.tsx',
  'Experiments.tsx',
  'Harness.tsx',
  'About.tsx',
];

for (const page of pages) {
  checkFile(path.join('web/src/pages', page), `Page: ${page}`);
}

console.log('\n4. Auditing Environment Configurations...');
checkFile('web/.env.example', 'Vite Environment Template');

if (failed) {
  console.error('\n[ERROR] Reproducibility verification failed.');
  process.exit(1);
} else {
  console.log('\n====================================================');
  console.log('All frontend assets, registries, and routes VERIFIED!');
  console.log('====================================================');
}
