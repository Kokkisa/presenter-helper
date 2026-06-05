'use strict';

// =====================================================================
// embed.js — fill in OpenAI embeddings for every null entry in
// tier1.json and chunks.json under app.getPath('userData').
//
// Run via Electron so the userData path matches the running app:
//
//   npm run embed
//
// Behaviour:
//   - Reads settings.json to grab ca_key.
//   - Walks tier1.json, then chunks.json. For each item whose
//     `embedding` field is null/undefined, calls OpenAI's
//     text-embedding-3-small and writes the result back IMMEDIATELY
//     (the whole file is re-written atomically after every successful
//     call), so the script can be killed at any point and re-run.
//   - 300 ms delay between every API call to stay under the basic-tier
//     rate-limit pulses.
//   - 10 s AbortController timeout per call. Timeouts increment a
//     skipped counter and the loop moves on — the script never hangs.
//   - Non-timeout errors (auth, quota, network) abort the loop with a
//     clear status; re-running picks up exactly where it stopped.
//   - Progress is printed inline: `Embedding 45/1211`.
// =====================================================================

const { app } = require('electron');
const fs = require('fs').promises;
const path = require('path');

app.setName('LiveCallAssistant');

const ENDPOINT   = 'https://api.openai.com/v1/embeddings';
const MODEL      = 'text-embedding-3-small';
const DELAY_MS   = 300;
const TIMEOUT_MS = 10000;

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function readJsonOrDefault(filePath, def) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return def;
    throw err;
  }
}

async function writeJsonAtomic(filePath, data) {
  const tmp = filePath + '.tmp';
  await fs.writeFile(tmp, JSON.stringify(data), 'utf8');
  await fs.rename(tmp, filePath);
}

// Single-shot embed with a hard 10 s deadline. AbortController tears
// the fetch down properly so a stalled call doesn't keep a socket
// half-open in the background.
async function embedOne(text, apiKey) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + apiKey },
      body: JSON.stringify({ model: MODEL, input: text }),
      signal: controller.signal
    });
    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error('HTTP ' + res.status + ': ' + body.slice(0, 160));
    }
    const json = await res.json();
    return json.data[0].embedding;
  } finally {
    clearTimeout(timer);
  }
}

function isAbort(err) {
  return err && (err.name === 'AbortError' || /aborted/i.test(err.message || ''));
}

// Loop body shared between tier1 and chunks. The `extractText`
// callback abstracts the difference (Tier 1 uses .question; doc chunks
// use .text).
async function embedFile(label, filePath, extractText, apiKey) {
  const items = await readJsonOrDefault(filePath, null);
  if (!Array.isArray(items)) {
    process.stdout.write(`-- ${label}: file missing or empty — skipping.\n`);
    return;
  }
  const total   = items.length;
  const pending = items.filter(it => it && (it.embedding === null || it.embedding === undefined)).length;
  if (pending === 0) {
    process.stdout.write(`-- ${label}: all ${total} entries already embedded.\n`);
    return;
  }
  process.stdout.write(`-> ${label}: ${pending}/${total} need embedding.\n`);

  let done = 0;
  let skipped = 0;

  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    if (!it || (it.embedding !== null && it.embedding !== undefined)) continue;

    const text = (extractText(it) || '').trim();
    if (!text) {
      skipped++;
      process.stdout.write(`!  ${label} [${i + 1}/${total}] empty text — skipped.\n`);
      continue;
    }

    process.stdout.write(`   Embedding ${done + skipped + 1}/${pending}  (${label} [${i + 1}/${total}])\r`);
    try {
      const emb = await embedOne(text, apiKey);
      items[i].embedding = emb;
      // Save IMMEDIATELY — re-running this script after a kill or
      // crash will see this entry as already embedded and skip it.
      await writeJsonAtomic(filePath, items);
      done++;
    } catch (err) {
      if (isAbort(err)) {
        skipped++;
        process.stdout.write(`\n!  ${label} [${i + 1}/${total}] timed out (>${TIMEOUT_MS}ms) — skipped.\n`);
      } else {
        process.stdout.write(
          `\nX  ${label} [${i + 1}/${total}] FAILED: ${err.message}\n` +
          `   Aborting. Re-run after fixing — progress saved entry-by-entry.\n`
        );
        return;
      }
    }
    await sleep(DELAY_MS);
  }

  process.stdout.write(`\nOK ${label}: ${done} embedded, ${skipped} skipped.\n`);
}

async function main() {
  const userData = app.getPath('userData');
  process.stdout.write(`userData: ${userData}\n`);

  const settings = (await readJsonOrDefault(path.join(userData, 'settings.json'), {})) || {};
  const apiKey = (settings.ca_key || '').trim();
  if (!apiKey) {
    process.stderr.write(
      'No OpenAI key in settings.json. Run setup.js with --openai-key first:\n' +
      '  npm run setup -- --openai-key sk-...\n'
    );
    app.exit(1);
    return;
  }

  await embedFile('tier1',  path.join(userData, 'tier1.json'),  e => e.question, apiKey);
  await embedFile('chunks', path.join(userData, 'chunks.json'), e => e.text,     apiKey);

  process.stdout.write('Done.\n');
  app.exit(0);
}

app.whenReady().then(() => {
  main().catch(err => {
    process.stderr.write('FATAL: ' + (err && err.stack ? err.stack : err) + '\n');
    app.exit(1);
  });
});

app.on('window-all-closed', () => app.quit());
