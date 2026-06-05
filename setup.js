'use strict';

// =====================================================================
// setup.js — offline bootstrap for LiveCallAssistant userData.
//
// Run via the Electron binary so app.getPath('userData') matches what
// the running app uses:
//
//   npm run setup -- --openai-key sk-...
//                    [--claude-key sk-ant-...]
//                    [--qa-file path1.txt] [--qa-file path2.txt] ...
//                    [--resume path.pdf]
//
// Behaviour:
//   - Settings flags update settings.json (preserving any other keys
//     already there).
//   - Each --qa-file is parsed in the GROUP/Q/A format used by the
//     renderer's Import button and *appended* to tier1.json. Existing
//     entries are preserved; entries are written with embedding:null
//     so embed.js can fill them in later.
//   - --resume reads a PDF, chunks it with the same heuristic the
//     renderer uses (~350-word paragraph-aware chunks), and appends to
//     chunks.json. Duplicate source filenames are skipped (matches the
//     renderer's behaviour).
//
// Durability: tier1.json and chunks.json are re-written after every
// single entry, atomically (tmp + rename). If the script is killed
// half-way through, everything that was added so far is on disk.
// =====================================================================

const { app } = require('electron');
const fs = require('fs').promises;
const path = require('path');
const pdfParse = require('pdf-parse');

// Match main.js so this script writes into the same userData folder
// the running app (dev and packaged) reads from.
app.setName('LiveCallAssistant');

// ---------- CLI -----------------------------------------------------
function parseArgs(argv) {
  const out = { qaFiles: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--openai-key')      out.openaiKey = argv[++i];
    else if (a === '--claude-key') out.claudeKey = argv[++i];
    else if (a === '--qa-file')    out.qaFiles.push(argv[++i]);
    else if (a === '--resume')     out.resume = argv[++i];
    else if (a === '--help' || a === '-h') out.help = true;
  }
  return out;
}

function printHelp() {
  process.stdout.write(
    'Usage: npm run setup -- [flags]\n' +
    '\n' +
    '  --openai-key <key>     Save to settings.json as ca_key\n' +
    '  --claude-key <key>     Save to settings.json as ca_claude_key\n' +
    '  --qa-file    <path>    Append GROUP/Q/A pairs from .txt to tier1.json\n' +
    '                         (repeatable; one flag per file)\n' +
    '  --resume     <path>    Chunk the PDF and append to chunks.json\n' +
    '  --help, -h             Show this help\n'
  );
}

// ---------- Parsers (mirrored from renderer.js) ---------------------
function parseQaTxt(text) {
  const lines = String(text || '').split(/\r?\n/);
  const pairs = [];
  let currentGroup = '';
  let currentQ = null;
  let currentA = null;

  const flush = () => {
    if (currentQ !== null && currentA !== null) {
      const q = currentQ.trim();
      const a = currentA.trim();
      if (q && a) pairs.push({ question: q, answer: a, group: currentGroup });
    }
    currentQ = null;
    currentA = null;
  };

  for (const raw of lines) {
    const trimmed = raw.trim();
    if (/^GROUP\s*:/i.test(trimmed)) {
      flush();
      currentGroup = trimmed.replace(/^GROUP\s*:/i, '').trim();
    } else if (/^Q\s*:/i.test(trimmed)) {
      flush();
      currentQ = trimmed.replace(/^Q\s*:/i, '').trim();
      currentA = null;
    } else if (/^A\s*:/i.test(trimmed)) {
      if (currentQ !== null) {
        currentA = trimmed.replace(/^A\s*:/i, '').trim();
      }
    } else if (currentA !== null) {
      currentA += '\n' + raw;
    }
  }
  flush();
  return pairs;
}

function chunkText(text, filename) {
  const paragraphs = String(text || '').split(/\n\n+/).filter(p => p.trim().length > 20);
  const chunks = [];
  let current = '';
  let idx = 0;
  for (const para of paragraphs) {
    if ((current + para).split(/\s+/).length > 350) {
      if (current) {
        chunks.push({ id: `${filename}_${idx++}`, text: current.trim(), source: filename, embedding: null });
      }
      current = para;
    } else {
      current += (current ? '\n\n' : '') + para;
    }
  }
  if (current.trim()) {
    chunks.push({ id: `${filename}_${idx}`, text: current.trim(), source: filename, embedding: null });
  }
  return chunks;
}

// ---------- File I/O ------------------------------------------------
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

function todayIso() { return new Date().toISOString().slice(0, 10); }
function newId(prefix) {
  return prefix + Date.now() + '_' + Math.random().toString(36).slice(2, 6);
}

// ---------- Steps ---------------------------------------------------
async function applySettings(userData, args) {
  if (!args.openaiKey && !args.claudeKey) return;
  const filePath = path.join(userData, 'settings.json');
  const settings = (await readJsonOrDefault(filePath, {})) || {};
  if (args.openaiKey) settings.ca_key = args.openaiKey;
  if (args.claudeKey) settings.ca_claude_key = args.claudeKey;
  // Don't overwrite existing provider/threshold/model if they're set —
  // setup is meant to be re-runnable without clobbering user choices.
  if (settings.ca_provider == null)  settings.ca_provider = 'openai';
  if (settings.ca_threshold == null) settings.ca_threshold = 0.82;
  await writeJsonAtomic(filePath, settings);
  process.stdout.write(`OK settings.json updated (${Object.keys(settings).length} key${Object.keys(settings).length === 1 ? '' : 's'}).\n`);
}

async function applyQaFiles(userData, args) {
  if (args.qaFiles.length === 0) return;
  const filePath = path.join(userData, 'tier1.json');
  const tier1 = (await readJsonOrDefault(filePath, [])) || [];
  const startCount = tier1.length;
  let added = 0;

  for (const qaPath of args.qaFiles) {
    const abs = path.resolve(qaPath);
    let text;
    try {
      text = await fs.readFile(abs, 'utf8');
    } catch (err) {
      process.stdout.write(`ERR ${qaPath}: ${err.message}\n`);
      continue;
    }
    const pairs = parseQaTxt(text);
    if (pairs.length === 0) {
      process.stdout.write(`!  ${path.basename(abs)}: no GROUP/Q/A pairs found.\n`);
      continue;
    }
    process.stdout.write(`-> ${path.basename(abs)}: ${pairs.length} pair${pairs.length === 1 ? '' : 's'}\n`);

    for (const p of pairs) {
      tier1.push({
        id: newId('t1_'),
        question: p.question,
        answer: p.answer,
        embedding: null,
        source: 'import',
        group: p.group || '',
        created: todayIso()
      });
      added++;
      // Atomic save after every single entry — script kill preserves
      // everything that was added before it.
      await writeJsonAtomic(filePath, tier1);
      process.stdout.write(`   imported ${added} (total ${tier1.length})\r`);
    }
  }
  process.stdout.write(`\nOK tier1.json: +${added} entries (was ${startCount}, now ${tier1.length}).\n`);
}

async function applyResume(userData, args) {
  if (!args.resume) return;
  const abs = path.resolve(args.resume);
  let buf;
  try {
    buf = await fs.readFile(abs);
  } catch (err) {
    process.stdout.write(`ERR ${args.resume}: ${err.message}\n`);
    return;
  }
  const parsed = await pdfParse(buf);
  const filename = path.basename(abs);

  const filePath = path.join(userData, 'chunks.json');
  const chunks = (await readJsonOrDefault(filePath, [])) || [];
  if (chunks.some(c => c.source === filename)) {
    process.stdout.write(`!  ${filename}: already present in chunks.json — skipping.\n`);
    return;
  }
  const newChunks = chunkText(parsed.text || '', filename);
  process.stdout.write(`-> ${filename}: ${newChunks.length} chunk${newChunks.length === 1 ? '' : 's'} from ${parsed.numpages || 0} page${(parsed.numpages || 0) === 1 ? '' : 's'}\n`);

  let added = 0;
  for (const c of newChunks) {
    chunks.push(c);
    added++;
    await writeJsonAtomic(filePath, chunks);
    process.stdout.write(`   added ${added}/${newChunks.length}\r`);
  }
  process.stdout.write(`\nOK chunks.json: +${added} chunks (total ${chunks.length}). embed.js will fill embeddings.\n`);
}

// ---------- Main ----------------------------------------------------
async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    printHelp();
    app.exit(0);
    return;
  }
  if (!args.openaiKey && !args.claudeKey && args.qaFiles.length === 0 && !args.resume) {
    printHelp();
    app.exit(0);
    return;
  }

  const userData = app.getPath('userData');
  await fs.mkdir(userData, { recursive: true });
  process.stdout.write(`userData: ${userData}\n`);

  await applySettings(userData, args);
  await applyQaFiles(userData, args);
  await applyResume(userData, args);

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
