'use strict';

// =====================================================================
// Live Call Assistant — renderer
// VAD + system-audio capture + Whisper transcription + GPT-4o streaming
// =====================================================================

// ---------- Constants ----------
// v0.3.0 — HEIGHT_PILL bumped from 64 to 100 because the Voice/Screen
// pill tab switcher is now always visible (between the header pill and
// the optional expanded panel). Must match main.js's HEIGHT_PILL.
const HEIGHT_PILL     = 100;
const HEIGHT_EXPANDED = 480;
const HEIGHT_SETTINGS = 480;

const THRESHOLD       = 0.016;   // RMS speech floor
const SILENCE_MS      = 1700;    // ms of quiet before triggering
const MIN_SPEECH_MS   = 500;     // ignore clips shorter than this
const MAX_HISTORY     = 8;       // last N messages sent to GPT
const FFT_SIZE        = 1024;

const FILLER_RE = /^(um+|uh+|hmm+|ok|okay|yeah|yes|no|right|sure|thanks|thank you|hi|hello|hey|bye|mm+|\.+|\s*)\.?$/i;

const LS_KEY = {
  apiKey:        'ca_key',
  model:         'ca_model',
  provider:      'ca_provider',
  claudeKey:     'ca_claude_key',
  groqKey:       'ca_groq_key',
  // v0.3.0 — Coding Mode / Screen capture settings
  codingLang:    'ca_coding_lang',
  autoSwitchTab: 'ca_auto_switch_tab',
  visionModel:   'ca_vision_model'
};

// v0.3.0 — Screen-capture hotkey. The authoritative value lives in main.js
// (SCREEN_CAPTURE_HOTKEY). This is purely a display string for the Settings
// row, kept in sync manually if main.js's constant ever changes.
const SCREEN_CAPTURE_HOTKEY_LABEL = 'Ctrl+Shift+S';

const MODEL_OPTIONS = {
  openai: [
    { value: 'gpt-4o',         label: 'gpt-4o (best quality)' },
    { value: 'gpt-4o-mini',    label: 'gpt-4o-mini (cheaper, faster)' },
    { value: 'gpt-3.5-turbo',  label: 'gpt-3.5-turbo (cheapest)' }
  ],
  claude: [
    { value: 'claude-sonnet-4-5', label: 'claude-sonnet-4-5 (best quality)' },
    { value: 'claude-haiku-4-5',  label: 'claude-haiku-4-5 (faster, cheaper)' }
  ]
};

// =====================================================================
// STORE — disk-backed via IPC, with an in-memory cache for sync reads.
//
// Every key the renderer ever read or wrote in v0.2.1 (api keys,
// provider/model, threshold, tier1, tier2, doc chunks) is owned by main
// and persisted as JSON in userData. localStorage is GONE — the cache
// here is the only sync surface, and it's hydrated once at boot.
//
// Reads (`storeGet`) are sync against the cache.
// Writes (`storeSet`) update the cache synchronously, then fire and
// await an IPC that performs an atomic disk write. Existing call sites
// that don't await still get the cache update for subsequent reads;
// call sites that care about durability (e.g. the embed-all loop) do
// await so the next entry only starts after the previous one is on
// disk. That's how progress survives a GPU/renderer crash.
// =====================================================================

const STORE_DEFAULT = {
  ca_key:              '',
  ca_claude_key:       '',
  ca_groq_key:         '',
  ca_provider:         'openai',
  ca_model:            '',
  ca_threshold:        0.82,
  // v0.3.0 — Coding Mode preferences (mirror main.js STORE_DEFAULT)
  ca_coding_lang:      'Python',
  ca_auto_switch_tab:  true,
  ca_vision_model:     'gpt-4o',
  ca_tier1:            [],
  ca_tier2:            [],
  ca_doc_chunks:       []
};

const __store = { ...STORE_DEFAULT };

async function loadStore() {
  for (const key of Object.keys(STORE_DEFAULT)) {
    try {
      const val = await window.electronAPI.getStore(key);
      __store[key] = (val === undefined || val === null) ? STORE_DEFAULT[key] : val;
    } catch (err) {
      console.warn('Store load failed for', key, err);
      __store[key] = STORE_DEFAULT[key];
    }
  }
}

function storeGet(key) {
  return __store[key];
}

async function storeSet(key, value) {
  __store[key] = value;
  await window.electronAPI.setStore(key, value);
}

// =====================================================================
// UTILITIES — v0.2.0 (embedding, semantic match, tiered storage)
//
// Pure-function building blocks for the three-tier answer system.
// Wired into the audio pipeline in Task 8 (RAG retrieval + tier matching).
// =====================================================================

// ── Embedding ────────────────────────────────────────────────────────
// Returns a 1536-dim Float vector from text-embedding-3-small.
// Throws on missing key or non-OK response (matches transcribe() style).
// options.signal — optional AbortSignal; lets callers impose a per-call
// timeout (used by the bulk embed-all loop in Q&A Bank).
async function embedText(text, options = {}) {
  const key = (storeGet('ca_key') || '').trim();
  if (!key) throw new Error('No OpenAI API key — open settings');

  const res = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` },
    body: JSON.stringify({ model: 'text-embedding-3-small', input: text }),
    signal: options.signal
  });

  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`Embedding ${res.status}: ${txt.slice(0, 120)}`);
  }
  const { data } = await res.json();
  return data[0].embedding;
}

// ── Cosine similarity ────────────────────────────────────────────────
// Returns 0 (not NaN) if either vector is zero-magnitude. OpenAI
// embeddings are never zero in practice but the guard keeps callers
// from getting NaN through to UI.
function cosineSimilarity(a, b) {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot   += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// ── Find best match in a tier ────────────────────────────────────────
// Scans entries, picks highest cosine score, returns entry only if
// score >= threshold; otherwise null.
function findMatch(embedding, entries, threshold) {
  let best = null;
  let bestScore = 0;
  for (const entry of entries) {
    if (!entry.embedding) continue;
    const score = cosineSimilarity(embedding, entry.embedding);
    if (score > bestScore) {
      best = entry;
      bestScore = score;
    }
  }
  return bestScore >= threshold ? best : null;
}

// ── Storage helpers ──────────────────────────────────────────────────
// Read off the in-memory cache hydrated by loadStore(). The `|| []`
// guards are belt-and-suspenders — STORE_DEFAULT already gives us empty
// arrays — but they keep call sites safe if the cache is ever cleared.
const getTier1     = () => storeGet('ca_tier1')      || [];
const getTier2     = () => storeGet('ca_tier2')      || [];
const getDocChunks = () => storeGet('ca_doc_chunks') || [];
const getThreshold = () => {
  const v = storeGet('ca_threshold');
  return typeof v === 'number' ? v : (parseFloat(v) || 0.82);
};

// ── Auto-cache Tier 3 (AI) answers into Tier 2 ──────────────────────
// Called after every successful AI generation. Embedding is supplied
// by the caller — usually the same embedding used to fail-match Tier
// 1/2, so we don't re-embed. Async so the disk write completes before
// the audio pipeline says the turn is done.
async function autoCache(transcript, embedding, answer) {
  const tier2 = getTier2();
  tier2.push({
    id: 't2_' + Date.now(),
    question: transcript,
    answer,
    embedding,
    source: 'auto',
    created: new Date().toISOString().slice(0, 10),
    reviewed: false
  });
  await storeSet('ca_tier2', tier2);
}

// ---------- DOM ----------
const $ = (sel) => document.querySelector(sel);
const dot          = $('#status-dot');
const titleState   = $('#title-state');
const muteBadge    = $('#mute-badge');
const warnBadge    = $('#warn-badge');
const btnToggle    = $('#btn-toggle');
const btnSettings  = $('#btn-settings');
const btnMin       = $('#btn-min');
const btnMax       = $('#btn-max');
const btnClose     = $('#btn-close');
const btnCopy      = $('#btn-copy');
const btnClear     = $('#btn-clear');
const btnSave      = $('#btn-save');
const sessionPanel = $('#session-panel');
const settingsPanel= $('#settings-panel');
const heardEl      = $('#heard-text');
const answerEl     = $('#answer-text');
const answerSpin   = $('#answer-spinner');
const answerBadge  = $('#answer-badge');
const btnPromote   = $('#btn-promote');
const historyList  = $('#history-list');
const apiKeyInput      = $('#api-key');
const claudeKeyInput   = $('#claude-key');
const groqKeyInput     = $('#groq-key');
const claudeKeyRow     = $('#claude-key-row');
const providerOpenAIBtn= $('#provider-openai');
const providerClaudeBtn= $('#provider-claude');
const modelSelect      = $('#model-select');
const thresholdSlider  = $('#threshold-slider');
const thresholdValue   = $('#threshold-value');
const saveStatus       = $('#save-status');

// Tab bar buttons + content panes
const tabButtons = document.querySelectorAll('.tab-btn');
const tabPanes   = document.querySelectorAll('.tab-content');

// Documents (Tab 2)
const btnUploadPdf     = $('#btn-upload-pdf');
const docStatusEl      = $('#doc-status');
const docListEl        = $('#doc-list');
const docEmptyEl       = $('#doc-empty');
const storageBarFillEl = $('#storage-bar-fill');
const storageUsageText = $('#storage-usage-text');

// Q&A Bank (Tab 3)
const qaForm           = $('#qa-form');
const qaFormTitle      = $('#qa-form-title');
const qaFormQuestion   = $('#qa-form-question');
const qaFormAnswer     = $('#qa-form-answer');
const qaFormStatus     = $('#qa-form-status');
const btnQaSave        = $('#btn-qa-save');
const btnQaCancel      = $('#btn-qa-cancel');
const btnAddTier1      = $('#btn-add-tier1');
const btnImportTier1   = $('#btn-import-tier1');
const btnEmbedAllTier1 = $('#btn-embed-all-tier1');
const btnClearTier2    = $('#btn-clear-tier2');
const tier1ListEl      = $('#tier1-list');
const tier2ListEl      = $('#tier2-list');
const tier1EmptyEl     = $('#tier1-empty');
const tier2EmptyEl     = $('#tier2-empty');
const qaBulkStatusEl   = $('#qa-bulk-status');
const qaStatsEl        = $('#qa-stats');

// v0.3.0 — Pill tab switcher + Screen panel
const pillBarEl           = $('#pill-bar');
const tabVoiceBtn         = $('#tab-voice');
const tabScreenBtn        = $('#tab-screen');
const panelVoice          = $('#panel-voice');
const panelScreen         = $('#panel-screen');
const screenStatus        = $('#screen-status');
const screenQuestionMeta  = $('#screen-question-meta');
const screenAnswerContent = $('#screen-answer-content');
// v0.3.1 — capture button + manual & feedback inputs in the Screen panel
const btnCapture          = $('#btn-capture');
const screenInputEl       = $('#screen-input');
const btnScreenInputSend  = $('#btn-screen-input-send');
const screenFeedbackEl    = $('#screen-feedback');
const btnScreenFeedbackSend = $('#btn-screen-feedback-send');
const screenFeedbackBarEl = $('#screen-feedback-bar');
// v0.3.2 — multi-screenshot queue UI: thumbnail strip + Submit/Clear
// buttons that drain or discard the pending captures.
const screenQueueBar      = $('#screen-queue-bar');
const screenThumbnails    = $('#screen-thumbnails');
const btnSubmitQueue      = $('#btn-submit-queue');
const btnClearQueue       = $('#btn-clear-queue');

// v0.3.0 — Coding Mode / Vision settings
const codingLangSelect    = $('#coding-lang-select');
const visionModelSelect   = $('#vision-model-select');
const autoSwitchToggle    = $('#auto-switch-toggle');
const hotkeyDisplay       = $('#hotkey-display');

// ---------- State ----------
const state = {
  running:      false,           // session active
  muted:        false,           // VAD paused
  settingsOpen: false,
  speechOn:     false,           // currently inside a speech segment
  silenceStart: 0,
  speechStart:  0,
  busy:         false,           // a Whisper/GPT cycle is underway
  pendingBlob:  null,            // queued audio while busy
  stream:       null,
  audioCtx:     null,
  analyser:     null,
  recorder:     null,
  bufferChunks: [],              // recorder chunks captured during speech
  rafId:        null,
  unhookMute:   null,
  convHistory:  [],              // [{role, content}]
  exchanges:    [],              // [{q, a, source, tier2Id?}] — for history list
  currentTier2Id: null,          // last Tier 2 entry id painted in the answer panel
                                 // (drives the Promote ↑ button's target)
  // v0.3.0 — Coding Mode. True between a screen-capture hotkey press and
  // user dismissing the screen panel (clearing the answer). Forces the
  // session-panel to stay expanded so the screen tab is visible.
  screenActive: false,
  screenAbort:  null,            // AbortController for in-flight vision stream
  // v0.3.2 — threaded conversation state for Coding Mode. `images` is
  // the array of (optional) screenshots committed to this session at
  // submit time; `turns` is the unified user/assistant transcript that
  // grows as the user sends follow-up corrections in the feedback bar.
  // Cleared / replaced on every Submit, manual-input send, or feedback
  // continuation that lands a new session.
  screenSession:   { images: [], turns: [] },
  // v0.3.2 — staging buffer for screenshots taken via Ctrl+Shift+S or
  // the 📸 pill button. Captures accumulate here (capped at
  // SCREENSHOT_QUEUE_MAX) without firing an API call. The Submit
  // button drains the queue into screenSession.images and runs one
  // multi-image vision turn. Clear button discards without submitting.
  screenshotQueue: []
};

// v0.3.2 — cap on queued (un-submitted) screenshots. Keeps total
// payload size reasonable and prevents accidental runaway via held
// hotkey.
const SCREENSHOT_QUEUE_MAX = 10;

// ---------- Settings (disk-backed via store) ----------
function updateModelOptions(provider) {
  const opts = MODEL_OPTIONS[provider] || MODEL_OPTIONS.openai;
  const current = modelSelect.value;
  modelSelect.innerHTML = '';
  for (const o of opts) {
    const el = document.createElement('option');
    el.value = o.value;
    el.textContent = o.label;
    modelSelect.appendChild(el);
  }
  // Preserve current selection if still valid for this provider; otherwise
  // default to the first option (i.e. the recommended model for the provider).
  if (opts.some(o => o.value === current)) {
    modelSelect.value = current;
  } else {
    modelSelect.value = opts[0].value;
  }
}

// Synchronous on the UI side — storeSet updates the cache immediately so
// any subsequent storeGet returns the new value. The disk write fires
// in the background via IPC. Callers that need disk durability before
// proceeding should `await setProvider(...)`.
function setProvider(provider) {
  const p = (provider === 'claude') ? 'claude' : 'openai';
  storeSet(LS_KEY.provider, p);  // fire-and-forget; cache updates sync
  providerOpenAIBtn.classList.toggle('active', p === 'openai');
  providerClaudeBtn.classList.toggle('active', p === 'claude');
  claudeKeyRow.style.display = (p === 'claude') ? '' : 'none';
  updateModelOptions(p);
  // Persist the (possibly reset) model so saveSettings doesn't have to.
  storeSet(LS_KEY.model, modelSelect.value);
}

function loadSettings() {
  apiKeyInput.value    = storeGet(LS_KEY.apiKey)    || '';
  claudeKeyInput.value = storeGet(LS_KEY.claudeKey) || '';
  groqKeyInput.value   = storeGet(LS_KEY.groqKey)   || '';
  const provider       = storeGet(LS_KEY.provider)  || 'openai';
  setProvider(provider);  // populates model options + applies UI state
  const savedModel     = storeGet(LS_KEY.model);
  if (savedModel && MODEL_OPTIONS[provider].some(o => o.value === savedModel)) {
    modelSelect.value = savedModel;
  }
  // Threshold slider — bound on both edges; getThreshold() handles default.
  const t = getThreshold();
  thresholdSlider.value = t;
  thresholdValue.textContent = t.toFixed(2);

  // v0.3.0 — Coding Mode settings. Each dropdown defaults to STORE_DEFAULT
  // if the saved value isn't in the option list.
  const cl = storeGet(LS_KEY.codingLang) || 'Python';
  if ([...codingLangSelect.options].some(o => o.value === cl)) {
    codingLangSelect.value = cl;
  }
  const vm = storeGet(LS_KEY.visionModel) || 'gpt-4o';
  if ([...visionModelSelect.options].some(o => o.value === vm)) {
    visionModelSelect.value = vm;
  }
  const ast = storeGet(LS_KEY.autoSwitchTab);
  autoSwitchToggle.checked = (ast === undefined || ast === null) ? true : !!ast;
  hotkeyDisplay.value = SCREEN_CAPTURE_HOTKEY_LABEL;
}

async function saveSettings() {
  // Sequential awaits — the IPC chain in main serializes writes anyway,
  // but doing them in order here keeps the saved-status spinner honest.
  await storeSet(LS_KEY.apiKey,    apiKeyInput.value.trim());
  await storeSet(LS_KEY.claudeKey, claudeKeyInput.value.trim());
  await storeSet(LS_KEY.groqKey,   groqKeyInput.value.trim());
  await storeSet(LS_KEY.model,     modelSelect.value);
  await storeSet('ca_threshold',   parseFloat(thresholdSlider.value));
  // v0.3.0 — Coding Mode settings
  await storeSet(LS_KEY.codingLang,    codingLangSelect.value);
  await storeSet(LS_KEY.visionModel,   visionModelSelect.value);
  await storeSet(LS_KEY.autoSwitchTab, !!autoSwitchToggle.checked);
  saveStatus.textContent = 'Saved.';
  setTimeout(() => { saveStatus.textContent = ''; }, 1500);
}

// Tab switching — visual only; tabs share the same settings panel container
// and the same boot-time loadSettings(), so no per-tab data loading needed.
function setActiveTab(name) {
  for (const btn of tabButtons) {
    btn.classList.toggle('active', btn.dataset.tab === name);
  }
  for (const pane of tabPanes) {
    pane.classList.toggle('active', pane.id === `tab-${name}`);
  }
}

// =====================================================================
// Q&A BANK — Tier 1 + Tier 2 storage + UI (Tab 3)
// =====================================================================

// Tier writers (the getters live up in the UTILITIES section). Both
// return Promises now; await them when ordering against subsequent
// reads matters (the embed-all loop does; quick UI mutations don't).
const setTier1 = (entries) => storeSet('ca_tier1', entries);
const setTier2 = (entries) => storeSet('ca_tier2', entries);

function newTier1Id() { return 't1_' + Date.now() + '_' + Math.random().toString(36).slice(2, 6); }
function todayIso()   { return new Date().toISOString().slice(0, 10); }

// ---- form state ----
// qaFormMode: 'add-tier1' | 'edit-tier1' | 'promote-tier2' | null
// qaFormEditingId: id of the entry being edited / promoted (or null for add)
// qaFormOriginal: { question, embedding } captured at form open — lets save
//                  preserve the embedding when the question text is unchanged.
let qaFormMode = null;
let qaFormEditingId = null;
let qaFormOriginal = null;

function showQaForm(visible) {
  qaForm.style.display = visible ? '' : 'none';
}

function setQaFormStatus(msg) {
  qaFormStatus.textContent = msg || '';
}

function startAddTier1() {
  qaFormMode = 'add-tier1';
  qaFormEditingId = null;
  qaFormOriginal = null;
  qaFormTitle.textContent = 'Add Tier 1 pair';
  qaFormQuestion.value = '';
  qaFormAnswer.value = '';
  setQaFormStatus('');
  showQaForm(true);
  qaFormQuestion.focus();
}

function startEditTier1(id) {
  const entry = getTier1().find(e => e.id === id);
  if (!entry) return;
  qaFormMode = 'edit-tier1';
  qaFormEditingId = id;
  qaFormOriginal = { question: entry.question, embedding: entry.embedding };
  qaFormTitle.textContent = 'Edit Tier 1 pair';
  qaFormQuestion.value = entry.question;
  qaFormAnswer.value = entry.answer;
  setQaFormStatus('');
  showQaForm(true);
  qaFormQuestion.focus();
}

function startPromoteTier2(id) {
  const entry = getTier2().find(e => e.id === id);
  if (!entry) return;
  qaFormMode = 'promote-tier2';
  qaFormEditingId = id;
  qaFormOriginal = { question: entry.question, embedding: entry.embedding };
  qaFormTitle.textContent = 'Promote to Tier 1 — rewrite in your own voice';
  qaFormQuestion.value = entry.question;
  qaFormAnswer.value = entry.answer;
  setQaFormStatus('');
  showQaForm(true);
  qaFormAnswer.focus();  // promote = rewrite answer, so focus there
}

function cancelQaForm() {
  qaFormMode = null;
  qaFormEditingId = null;
  qaFormOriginal = null;
  showQaForm(false);
  setQaFormStatus('');
}

function saveQaForm() {
  const q = qaFormQuestion.value.trim();
  const a = qaFormAnswer.value.trim();
  if (!q || !a) {
    setQaFormStatus('Both question and answer are required.');
    return;
  }

  if (qaFormMode === 'add-tier1') {
    const tier1 = getTier1();
    tier1.push({
      id: newTier1Id(),
      question: q,
      answer: a,
      embedding: null,
      source: 'manual',
      created: todayIso()
    });
    setTier1(tier1);

  } else if (qaFormMode === 'edit-tier1') {
    const tier1 = getTier1();
    const idx = tier1.findIndex(e => e.id === qaFormEditingId);
    if (idx >= 0) {
      const entry = tier1[idx];
      // If the question changed, the existing embedding no longer matches.
      const qChanged = qaFormOriginal && qaFormOriginal.question !== q;
      tier1[idx] = {
        ...entry,
        question: q,
        answer: a,
        embedding: qChanged ? null : entry.embedding
      };
      setTier1(tier1);
    }

  } else if (qaFormMode === 'promote-tier2') {
    const tier1 = getTier1();
    const tier2 = getTier2();
    // Reuse the Tier 2 embedding if the question wasn't edited; otherwise
    // clear it — Tier 1 has its own per-entry Embed button.
    const qUnchanged = qaFormOriginal && qaFormOriginal.question === q;
    tier1.push({
      id: newTier1Id(),
      question: q,
      answer: a,
      embedding: qUnchanged ? qaFormOriginal.embedding : null,
      source: 'manual',
      created: todayIso()
    });
    setTier1(tier1);
    setTier2(tier2.filter(e => e.id !== qaFormEditingId));
  }

  cancelQaForm();
  renderQaBank();
}

// ---- Embed (Tier 1 only — Tier 2 entries are embedded by autoCache) ----
async function embedTier1Entry(id) {
  const tier1 = getTier1();
  const idx = tier1.findIndex(e => e.id === id);
  if (idx < 0) return;

  // Optimistic UI update — paint 'Embedding...' on the matching row.
  const itemEl = tier1ListEl.querySelector(`.qa-item[data-id="${id}"]`);
  const statusEl = itemEl && itemEl.querySelector('.qa-item-status');
  if (statusEl) {
    statusEl.textContent = 'Embedding...';
    statusEl.className = 'qa-item-status embedding';
  }

  try {
    const embedding = await embedText(tier1[idx].question);
    // Re-read in case the array changed during the await.
    const fresh = getTier1();
    const j = fresh.findIndex(e => e.id === id);
    if (j >= 0) {
      fresh[j].embedding = embedding;
      setTier1(fresh);
    }
    renderQaBank();
  } catch (err) {
    console.error('Embed failed:', err);
    if (statusEl) {
      statusEl.textContent = 'Embed failed: ' + (err.message || 'unknown error');
      statusEl.className = 'qa-item-status error';
    }
  }
}

// ---- Delete / toggle reviewed / clear all ----
function deleteTier1Entry(id) {
  if (!confirm('Delete this Tier 1 pair?')) return;
  setTier1(getTier1().filter(e => e.id !== id));
  renderQaBank();
}

function deleteTier2Entry(id) {
  if (!confirm('Delete this cached answer?')) return;
  setTier2(getTier2().filter(e => e.id !== id));
  renderQaBank();
}

function toggleTier2Reviewed(id) {
  const tier2 = getTier2();
  const entry = tier2.find(e => e.id === id);
  if (!entry) return;
  entry.reviewed = !entry.reviewed;
  setTier2(tier2);
  renderQaBank();
}

function clearTier2() {
  if (!confirm('Clear all auto-cached answers? This cannot be undone.')) return;
  setTier2([]);
  renderQaBank();
}

// ---- Bulk import + bulk embed (Tier 1) -------------------------------
// Parse format:
//   GROUP: <name>          (optional; applies to all following Q/A until
//                           the next GROUP line; defaults to '')
//   Q: <question>
//   A: <answer>            (continuation lines until next Q/GROUP/EOF
//                           become part of the answer)
// Lines outside a Q/A block are ignored. Empty Q or A drops the pair.
function parseQaTxt(text) {
  const lines = String(text || '').split(/\r?\n/);
  const pairs = [];
  let currentGroup = '';
  let currentQ = null;
  let currentA = null;  // null = not yet collecting; string = collecting

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
      // Continuation of the current answer — preserve as a new line.
      currentA += '\n' + raw;
    }
  }
  flush();
  return pairs;
}

function setQaBulkStatus(msg) {
  if (msg) {
    qaBulkStatusEl.textContent = msg;
    qaBulkStatusEl.style.display = '';
  } else {
    qaBulkStatusEl.style.display = 'none';
    qaBulkStatusEl.textContent = '';
  }
}

async function importTier1FromTxt() {
  if (!window.electronAPI || !window.electronAPI.openTxtDialog || !window.electronAPI.readTxt) {
    setQaBulkStatus('Import unavailable — Electron IPC not ready.');
    return;
  }

  const filePath = await window.electronAPI.openTxtDialog();
  if (!filePath) return;

  setQaBulkStatus('Reading file...');
  const result = await window.electronAPI.readTxt(filePath);
  if (result.error) {
    setQaBulkStatus(`Failed to read file: ${result.error}`);
    return;
  }

  const pairs = parseQaTxt(result.text || '');
  if (pairs.length === 0) {
    setQaBulkStatus(
      `No Q&A pairs found in ${result.filename}. Expected GROUP/Q/A format.`
    );
    return;
  }

  const ok = confirm(
    `Import ${pairs.length} Q&A pair${pairs.length === 1 ? '' : 's'} ` +
    `from ${result.filename} into Tier 1?`
  );
  if (!ok) {
    setQaBulkStatus('');
    return;
  }

  const tier1 = getTier1();
  for (const p of pairs) {
    tier1.push({
      id: newTier1Id(),
      question: p.question,
      answer: p.answer,
      embedding: null,
      source: 'import',
      group: p.group || '',
      created: todayIso()
    });
  }
  setTier1(tier1);
  renderQaBank();

  setQaBulkStatus(
    `Imported ${pairs.length} pair${pairs.length === 1 ? '' : 's'} — ` +
    `click "Embed all unembedded" to make them searchable.`
  );
}

// ---- Tunables for the bulk embed loop -------------------------------
// 300 ms between calls keeps us under OpenAI's rate-limit pulses for
// text-embedding-3-small on the basic API tier (~3 RPS sustained is
// safe and well below the burst ceiling). 10 s is a generous per-call
// ceiling — a healthy embedding round-trip is well under 1 s, so a
// timeout this large only fires on real network stalls. Batch size of
// 50 controls UX cadence; disk writes happen per-entry inside the loop
// via setTier1 -> storeSet -> set-store IPC, so progress is durable on
// disk even if the renderer is closed or crashes mid-batch.
const EMBED_BATCH_SIZE = 50;
const EMBED_DELAY_MS   = 300;
const EMBED_TIMEOUT_MS = 10000;

// Sequential + paced + per-call timeout. Resumable: any Tier 1 entry
// whose embedding is still null after this run can be retried by
// clicking the button again — storage is updated after every successful
// call.
//
// Error policy:
// - Timeout (10s, AbortError) → skip that entry, increment skipped,
//   continue. This is what the "never freeze" requirement looks like.
// - Other errors (auth, quota, network failure) → abort the whole loop
//   with a resumable status, so the user can fix the underlying issue
//   and click again. Retrying every entry against a broken key would
//   just burn time and produce 200 identical errors.
async function embedAllTier1Unembedded() {
  const targets = getTier1().filter(e => !e.embedding);
  if (targets.length === 0) {
    setQaBulkStatus('All Tier 1 entries are already embedded.');
    setTimeout(() => setQaBulkStatus(''), 2000);
    return;
  }

  btnImportTier1.disabled = true;
  btnEmbedAllTier1.disabled = true;
  btnAddTier1.disabled = true;

  const total = targets.length;
  const totalBatches = Math.ceil(total / EMBED_BATCH_SIZE);
  let done = 0;
  let skipped = 0;

  try {
    for (let batchIdx = 0; batchIdx < totalBatches; batchIdx++) {
      const batchStart = batchIdx * EMBED_BATCH_SIZE;
      const batchEnd   = Math.min(batchStart + EMBED_BATCH_SIZE, total);
      const batch      = targets.slice(batchStart, batchEnd);

      for (let i = 0; i < batch.length; i++) {
        const target  = batch[i];
        const overall = batchStart + i + 1;
        setQaBulkStatus(
          `Batch ${batchIdx + 1}/${totalBatches} — embedding ${overall}/${total}` +
          (skipped > 0 ? ` · ${skipped} skipped` : '')
        );

        // AbortController gives us a hard cancel that actually tears down
        // the fetch (and releases the socket) instead of letting a hung
        // request linger in the background.
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), EMBED_TIMEOUT_MS);

        try {
          const emb = await embedText(target.question, { signal: controller.signal });
          clearTimeout(timeoutId);
          const cur = getTier1();
          const idx = cur.findIndex(e => e.id === target.id);
          if (idx >= 0) {
            cur[idx].embedding = emb;
            // Await — guarantees this entry is on disk before the next
            // call starts, so a renderer crash or kill mid-batch never
            // loses a successful embed.
            await setTier1(cur);
          }
          done++;
        } catch (err) {
          clearTimeout(timeoutId);
          // AbortError = our 10s timeout. Skip and continue, never freeze.
          // Some Electron versions surface the abort as a TypeError with
          // 'aborted' in the message, so check both.
          const aborted = err && (err.name === 'AbortError' || /aborted/i.test(err.message || ''));
          if (aborted) {
            skipped++;
            console.warn(
              `Embed timeout on entry ${target.id} (${overall}/${total}) — skipped.`
            );
          } else {
            setQaBulkStatus(
              `Embed failed at ${overall}/${total}: ${err.message}. ` +
              `Click again to resume.`
            );
            renderQaBank();
            return;
          }
        }

        // Pace within the batch — last call in a batch doesn't sleep
        // because the batch-boundary block below handles that.
        if (i < batch.length - 1) {
          await sleep(EMBED_DELAY_MS);
        }
      }

      // Batch boundary: refresh the per-row indicators in the visible
      // Tier 1 list and announce the batch milestone.
      renderQaBank();
      if (batchIdx < totalBatches - 1) {
        setQaBulkStatus(
          `Batch ${batchIdx + 1}/${totalBatches} done — ` +
          `${done} embedded · ${skipped} skipped. Continuing...`
        );
        await sleep(EMBED_DELAY_MS);
      }
    }

    const tail = skipped > 0
      ? ` · ${skipped} skipped (click again to retry)`
      : '';
    setQaBulkStatus(`Embedded ${done}/${total} entries${tail}.`);
    setTimeout(() => setQaBulkStatus(''), 4000);
  } finally {
    btnImportTier1.disabled = false;
    btnAddTier1.disabled = false;
    // btnEmbedAllTier1's disabled state is re-derived by renderTier1().
    renderQaBank();
  }
}

// ---- Rendering ----
function renderQaBank() {
  renderTier1();
  renderTier2();
  renderQaStats();
}

function renderTier1() {
  const entries = getTier1();
  tier1ListEl.innerHTML = '';
  for (const entry of entries) {
    tier1ListEl.appendChild(buildTier1Item(entry));
  }
  tier1EmptyEl.style.display = entries.length === 0 ? '' : 'none';

  const pending = entries.filter(e => !e.embedding).length;
  btnEmbedAllTier1.disabled = pending === 0;
  btnEmbedAllTier1.textContent = pending > 0
    ? `Embed all unembedded (${pending})`
    : 'Embed all unembedded';
}

function renderTier2() {
  const entries = getTier2();
  tier2ListEl.innerHTML = '';
  // Most recent first — cache builds up over time so newest is most relevant.
  for (const entry of [...entries].reverse()) {
    tier2ListEl.appendChild(buildTier2Item(entry));
  }
  tier2EmptyEl.style.display = entries.length === 0 ? '' : 'none';
}

function renderQaStats() {
  const t1 = getTier1().length;
  const t2 = getTier2().length;
  qaStatsEl.textContent = `Tier 1: ${t1} pair${t1 === 1 ? '' : 's'} · Tier 2: ${t2} pair${t2 === 1 ? '' : 's'}`;
}

function buildTier1Item(entry) {
  const root = document.createElement('div');
  root.className = 'qa-item';
  root.dataset.id = entry.id;

  const q = document.createElement('div');
  q.className = 'qa-item-q';
  q.textContent = entry.question;
  root.appendChild(q);

  const a = document.createElement('div');
  a.className = 'qa-item-a';
  a.textContent = entry.answer;
  root.appendChild(a);

  const meta = document.createElement('div');
  meta.className = 'qa-item-meta';
  const status = document.createElement('span');
  status.className = 'qa-item-status ' + (entry.embedding ? 'embedded' : 'unembedded');
  status.textContent = entry.embedding ? 'Embedded' : 'Not embedded';
  meta.appendChild(status);
  const date = document.createElement('span');
  date.textContent = entry.created || '';
  meta.appendChild(date);
  root.appendChild(meta);

  const actions = document.createElement('div');
  actions.className = 'qa-item-actions';

  const embedBtn = document.createElement('button');
  embedBtn.type = 'button';
  embedBtn.className = 'btn-mini';
  embedBtn.dataset.action = 'embed';
  embedBtn.textContent = entry.embedding ? 'Re-embed' : 'Embed';
  actions.appendChild(embedBtn);

  const editBtn = document.createElement('button');
  editBtn.type = 'button';
  editBtn.className = 'btn-mini';
  editBtn.dataset.action = 'edit';
  editBtn.textContent = 'Edit';
  actions.appendChild(editBtn);

  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'btn-mini btn-danger';
  delBtn.dataset.action = 'delete';
  delBtn.textContent = 'Delete';
  actions.appendChild(delBtn);

  root.appendChild(actions);
  return root;
}

function buildTier2Item(entry) {
  const root = document.createElement('div');
  root.className = 'qa-item tier2';
  root.dataset.id = entry.id;

  const q = document.createElement('div');
  q.className = 'qa-item-q';
  q.textContent = entry.question;
  root.appendChild(q);

  const a = document.createElement('div');
  a.className = 'qa-item-a';
  a.textContent = entry.answer;
  root.appendChild(a);

  const meta = document.createElement('div');
  meta.className = 'qa-item-meta';
  if (entry.reviewed) {
    const badge = document.createElement('span');
    badge.className = 'qa-item-badge reviewed';
    badge.textContent = 'Reviewed';
    meta.appendChild(badge);
  }
  const status = document.createElement('span');
  status.className = 'qa-item-status ' + (entry.embedding ? 'embedded' : 'unembedded');
  status.textContent = entry.embedding ? 'Embedded' : 'No embedding';
  meta.appendChild(status);
  const date = document.createElement('span');
  date.textContent = entry.created || '';
  meta.appendChild(date);
  root.appendChild(meta);

  const actions = document.createElement('div');
  actions.className = 'qa-item-actions';

  const promoteBtn = document.createElement('button');
  promoteBtn.type = 'button';
  promoteBtn.className = 'btn-mini btn-promote';
  promoteBtn.dataset.action = 'promote';
  promoteBtn.textContent = '↑ Promote';
  actions.appendChild(promoteBtn);

  const reviewBtn = document.createElement('button');
  reviewBtn.type = 'button';
  reviewBtn.className = 'btn-mini';
  reviewBtn.dataset.action = 'review';
  reviewBtn.textContent = entry.reviewed ? 'Unmark' : 'Mark reviewed';
  actions.appendChild(reviewBtn);

  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'btn-mini btn-danger';
  delBtn.dataset.action = 'delete';
  delBtn.textContent = 'Delete';
  actions.appendChild(delBtn);

  root.appendChild(actions);
  return root;
}

// =====================================================================
// DOCUMENTS — PDF upload + chunking + embedding (Tab 2)
//
// Renderer can't `require('pdf-parse')` (contextIsolation=true).
// File dialog + PDF parsing happen in main; renderer chunks the text
// and embeds via existing embedText().
// =====================================================================

const DOC_STORAGE_WARN_MB = 3;
const DOC_STORAGE_MAX_MB  = 4.5;

const setDocChunks = (chunks) => storeSet('ca_doc_chunks', chunks);

// Chunker per CLAUDE.md spec snippet. Paragraph-aware (splits on \n\n+),
// groups paragraphs into ~400-token chunks (heuristic: > 350 words flushes).
// No overlap — the spec's strategy text mentions 50-token overlap, but the
// spec's code snippet doesn't implement it, and for paragraph-coherent
// personal-doc chunks the overlap costs more storage than it gains in recall.
function chunkText(text, filename) {
  const paragraphs = text.split(/\n\n+/).filter(p => p.trim().length > 20);
  const chunks = [];
  let current = '';
  let idx = 0;
  for (const para of paragraphs) {
    if ((current + para).split(/\s+/).length > 350) {
      if (current) {
        chunks.push({
          id: `${filename}_${idx++}`,
          text: current.trim(),
          source: filename,
          embedding: null
        });
      }
      current = para;
    } else {
      current += (current ? '\n\n' : '') + para;
    }
  }
  if (current.trim()) {
    chunks.push({
      id: `${filename}_${idx}`,
      text: current.trim(),
      source: filename,
      embedding: null
    });
  }
  return chunks;
}

function storageBytes(chunks) {
  return new Blob([JSON.stringify(chunks || getDocChunks())]).size;
}

function formatBytes(bytes) {
  if (bytes < 1024)               return `${bytes} B`;
  if (bytes < 1024 * 1024)        return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function setDocStatus(msg) {
  if (msg) {
    docStatusEl.textContent = msg;
    docStatusEl.style.display = '';
  } else {
    docStatusEl.style.display = 'none';
    docStatusEl.textContent = '';
  }
}

async function uploadPdfs() {
  if (!window.electronAPI || !window.electronAPI.openPdfDialog) {
    setDocStatus('Upload unavailable — Electron IPC not ready.');
    return;
  }

  const paths = await window.electronAPI.openPdfDialog();
  if (!paths || paths.length === 0) return;

  btnUploadPdf.disabled = true;
  try {
    for (const filePath of paths) {
      await uploadOnePdf(filePath);
    }
    setDocStatus('Upload complete.');
    setTimeout(() => setDocStatus(''), 2500);
  } finally {
    btnUploadPdf.disabled = false;
  }
}

async function uploadOnePdf(filePath) {
  setDocStatus(`Parsing ${filePathBasename(filePath)}...`);
  const parsed = await window.electronAPI.parsePdf(filePath);

  if (parsed.error) {
    setDocStatus(`Failed to parse ${filePathBasename(filePath)}: ${parsed.error}`);
    await sleep(2500);
    return;
  }
  if (!parsed.text || parsed.text.trim().length < 40) {
    setDocStatus(`No usable text extracted from ${parsed.filename}.`);
    await sleep(2500);
    return;
  }

  // Reject duplicate filenames — keeps chunk ids unique and avoids
  // ambiguous file cards.
  const existing = getDocChunks();
  if (existing.some(c => c.source === parsed.filename)) {
    setDocStatus(`A file named ${parsed.filename} is already uploaded. Delete it first.`);
    await sleep(3000);
    return;
  }

  const newChunks = chunkText(parsed.text, parsed.filename);
  if (newChunks.length === 0) {
    setDocStatus(`No chunks produced from ${parsed.filename}.`);
    await sleep(2500);
    return;
  }

  // Hard storage cap — measure prospective size BEFORE committing.
  const prospective = storageBytes(existing.concat(newChunks));
  if (prospective > DOC_STORAGE_MAX_MB * 1024 * 1024) {
    setDocStatus(
      `Aborted — ${parsed.filename} would push storage past ${DOC_STORAGE_MAX_MB} MB.`
    );
    await sleep(4000);
    return;
  }

  // Save chunks (without embeddings yet) so partial progress survives.
  setDocChunks(existing.concat(newChunks));
  renderDocList();
  renderStorageBar();

  // Embed sequentially. Save after each so a mid-flight failure leaves
  // the user with the chunks that did succeed.
  setDocStatus(`Embedding ${parsed.filename} (0/${newChunks.length})...`);
  let embedded = 0;
  for (let i = 0; i < newChunks.length; i++) {
    try {
      const emb = await embedText(newChunks[i].text);
      const cur = getDocChunks();
      const idx = cur.findIndex(c => c.id === newChunks[i].id);
      if (idx >= 0) {
        cur[idx].embedding = emb;
        setDocChunks(cur);
      }
      embedded++;
      setDocStatus(`Embedding ${parsed.filename} (${embedded}/${newChunks.length})...`);
      renderDocList();
      renderStorageBar();
    } catch (err) {
      setDocStatus(
        `Embedding failed at ${embedded + 1}/${newChunks.length}: ${err.message}. ` +
        `Use Re-embed when ready.`
      );
      await sleep(4000);
      return;
    }
  }
}

async function reEmbedFile(filename) {
  const chunks = getDocChunks();
  const targets = chunks.filter(c => c.source === filename && !c.embedding);
  if (targets.length === 0) {
    setDocStatus(`${filename}: already fully embedded.`);
    setTimeout(() => setDocStatus(''), 2000);
    return;
  }

  btnUploadPdf.disabled = true;
  try {
    let done = 0;
    for (const target of targets) {
      try {
        const emb = await embedText(target.text);
        const cur = getDocChunks();
        const idx = cur.findIndex(c => c.id === target.id);
        if (idx >= 0) {
          cur[idx].embedding = emb;
          setDocChunks(cur);
        }
        done++;
        setDocStatus(`Re-embedding ${filename} (${done}/${targets.length})...`);
        renderDocList();
        renderStorageBar();
      } catch (err) {
        setDocStatus(`Re-embed failed: ${err.message}`);
        await sleep(3500);
        return;
      }
    }
    setDocStatus(`Re-embedded ${filename} — ${done}/${targets.length}.`);
    setTimeout(() => setDocStatus(''), 2500);
  } finally {
    btnUploadPdf.disabled = false;
  }
}

function deleteFile(filename) {
  if (!confirm(`Delete ${filename} and all its chunks?`)) return;
  const remaining = getDocChunks().filter(c => c.source !== filename);
  setDocChunks(remaining);
  renderDocList();
  renderStorageBar();
}

// ---- Rendering ----
function groupChunksByFile(chunks) {
  // Preserves first-seen order so newly uploaded files show last.
  const order = [];
  const groups = new Map();
  for (const c of chunks) {
    if (!groups.has(c.source)) {
      groups.set(c.source, []);
      order.push(c.source);
    }
    groups.get(c.source).push(c);
  }
  return order.map(name => ({ name, chunks: groups.get(name) }));
}

function renderDocList() {
  const chunks = getDocChunks();
  const files = groupChunksByFile(chunks);
  docListEl.innerHTML = '';
  for (const f of files) {
    docListEl.appendChild(buildDocItem(f));
  }
  docEmptyEl.style.display = files.length === 0 ? '' : 'none';
}

function buildDocItem(file) {
  const root = document.createElement('div');
  root.className = 'doc-item';
  root.dataset.source = file.name;

  const total    = file.chunks.length;
  const embedded = file.chunks.filter(c => c.embedding).length;
  const bytes    = new Blob([JSON.stringify(file.chunks)]).size;

  const name = document.createElement('div');
  name.className = 'doc-item-name';
  name.textContent = file.name;
  root.appendChild(name);

  const meta = document.createElement('div');
  meta.className = 'doc-item-meta';

  const status = document.createElement('span');
  const statusClass =
    embedded === total ? 'embedded' :
    embedded === 0     ? 'unembedded' : 'partial';
  status.className = 'doc-item-status ' + statusClass;
  status.textContent = `${embedded}/${total} embedded`;
  meta.appendChild(status);

  const chunkCount = document.createElement('span');
  chunkCount.textContent = `${total} chunk${total === 1 ? '' : 's'}`;
  meta.appendChild(chunkCount);

  const sizeEl = document.createElement('span');
  sizeEl.textContent = formatBytes(bytes);
  meta.appendChild(sizeEl);

  root.appendChild(meta);

  const actions = document.createElement('div');
  actions.className = 'doc-item-actions';

  if (embedded < total) {
    const reEmbedBtn = document.createElement('button');
    reEmbedBtn.type = 'button';
    reEmbedBtn.className = 'btn-mini';
    reEmbedBtn.dataset.action = 're-embed';
    reEmbedBtn.textContent = 'Re-embed';
    actions.appendChild(reEmbedBtn);
  }

  const delBtn = document.createElement('button');
  delBtn.type = 'button';
  delBtn.className = 'btn-mini btn-danger';
  delBtn.dataset.action = 'delete';
  delBtn.textContent = 'Delete';
  actions.appendChild(delBtn);

  root.appendChild(actions);
  return root;
}

function renderStorageBar() {
  const bytes  = storageBytes();
  const max    = DOC_STORAGE_MAX_MB * 1024 * 1024;
  const warn   = DOC_STORAGE_WARN_MB * 1024 * 1024;
  const pct    = Math.min(100, (bytes / max) * 100);

  storageBarFillEl.style.width = `${pct.toFixed(1)}%`;
  storageBarFillEl.classList.toggle('warn', bytes >= warn);
  storageUsageText.textContent = `${formatBytes(bytes)} / ${DOC_STORAGE_MAX_MB} MB`;
}

// Tiny helpers used above.
function filePathBasename(p) {
  return String(p).split(/[\\/]/).pop();
}
function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

// ---------- UI helpers ----------
function setStatus(color, label) {
  dot.classList.remove('gray', 'cyan', 'green', 'amber', 'red');
  dot.classList.add(color);
  titleState.textContent = ' · ' + label;
}

// Source badge on the assistant section. Called after every successful
// transcript (tier1/tier2/ai) and when a history item is restored.
// Pass source=null to clear the badge and hide the Promote button.
//
// Uses explicit display values ('inline-flex' / 'inline-block') rather
// than '' to avoid any browser-edge-case ambiguity around clearing the
// inline display property the HTML starts with (style="display: none").
function setAnswerBadge(source, tier2Id) {
  answerBadge.className = 'answer-badge';
  answerBadge.textContent = '';
  answerBadge.style.display = 'none';
  btnPromote.style.display = 'none';
  state.currentTier2Id = null;

  if (!source) return;

  if (source === 'tier1') {
    answerBadge.classList.add('badge-tier1');
    answerBadge.textContent = '📋 Q&A';
    answerBadge.style.display = 'inline-flex';
  } else if (source === 'tier2') {
    answerBadge.classList.add('badge-tier2');
    answerBadge.textContent = '⚡ Cached';
    answerBadge.style.display = 'inline-flex';
    btnPromote.style.display = 'inline-block';
    state.currentTier2Id = tier2Id || null;
  } else if (source === 'ai') {
    const provider = storeGet(LS_KEY.provider) || 'openai';
    if (provider === 'claude') {
      answerBadge.classList.add('badge-claude');
      answerBadge.textContent = '✦ Claude';
    } else {
      answerBadge.classList.add('badge-gpt');
      answerBadge.textContent = '🤖 GPT';
    }
    answerBadge.style.display = 'inline-flex';
  }
}

function setWinHeight(px) {
  if (window.electronAPI && window.electronAPI.setWinHeight) {
    window.electronAPI.setWinHeight(px);
  }
}

function showSession(show) {
  sessionPanel.classList.toggle('show', !!show);
}
function showSettings(show) {
  settingsPanel.classList.toggle('show', !!show);
  state.settingsOpen = !!show;
}

// v0.3.0 — pill-bar is a sibling of the panels (always-visible header
// area), so its visibility is its own concern: shown everywhere except
// when the Settings panel is open (settings has its own internal tabs
// and the pill tabs would be confusing there).
function showPillBar(show) {
  pillBarEl.style.display = show ? '' : 'none';
}

function refreshLayout() {
  // Priority: settings > session > pill. v0.3.0 — state.screenActive
  // also forces the session panel open so the Screen tab is visible
  // even if a voice session isn't currently running. The pill tab
  // switcher stays visible in idle and expanded states.
  if (state.settingsOpen) {
    showSettings(true);
    showSession(false);
    showPillBar(false);
    setWinHeight(HEIGHT_SETTINGS);
  } else if (state.running || state.screenActive) {
    showSettings(false);
    showSession(true);
    showPillBar(true);
    setWinHeight(HEIGHT_EXPANDED);
  } else {
    showSettings(false);
    showSession(false);
    showPillBar(true);
    setWinHeight(HEIGHT_PILL);
  }
}

function renderHistory() {
  historyList.innerHTML = '';
  // Most recent on top, skip the in-flight one (last one is shown in main panel)
  const items = state.exchanges.slice(0, -1).slice(-5).reverse();
  if (items.length === 0) {
    historyList.innerHTML = '<div style="color:var(--text-dim); font-size:11px;">No previous questions yet.</div>';
    return;
  }
  for (const ex of items) {
    const div = document.createElement('div');
    div.className = 'history-item';
    div.innerHTML = `<span class="q-prefix">Q:</span>${escapeHtml(ex.q)}`;
    div.addEventListener('click', () => {
      heardEl.textContent  = ex.q;
      answerEl.textContent = ex.a;
      setAnswerBadge(ex.source, ex.tier2Id);
      // Belt-and-suspenders: explicitly re-assert the Promote button +
      // currentTier2Id for Tier 2 exchanges so the button is guaranteed
      // visible on history restore regardless of whatever setAnswerBadge
      // happens to do with display semantics.
      if (ex.source === 'tier2' && ex.tier2Id) {
        state.currentTier2Id = ex.tier2Id;
        btnPromote.style.display = 'inline-block';
      }
    });
    historyList.appendChild(div);
  }
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

// Aggressively strip every markdown artifact GPT might emit. Spoken interview
// answers should never contain backticks, asterisks, code fences, or headings.
function stripMarkdown(s) {
  return String(s)
    .replace(/```[a-z]*\n?/gi, '')      // code fence open with optional language
    .replace(/```/g, '')                 // code fence close
    .replace(/`/g, '')                   // inline backticks
    .replace(/\*\*/g, '')                // bold markers
    .replace(/^\s*#+\s+/gm, '')          // ATX headings
    .replace(/^\s*[*+\-]\s+/gm, '')      // leading bullets
    .replace(/\*/g, '')                  // remaining asterisks (italic)
    .replace(/^\s*>\s+/gm, '');          // blockquote markers
}

// ---------- Audio capture ----------
async function getSystemAudioStream() {
  // Try desktopCapturer (Windows WASAPI loopback)
  try {
    const sources = await window.electronAPI.getDesktopSources();
    if (!sources || sources.length === 0) throw new Error('No desktop sources');
    const src = sources[0];

    const raw = await navigator.mediaDevices.getUserMedia({
      audio: {
        mandatory: {
          chromeMediaSource: 'desktop',
          chromeMediaSourceId: src.id
        }
      },
      video: {
        mandatory: {
          chromeMediaSource: 'desktop',
          chromeMediaSourceId: src.id,
          maxWidth: 1, maxHeight: 1, maxFrameRate: 1
        }
      }
    });

    raw.getVideoTracks().forEach(t => t.stop());
    const audioTracks = raw.getAudioTracks();
    if (audioTracks.length === 0) throw new Error('No audio track from desktop source');
    warnBadge.classList.remove('show');
    return new MediaStream(audioTracks);
  } catch (err) {
    console.warn('System audio capture failed, falling back to mic:', err);
    warnBadge.classList.add('show');
    const mic = await navigator.mediaDevices.getUserMedia({ audio: true });
    return mic;
  }
}

// ---------- VAD loop ----------
// Strategy: keep AnalyserNode running continuously (cheap), but only spin up
// a MediaRecorder while a speech segment is in progress. Each recorder.stop()
// finalizes a complete, valid WebM blob (with EBML headers) for Whisper.
function startVadLoop() {
  const buf = new Float32Array(state.analyser.fftSize);

  const tick = () => {
    if (!state.running) return;
    state.rafId = requestAnimationFrame(tick);
    if (state.muted) return;

    state.analyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (let i = 0; i < buf.length; i++) sum += buf[i] * buf[i];
    const rms = Math.sqrt(sum / buf.length);

    const now = performance.now();

    if (rms > THRESHOLD) {
      if (!state.speechOn) {
        state.speechOn    = true;
        state.speechStart = now;
        beginSegmentRecording();
        setStatus('green', 'Hearing speech');
      }
      state.silenceStart = 0;
    } else if (state.speechOn) {
      if (state.silenceStart === 0) state.silenceStart = now;
      if (now - state.silenceStart >= SILENCE_MS) {
        const duration = now - state.speechStart;
        state.speechOn = false;
        state.silenceStart = 0;
        endSegmentRecording(duration >= MIN_SPEECH_MS);
      }
    }
  };

  state.rafId = requestAnimationFrame(tick);
}

function beginSegmentRecording() {
  if (!state.stream) return;
  state.bufferChunks = [];
  try {
    const rec = new MediaRecorder(state.stream, { mimeType: 'audio/webm' });
    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) state.bufferChunks.push(e.data);
    };
    rec.onstop = () => {
      const wantProcess = rec._wantProcess === true;
      const blob = new Blob(state.bufferChunks, { type: 'audio/webm' });
      state.bufferChunks = [];
      if (wantProcess && blob.size > 0) finalizeSegment(blob);
      else setStatus(state.muted ? 'gray' : 'cyan', state.muted ? 'Muted' : 'Listening');
    };
    state.recorder = rec;
    rec.start(); // no timeslice — one final chunk on stop = complete WebM
  } catch (err) {
    console.error('MediaRecorder start failed', err);
    setStatus('red', 'Recorder error: ' + err.message);
  }
}

function endSegmentRecording(shouldProcess) {
  const rec = state.recorder;
  state.recorder = null;
  if (!rec) return;
  rec._wantProcess = !!shouldProcess;
  try {
    if (rec.state !== 'inactive') rec.stop();
  } catch (err) {
    console.warn('MediaRecorder stop failed', err);
  }
}

function finalizeSegment(blob) {
  setStatus('amber', 'Processing');
  if (state.busy) {
    // Queue (overwrite — only latest pending is kept to avoid pileups)
    state.pendingBlob = blob;
    return;
  }
  processSegment(blob);
}

// =====================================================================
// THREE-TIER ANSWER PIPELINE
//
// Every transcript flows through three tiers in order. First match wins.
//   Tier 1 — hand-written Q&A bank (semantic match on embedding)
//   Tier 2 — auto-cached AI answers (semantic match on embedding)
//   Tier 3 — RAG (top-4 doc chunks) + AI generation, then autoCache
// =====================================================================

async function processTranscript(transcript, exchange) {
  const tier1 = getTier1();
  const tier2 = getTier2();
  const chunks = getDocChunks();

  const hasT1Embeddings    = tier1.some(e => e.embedding);
  const hasT2Embeddings    = tier2.some(e => e.embedding);
  const hasChunkEmbeddings = chunks.some(c => c.embedding);
  const anyEmbeddings = hasT1Embeddings || hasT2Embeddings || hasChunkEmbeddings;

  // Only spend an embedding call if there's something to match against.
  // Saves ~300ms (and an API call) when the user is starting out with no
  // Tier 1 entries / no Tier 2 cache / no PDFs uploaded.
  let embedding = null;
  if (anyEmbeddings) {
    try {
      embedding = await embedText(transcript);
    } catch (err) {
      // Embedding-only failure (network, missing key, rate limit) — fall
      // through to AI without tier matching rather than failing the whole
      // turn. AI may still succeed if the failure was specifically the
      // embedding endpoint.
      console.warn('Transcript embed failed, skipping tier matching:', err);
    }
  }

  const threshold = getThreshold();

  // Tier 1 — hand-written
  if (embedding && hasT1Embeddings) {
    const match = findMatch(embedding, tier1, threshold);
    if (match) {
      answerEl.textContent = match.answer;
      answerEl.scrollTop = 0;
      commitExchange(transcript, match.answer, exchange, 'tier1');
      setAnswerBadge('tier1');
      return;
    }
  }

  // Tier 2 — auto-cached
  if (embedding && hasT2Embeddings) {
    const match = findMatch(embedding, tier2, threshold);
    if (match) {
      answerEl.textContent = match.answer;
      answerEl.scrollTop = 0;
      exchange.tier2Id = match.id;
      commitExchange(transcript, match.answer, exchange, 'tier2');
      setAnswerBadge('tier2', match.id);
      return;
    }
  }

  // Tier 3 — RAG + AI generation
  const retrieved = (embedding && hasChunkEmbeddings)
    ? retrieveRelevantChunks(embedding, 4)
    : [];
  await generateAnswer(transcript, retrieved, exchange);
  // generateAnswer streams → finalizeAnswer → commitExchange(..., 'ai')
  setAnswerBadge('ai');

  // Auto-cache only when we have a usable embedding AND the AI actually
  // produced an answer (skip on stream errors / empty replies).
  if (embedding && exchange.a) {
    autoCache(transcript, embedding, exchange.a);
    // Live-update the Q&A Bank tab in case the user is looking at it.
    renderQaBank();
  }
}

async function processSegment(blob) {
  state.busy = true;
  try {
    const text = await transcribe(blob);
    const trimmed = (text || '').trim();
    if (!trimmed || trimmed.length < 6 || FILLER_RE.test(trimmed) || trimmed.split(/\s+/).length < 3) {
      // Skip noise
      setStatus(state.muted ? 'gray' : 'cyan', state.muted ? 'Muted' : 'Listening');
      return;
    }
    heardEl.textContent = trimmed;
    answerEl.textContent = '';
    answerSpin.style.display = 'inline';
    setAnswerBadge(null);

    const exchange = { q: trimmed, a: '', source: null };
    state.exchanges.push(exchange);
    renderHistory();

    await processTranscript(trimmed, exchange);
    answerSpin.style.display = 'none';
    setStatus(state.muted ? 'gray' : 'cyan', state.muted ? 'Muted' : 'Listening');
  } catch (err) {
    console.error(err);
    setStatus('red', 'Error: ' + err.message);
    answerSpin.style.display = 'none';
  } finally {
    state.busy = false;
    if (state.pendingBlob) {
      const next = state.pendingBlob;
      state.pendingBlob = null;
      processSegment(next);
    }
  }
}

// ---------- Whisper transcription (Groq → OpenAI) ----------
// If a Groq key is set, transcribe via Groq's whisper-large-v3-turbo first
// (much faster). On ANY Groq failure — missing/expired key, network, non-OK
// response — fall back silently to OpenAI Whisper. With no Groq key we go
// straight to OpenAI. Groq's endpoint is OpenAI-compatible, so the request
// shape (FormData, Bearer auth, { text } response) is identical.
async function transcribe(blob) {
  const groqKey = (storeGet(LS_KEY.groqKey) || '').trim();

  if (groqKey) {
    try {
      return await transcribeGroq(blob, groqKey);
    } catch (err) {
      console.warn('Groq transcription failed, falling back to OpenAI:', err.message);
      // fall through to OpenAI
    }
  }

  return transcribeOpenAI(blob);
}

async function transcribeGroq(blob, apiKey) {
  const fd = new FormData();
  fd.append('file', blob, 'audio.webm');
  fd.append('model', 'whisper-large-v3-turbo');

  const res = await fetch('https://api.groq.com/openai/v1/audio/transcriptions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: fd
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`Groq Whisper ${res.status}: ${txt.slice(0, 120)}`);
  }
  const data = await res.json();
  return data.text || '';
}

async function transcribeOpenAI(blob) {
  const apiKey = (storeGet(LS_KEY.apiKey) || '').trim();
  if (!apiKey) throw new Error('No API key — open settings');

  const fd = new FormData();
  fd.append('file', blob, 'audio.webm');
  fd.append('model', 'whisper-1');

  const res = await fetch('https://api.openai.com/v1/audio/transcriptions', {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}` },
    body: fd
  });
  if (!res.ok) {
    const txt = await res.text().catch(() => '');
    throw new Error(`Whisper ${res.status}: ${txt.slice(0, 120)}`);
  }
  const data = await res.json();
  return data.text || '';
}

// ---------- System prompt builder ----------
// Shared by both providers. retrievedChunks comes from the RAG retrieval
// in processTranscript(); empty array means no PDFs uploaded yet or no
// chunk was relevant enough — we just omit the document block.
function buildSystemPrompt(retrievedChunks = []) {
  const rules =
    `You are a presentation assistant helping a presenter answer questions from the audience.\n\n` +

    `Answer based on the uploaded presentation and research paper. ` +
    `Be concise, accurate, and grounded in the uploaded documents. ` +
    `Maximum 3 short paragraphs.\n\n` +

    `Never open with filler phrases like "Certainly", "Great question", ` +
    `"Sure", "Of course", "Absolutely". Start directly with substance.\n\n` +

    `Never define a term already used in the question. ` +
    `Go straight to the approach, not the definition.\n\n` +

    `Output plain text only. No markdown. No backticks. ` +
    `No asterisks. No headings. No code blocks.`;

  if (retrievedChunks.length === 0) return rules;

  const rag = '\n\nRelevant background from uploaded documents:\n' +
    retrievedChunks.map(c => `[From ${c.source}]\n${c.text}`).join('\n\n---\n\n');
  return rules + rag;
}

// ---------- RAG retrieval ----------
// Cosine-rank all embedded doc chunks against the transcript embedding,
// return the top N. No similarity floor — even a loose match is useful
// context when the question is open-ended. The matching system relies
// on the threshold for Tier 1/Tier 2 hits, not for RAG context.
function retrieveRelevantChunks(embedding, topN = 4) {
  const chunks = getDocChunks().filter(c => c.embedding);
  if (chunks.length === 0) return [];
  return chunks
    .map(c => ({ ...c, score: cosineSimilarity(embedding, c.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
}

// ---------- Provider dispatch ----------
async function generateAnswer(question, retrievedChunks, exchange) {
  const systemPrompt = buildSystemPrompt(retrievedChunks);
  const provider = storeGet(LS_KEY.provider) || 'openai';
  return provider === 'claude'
    ? streamClaudeAnswer(question, systemPrompt, exchange)
    : streamOpenAIAnswer(question, systemPrompt, exchange);
}

// ---------- Shared exchange housekeeping ----------
// Records a completed Q→A pair onto the exchange + conversation history.
// Used by both AI streaming finalisation and instant Tier 1/Tier 2 cache
// hits (Task 8). The source value flows into Task 9's badge rendering.
function commitExchange(question, answer, exchange, source) {
  exchange.a = answer;
  exchange.source = source;
  state.convHistory.push({ role: 'user',      content: question });
  state.convHistory.push({ role: 'assistant', content: answer });
  if (state.convHistory.length > MAX_HISTORY) {
    state.convHistory = state.convHistory.slice(-MAX_HISTORY);
  }
  renderHistory();
}

// Both providers stream tokens to the UI directly; this records the final
// answer once streaming completes. Strip markdown so future model responses
// mirror plain-text style on the next turn.
function finalizeAnswer(question, rawAnswer, exchange) {
  commitExchange(question, stripMarkdown(rawAnswer), exchange, 'ai');
}

// ---------- OpenAI: GPT-4o/4o-mini streaming ----------
async function streamOpenAIAnswer(question, systemPrompt, exchange) {
  const apiKey = (storeGet(LS_KEY.apiKey) || '').trim();
  const model  =  storeGet(LS_KEY.model)  || 'gpt-4o';
  if (!apiKey) throw new Error('No OpenAI API key — open settings');

  const messages = [
    { role: 'system', content: systemPrompt },
    ...state.convHistory.slice(-MAX_HISTORY),
    { role: 'user', content: question }
  ];

  const res = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model,
      messages,
      stream: true,
      max_tokens: 300,
      temperature: 0.4
    })
  });

  if (!res.ok || !res.body) {
    const txt = await res.text().catch(() => '');
    throw new Error(`GPT ${res.status}: ${txt.slice(0, 120)}`);
  }

  const reader  = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buf = '';
  let answer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      try {
        const json = JSON.parse(payload);
        const delta = json.choices?.[0]?.delta?.content;
        if (delta) {
          answer += delta;
          // Follow the stream only if the user is already parked at the
          // bottom; otherwise leave their scroll position alone (no forced
          // auto-scroll while they're reading mid-answer). Measure BEFORE
          // updating the text, since the update grows scrollHeight.
          const atBottom =
            answerEl.scrollHeight - answerEl.scrollTop - answerEl.clientHeight < 20;
          answerEl.textContent = stripMarkdown(answer);
          if (atBottom) answerEl.scrollTop = answerEl.scrollHeight;
        }
      } catch {
        // ignore partial frames
      }
    }
  }

  finalizeAnswer(question, answer, exchange);
}

// ---------- Anthropic: Claude streaming ----------
// Anthropic's API rejects browser-origin requests by default; the explicit
// `anthropic-dangerous-direct-browser-access` header opts in. We accept the
// risk because this is a single-user local desktop app with the key stored
// in userData/settings.json on disk (same trust model as the OpenAI key
// already in use).
async function streamClaudeAnswer(question, systemPrompt, exchange) {
  const apiKey = (storeGet(LS_KEY.claudeKey) || '').trim();
  const model  =  storeGet(LS_KEY.model)     || 'claude-sonnet-4-5';
  if (!apiKey) throw new Error('No Claude API key — open settings');

  // Claude API takes the system prompt separately, and `messages` is only
  // user/assistant turns. convHistory already obeys that shape.
  const messages = [
    ...state.convHistory.slice(-MAX_HISTORY),
    { role: 'user', content: question }
  ];

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true'
    },
    body: JSON.stringify({
      model,
      max_tokens: 300,
      stream: true,
      system: systemPrompt,
      messages
    })
  });

  if (!res.ok || !res.body) {
    const txt = await res.text().catch(() => '');
    throw new Error(`Claude ${res.status}: ${txt.slice(0, 120)}`);
  }

  const reader  = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buf = '';
  let answer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      try {
        const json = JSON.parse(payload);
        // Claude emits typed events; only content_block_delta carries tokens.
        // Other events (message_start, content_block_start/stop, message_delta,
        // message_stop, ping) are ignored.
        if (json.type === 'content_block_delta') {
          const token = json.delta && json.delta.text;
          if (token) {
            answer += token;
            // Follow the stream only if the user is already at the bottom;
            // otherwise don't move their viewport. Measure before the text
            // update grows scrollHeight.
            const atBottom =
              answerEl.scrollHeight - answerEl.scrollTop - answerEl.clientHeight < 20;
            answerEl.textContent = stripMarkdown(answer);
            if (atBottom) answerEl.scrollTop = answerEl.scrollHeight;
          }
        }
      } catch {
        // ignore partial frames
      }
    }
  }

  finalizeAnswer(question, answer, exchange);
}

// ---------- Session control ----------
async function startSession() {
  const apiKey = (storeGet(LS_KEY.apiKey) || '').trim();
  if (!apiKey) {
    // Soft warning — allow audio capture / VAD to be tested without a key.
    // transcribe() will throw when the first segment hits.
    console.warn('No API key set — transcription will fail until you save one in Settings.');
  }
  try {
    setStatus('amber', 'Starting...');
    state.stream = await getSystemAudioStream();

    state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = state.audioCtx.createMediaStreamSource(state.stream);
    state.analyser = state.audioCtx.createAnalyser();
    state.analyser.fftSize = FFT_SIZE;
    source.connect(state.analyser);
    // Do NOT connect analyser to destination — we don't want to play it back
    // MediaRecorder is started/stopped per speech segment in beginSegmentRecording().

    state.running = true;
    state.speechOn = false;
    state.silenceStart = 0;
    btnToggle.textContent = 'Stop';
    btnToggle.classList.remove('btn-primary');
    btnToggle.classList.add('btn-danger');
    setStatus('cyan', 'Listening');
    refreshLayout();
    startVadLoop();
  } catch (err) {
    console.error(err);
    setStatus('red', 'Start failed: ' + err.message);
    await stopSession();
  }
}

async function stopSession() {
  state.running = false;
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = null;
  }
  if (state.recorder) {
    try { if (state.recorder.state !== 'inactive') state.recorder.stop(); } catch {}
    state.recorder = null;
  }
  if (state.stream) {
    state.stream.getTracks().forEach(t => { try { t.stop(); } catch {} });
    state.stream = null;
  }
  if (state.audioCtx) {
    try { await state.audioCtx.close(); } catch {}
    state.audioCtx = null;
  }
  state.analyser = null;
  state.bufferChunks = [];
  state.busy = false;
  state.pendingBlob = null;
  state.speechOn = false;
  btnToggle.textContent = 'Start';
  btnToggle.classList.add('btn-primary');
  btnToggle.classList.remove('btn-danger');
  setStatus('gray', 'Idle');
  refreshLayout();
}

function clearSession() {
  state.convHistory = [];
  state.exchanges = [];
  heardEl.textContent = '';
  answerEl.textContent = '';
  setAnswerBadge(null);
  renderHistory();
}

// ---------- Mute toggle ----------
function toggleMute() {
  if (!state.running) return;
  state.muted = !state.muted;
  muteBadge.classList.toggle('show', state.muted);
  if (state.muted) {
    setStatus('gray', 'Muted');
  } else {
    setStatus('cyan', 'Listening');
  }
}

// ---------- Wire up UI ----------
btnToggle.addEventListener('click', () => {
  if (state.running) stopSession();
  else startSession();
});


btnSettings.addEventListener('click', () => {
  showSettings(!state.settingsOpen);
  refreshLayout();
});

// Custom window-chrome buttons (frame:false means no native ones).
btnMin.addEventListener('click', () => {
  if (window.electronAPI && window.electronAPI.winMinimize) {
    window.electronAPI.winMinimize();
  }
});

btnMax.addEventListener('click', () => {
  if (window.electronAPI && window.electronAPI.winMaximize) {
    window.electronAPI.winMaximize();
  }
});

// Keep the maximize glyph + title in sync with the actual window state.
// Main fires this on every maximize / unmaximize, including the native
// double-click-titlebar path, so this stays correct even when we didn't
// initiate the change ourselves.
if (window.electronAPI && window.electronAPI.onMaximizeChanged) {
  window.electronAPI.onMaximizeChanged((isMax) => {
    btnMax.textContent = isMax ? '❐' : '▢';
  });
}

btnClose.addEventListener('click', async () => {
  await stopSession();
  window.close();
});

btnCopy.addEventListener('click', async () => {
  const text = answerEl.textContent || '';
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    const original = btnCopy.textContent;
    btnCopy.textContent = 'Copied';
    setTimeout(() => { btnCopy.textContent = original; }, 1200);
  } catch (err) {
    console.warn('Clipboard failed', err);
  }
});

btnClear.addEventListener('click', () => {
  clearSession();
});

// Promote ↑ — visible only when the current answer came from Tier 2.
// Opens the settings panel on the Q&A tab with the promote form prefilled
// from the matched Tier 2 entry.
btnPromote.addEventListener('click', () => {
  const id = state.currentTier2Id;
  if (!id) return;
  // Guard against the entry having been deleted (e.g. via Clear all) since
  // it was painted in the answer panel.
  if (!getTier2().some(e => e.id === id)) {
    setAnswerBadge(null);
    return;
  }
  state.settingsOpen = true;
  setActiveTab('qa');
  refreshLayout();
  startPromoteTier2(id);
});

btnSave.addEventListener('click', () => {
  saveSettings();
});

providerOpenAIBtn.addEventListener('click', () => setProvider('openai'));
providerClaudeBtn.addEventListener('click', () => setProvider('claude'));

for (const btn of tabButtons) {
  btn.addEventListener('click', () => setActiveTab(btn.dataset.tab));
}

// Live-update the value badge as the slider is dragged. The actual write
// to ca_threshold happens on Save (consistent with how the other Tab 1
// fields behave).
thresholdSlider.addEventListener('input', () => {
  thresholdValue.textContent = parseFloat(thresholdSlider.value).toFixed(2);
});

// ---- Documents wiring ----
btnUploadPdf.addEventListener('click', uploadPdfs);
docListEl.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const source = btn.closest('.doc-item')?.dataset.source;
  if (!source) return;
  const action = btn.dataset.action;
  if      (action === 're-embed') reEmbedFile(source);
  else if (action === 'delete')   deleteFile(source);
});

// ---- Q&A Bank wiring ----
btnAddTier1.addEventListener('click', startAddTier1);
btnImportTier1.addEventListener('click', importTier1FromTxt);
btnEmbedAllTier1.addEventListener('click', embedAllTier1Unembedded);
btnClearTier2.addEventListener('click', clearTier2);
btnQaSave.addEventListener('click', saveQaForm);
btnQaCancel.addEventListener('click', cancelQaForm);

// Event delegation for dynamically-rendered list items.
tier1ListEl.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const id = btn.closest('.qa-item')?.dataset.id;
  if (!id) return;
  const action = btn.dataset.action;
  if      (action === 'embed')  embedTier1Entry(id);
  else if (action === 'edit')   startEditTier1(id);
  else if (action === 'delete') deleteTier1Entry(id);
});

tier2ListEl.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-action]');
  if (!btn) return;
  const id = btn.closest('.qa-item')?.dataset.id;
  if (!id) return;
  const action = btn.dataset.action;
  if      (action === 'promote') startPromoteTier2(id);
  else if (action === 'review')  toggleTier2Reviewed(id);
  else if (action === 'delete')  deleteTier2Entry(id);
});

// =====================================================================
// CODING MODE (v0.3.0) — Screen capture → Vision API → code answer
//
// Pipeline:
//   1. main.js's SCREEN_CAPTURE_HOTKEY fires captureScreen(), which sends
//      a JPEG base64 over IPC.
//   2. handleScreenshot() activates the Screen tab, calls the vision API
//      via streamVisionAnswer(), and streams the model's response into
//      #screen-answer-content live.
//   3. After streaming completes, parseCodeAnswer() splits the structured
//      response and renderCodeBlock() builds a syntax-highlighted view.
//
// Voice-tab UI and audio pipeline are not touched here — Coding Mode is
// fully additive.
// =====================================================================

function switchTab(name) {
  const isVoice = name === 'voice';
  tabVoiceBtn.classList.toggle('active', isVoice);
  tabScreenBtn.classList.toggle('active', !isVoice);
  panelVoice.classList.toggle('active', isVoice);
  panelScreen.classList.toggle('active', !isVoice);
}

// v0.3.2 — broader prompt covering MCQ, Fill-in-Blank, Scenario, DSA,
// SQL, System Design, and Behavioral. Output schema changed to
// TYPE / QUESTION / ANSWER / REASONING — parseVisionAnswer() expects
// this shape. REASONING is mandatory for every question type.
function buildVisionSystemPrompt() {
  return (
    'You are a presentation assistant helping answer questions ' +
    'about the uploaded presentation and research paper from screenshots.\n\n' +

    'CRITICAL — Read the ENTIRE screenshot from top to bottom ' +
    'before doing anything else. This includes:\n' +
    '- The question text\n' +
    '- Any existing code\n' +
    '- Output panels and console logs\n' +
    '- Error messages especially those below the code area\n' +
    '- Any highlighted lines or red underlines\n\n' +

    'PRIORITY RULE: If ANY error message is visible anywhere ' +
    'on screen — fixing that error is your primary task. ' +
    'Do not solve a different problem.\n\n' +

    'Question types:\n' +
    '- MCQ: State correct option letter and text. Explain why ' +
    'correct and why others are wrong.\n' +
    '- Fill-in-Blank: Give exact missing word or phrase. ' +
    'Brief concept explanation.\n' +
    '- DSA / Coding: Write complete working code in a ' +
    'triple-backtick fence with correct language tag. ' +
    'Code must be correct, handle edge cases, ready to run. ' +
    'Match the language shown in question or existing code. ' +
    'If no language shown, infer from context.\n' +
    '- Error Fix: Show corrected code with fix marked in a comment. ' +
    'Explain the cause and why the fix works.\n' +
    '- System Design: Structured breakdown — components, ' +
    'data flow, decisions, trade-offs.\n' +
    '- Behavioral: STAR format — Situation, Task, Action, Result.\n\n' +

    'Output format — exact prefixes in this order:\n' +
    'TYPE: [MCQ / Fill-in-Blank / DSA / Error-Fix / System-Design / Behavioral]\n' +
    'QUESTION: [exact question text from screenshot]\n' +
    'ERROR DETECTED: [exact error message if visible — omit if none]\n' +
    'ANSWER:\n' +
    '[complete answer]\n' +
    'REASONING:\n' +
    '[explanation — never omit]'
  );
}

// v0.3.2 — builds a provider-specific messages array from the
// session's images + text turns. The first user turn carries ALL
// images (in order); subsequent turns are text-only. Both OpenAI and
// Claude accept multiple image blocks in a single content array, so
// the wire format is just the same single-image shape repeated.
function buildVisionMessages(provider, images, turns) {
  const messages = [];
  const imageList = Array.isArray(images) ? images : [];
  let firstUserAttached = false;
  for (const turn of turns) {
    if (turn.role === 'user' && imageList.length > 0 && !firstUserAttached) {
      const content = [];
      for (const img of imageList) {
        if (provider === 'openai') {
          content.push({
            type: 'image_url',
            image_url: { url: 'data:image/jpeg;base64,' + img, detail: 'high' }
          });
        } else { // claude
          content.push({
            type: 'image',
            source: { type: 'base64', media_type: 'image/jpeg', data: img }
          });
        }
      }
      content.push({ type: 'text', text: turn.text });
      messages.push({ role: 'user', content });
      firstUserAttached = true;
    } else {
      messages.push({ role: turn.role, content: turn.text });
    }
  }
  return messages;
}

async function callOpenAIVision(messages, systemPrompt, model, signal) {
  const apiKey = (storeGet(LS_KEY.apiKey) || '').trim();
  return fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + apiKey
    },
    body: JSON.stringify({
      model,
      max_tokens: 1500,
      stream: true,
      messages: [{ role: 'system', content: systemPrompt }, ...messages]
    })
  });
}

async function callClaudeVision(messages, systemPrompt, model, signal) {
  const apiKey = (storeGet(LS_KEY.claudeKey) || '').trim();
  return fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    signal,
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
      'anthropic-dangerous-direct-browser-access': 'true'
    },
    body: JSON.stringify({
      model,
      max_tokens: 1500,
      stream: true,
      system: systemPrompt,
      messages
    })
  });
}

// Streams the vision response into #screen-answer-content live. Returns
// the fully-assembled text once the stream ends. Throws on HTTP errors
// and propagates AbortError if the caller cancels via signal.
//
// v0.3.1 — `messages` is now a provider-shaped array (built by
// buildVisionMessages), not a raw base64. This is what enables
// multi-turn follow-ups within the same screen session.
async function streamVisionAnswer(messages, systemPrompt, signal) {
  const visionModel = storeGet(LS_KEY.visionModel) || 'gpt-4o';
  const isClaude = visionModel.indexOf('claude') === 0;

  const response = isClaude
    ? await callClaudeVision(messages, systemPrompt, visionModel, signal)
    : await callOpenAIVision(messages, systemPrompt, visionModel, signal);

  if (!response.ok || !response.body) {
    const errText = await response.text().catch(() => '');
    throw new Error('Vision API ' + response.status + ': ' + errText.slice(0, 200));
  }

  const reader  = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buf = '';
  let fullText = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop() || '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') continue;
      try {
        const json = JSON.parse(payload);
        let delta = '';
        if (isClaude) {
          if (json.type === 'content_block_delta') {
            delta = (json.delta && json.delta.text) || '';
          }
        } else {
          delta = (json.choices && json.choices[0] && json.choices[0].delta && json.choices[0].delta.content) || '';
        }
        if (delta) {
          fullText += delta;
          // Live render: raw text during streaming, code highlighting
          // applied in the final pass after the stream completes.
          // Follow only if already at the bottom — never yank the viewport
          // while the user is reading. Measure before the text update.
          const atBottom =
            screenAnswerContent.scrollHeight - screenAnswerContent.scrollTop - screenAnswerContent.clientHeight < 20;
          screenAnswerContent.textContent = fullText;
          if (atBottom) screenAnswerContent.scrollTop = screenAnswerContent.scrollHeight;
        }
      } catch {
        // ignore partial frames
      }
    }
  }
  return fullText;
}

// Splits the model's structured response. Tolerant of leading whitespace
// and case on the TYPE/QUESTION/SOLUTION markers; everything after the
// v0.3.2 — parses the four-section schema produced by the prompt.
// State machine: TYPE/QUESTION are single-line; ANSWER and REASONING
// each span until the next prefix or end of stream. Tolerant of
// leading whitespace and case on the markers. Returns blank strings
// for any section the model omitted.
function parseVisionAnswer(rawText) {
  const lines = String(rawText || '').split('\n');
  let type = '', question = '', answer = '', reasoning = '';
  let section = null;  // null | 'answer' | 'reasoning'
  for (const line of lines) {
    if (/^\s*TYPE\s*:/i.test(line)) {
      type = line.replace(/^\s*TYPE\s*:/i, '').trim();
      section = null;
    } else if (/^\s*QUESTION\s*:/i.test(line)) {
      question = line.replace(/^\s*QUESTION\s*:/i, '').trim();
      section = null;
    } else if (/^\s*ANSWER\s*:/i.test(line)) {
      section = 'answer';
      const rest = line.replace(/^\s*ANSWER\s*:/i, '').trim();
      if (rest) answer += rest + '\n';
    } else if (/^\s*REASONING\s*:/i.test(line)) {
      section = 'reasoning';
      const rest = line.replace(/^\s*REASONING\s*:/i, '').trim();
      if (rest) reasoning += rest + '\n';
    } else if (section === 'answer') {
      answer += line + '\n';
    } else if (section === 'reasoning') {
      reasoning += line + '\n';
    }
  }
  return {
    type,
    question,
    answer:    answer.trim(),
    reasoning: reasoning.trim()
  };
}

function detectLanguage(text) {
  // SQL keywords are short and high-signal — check first to avoid mis-
  // detecting "SELECT" inside a Python docstring etc.
  if (/\b(SELECT|FROM\b|WHERE\b|JOIN\b|GROUP BY|ORDER BY|INSERT INTO|CREATE TABLE)\b/i.test(text)) return 'sql';
  if (/\bdef\s+\w+\s*\(|^import\s+\w|\bprint\(/m.test(text)) return 'python';
  if (/public\s+class|System\.out\.|public\s+static\s+void/.test(text)) return 'java';
  if (/\b(function\s+\w|const\s+\w|let\s+\w|=>\s*[\{\(])/.test(text)) return 'javascript';
  const pref = (storeGet(LS_KEY.codingLang) || 'python').toLowerCase();
  if (pref === 'c++') return 'cpp';
  return pref;
}

// v0.3.2 — renders the ANSWER section. If the text contains a
// triple-backtick code fence, builds a Prism-highlighted code block
// (with optional prose above / below the fence). If there's no fence
// — MCQ option, Fill-in-Blank word, Scenario answer, etc. — renders
// the answer as plain prose without a code wrapper.
function renderAnswerBody(answerText) {
  const text = String(answerText || '');
  if (!text.trim()) return '';

  const codeMatch = text.match(/```([\w+\-]*)\n([\s\S]*?)```/);

  if (!codeMatch) {
    // No code fence — pure prose answer.
    return '<div class="answer-text-plain">' + escapeHtml(text.trim()) + '</div>';
  }

  // Code answer — pull out fence + any prose on either side of it.
  const code        = codeMatch[2];
  const fenceLang   = codeMatch[1];
  const beforeProse = text.slice(0, codeMatch.index).trim();
  const afterProse  = text.slice(codeMatch.index + codeMatch[0].length).trim();
  const lang        = (fenceLang || detectLanguage(code) || 'python').toLowerCase();

  const beforeHtml = beforeProse
    ? '<div class="answer-text-plain" style="margin-bottom: 8px;">' + escapeHtml(beforeProse) + '</div>'
    : '';
  const afterHtml = afterProse
    ? '<div class="answer-text-plain" style="margin-top: 8px;">' + escapeHtml(afterProse) + '</div>'
    : '';

  return (
    beforeHtml +
    '<div class="code-block-wrapper">' +
      '<div class="code-header">' +
        '<span class="lang-badge">' + escapeHtml(lang) + '</span>' +
        '<button class="copy-btn" type="button">Copy</button>' +
      '</div>' +
      '<pre class="code-pre"><code class="language-' + escapeHtml(lang) + '">' +
        escapeHtml(code.trim()) +
      '</code></pre>' +
    '</div>' +
    afterHtml
  );
}

// v0.3.2 — renders the REASONING section below the answer. Always
// plain prose with a small uppercase "REASONING" label above.
function renderReasoning(reasoningText) {
  const text = String(reasoningText || '').trim();
  if (!text) return '';
  return (
    '<div class="reasoning-block">' +
      '<div class="reasoning-label">REASONING</div>' +
      '<div class="reasoning-text">' + escapeHtml(text) + '</div>' +
    '</div>'
  );
}

function copyCodeFromButton(btn) {
  const wrap = btn.closest('.code-block-wrapper');
  if (!wrap) return;
  const codeEl = wrap.querySelector('code');
  if (!codeEl) return;
  const text = codeEl.innerText || codeEl.textContent || '';
  navigator.clipboard.writeText(text).then(() => {
    const original = btn.textContent;
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = original; }, 1600);
  }).catch(err => {
    console.warn('Clipboard write failed:', err);
  });
}

// v0.3.1 — UI helpers for the new input bars.
function showFeedbackBar(show) {
  screenFeedbackBarEl.style.display = show ? '' : 'none';
}

function setScreenInputsBusy(busy) {
  // Disable BOTH input rows during an in-flight stream so the user can't
  // queue an inconsistent turn (would leave consecutive user messages in
  // the conversation, which OpenAI/Claude reject).
  if (screenInputEl)         screenInputEl.disabled         = busy;
  if (btnScreenInputSend)    btnScreenInputSend.disabled    = busy;
  if (screenFeedbackEl)      screenFeedbackEl.disabled      = busy;
  if (btnScreenFeedbackSend) btnScreenFeedbackSend.disabled = busy;
}

// v0.3.2 — heart of the threaded screen-conversation. Reads the current
// state.screenSession (images + turns), builds provider-shaped messages,
// streams the response, parses + renders ANSWER and REASONING, then
// appends the assistant turn to the session so future follow-ups carry
// full context.
async function runVisionTurn() {
  // Abort any in-flight stream — happens when a new submission lands
  // before the previous answer finished, or when the user sends another
  // follow-up too quickly. The previous run's AbortError is swallowed
  // by its own try/catch.
  if (state.screenAbort) {
    try { state.screenAbort.abort(); } catch { /* nothing */ }
    state.screenAbort = null;
  }

  screenStatus.textContent        = '🔍 Reading...';
  screenAnswerContent.innerHTML   = '';
  screenAnswerContent.textContent = '';

  // Pre-flight API key check — fail fast with a clear message rather
  // than 401-ing the user later.
  const visionModel = storeGet(LS_KEY.visionModel) || 'gpt-4o';
  const isClaude    = visionModel.indexOf('claude') === 0;
  const provider    = isClaude ? 'claude' : 'openai';
  const apiKey      = (storeGet(isClaude ? LS_KEY.claudeKey : LS_KEY.apiKey) || '').trim();
  if (!apiKey) {
    screenStatus.textContent = '';
    screenAnswerContent.textContent = isClaude
      ? 'Set your Claude API key in Settings before using Screen Capture.'
      : 'Set your OpenAI API key in Settings before using Screen Capture.';
    return;
  }

  const controller = new AbortController();
  state.screenAbort = controller;
  setScreenInputsBusy(true);

  try {
    const messages     = buildVisionMessages(provider, state.screenSession.images, state.screenSession.turns);
    const systemPrompt = buildVisionSystemPrompt();
    const fullText     = await streamVisionAnswer(messages, systemPrompt, controller.signal);

    // Record the assistant turn so the next follow-up carries the full
    // thread context. We do this before rendering so even a render-time
    // throw leaves the session coherent.
    if (fullText) {
      state.screenSession.turns.push({ role: 'assistant', text: fullText });
    }

    if (!fullText.trim()) {
      screenStatus.textContent = '';
      screenAnswerContent.textContent = 'No response received from the vision model.';
      return;
    }

    const parsed = parseVisionAnswer(fullText);

    // If the model didn't follow the TYPE/QUESTION/ANSWER/REASONING
    // schema at all, show the raw text rather than dropping a partial
    // response.
    if (!parsed.type && !parsed.question && !parsed.answer && !parsed.reasoning) {
      screenStatus.textContent = '';
      screenAnswerContent.textContent = fullText;
      showFeedbackBar(true);
      return;
    }
    if (!parsed.answer && !parsed.question && !parsed.reasoning) {
      screenStatus.textContent = '';
      screenAnswerContent.textContent =
        'No question detected — try again on a screen with text.';
      return;
    }

    screenStatus.textContent = '';

    // Only update the question meta if this turn produced new
    // type/question info. Follow-up turns usually skip these markers,
    // so we preserve whatever badge/question we already had.
    if (parsed.type || parsed.question) {
      const metaParts = [];
      if (parsed.type)     metaParts.push('<span class="screen-meta-badge">' + escapeHtml(parsed.type) + '</span>');
      if (parsed.question) metaParts.push(escapeHtml(parsed.question));
      screenQuestionMeta.innerHTML = metaParts.join(' ');
    }

    // v0.3.2 — render ANSWER + REASONING together. renderAnswerBody
    // returns a code block if the answer contains a fence, else plain
    // prose. renderReasoning returns '' if reasoning is missing.
    const answerHtml    = renderAnswerBody(parsed.answer || parsed.question || '');
    const reasoningHtml = renderReasoning(parsed.reasoning);
    screenAnswerContent.innerHTML = answerHtml + reasoningHtml;

    // Prism highlight only if a code block was actually rendered —
    // avoids needlessly walking the DOM on prose-only answers.
    if (screenAnswerContent.querySelector('code[class^="language-"]')) {
      if (window.Prism && typeof window.Prism.highlightAllUnder === 'function') {
        window.Prism.highlightAllUnder(screenAnswerContent);
      } else if (window.Prism && typeof window.Prism.highlightAll === 'function') {
        window.Prism.highlightAll();
      }
    }

    // Surface the feedback bar now that an answer is rendered.
    showFeedbackBar(true);
  } catch (err) {
    if (err && err.name === 'AbortError') {
      // Superseded by a newer turn — silent.
      return;
    }
    const msg = (err && err.message) ? err.message : String(err);
    screenStatus.textContent = '';
    if (/Failed to fetch|Network|NetworkError/i.test(msg)) {
      screenAnswerContent.textContent = '❌ Network error — check connection and try again.';
    } else {
      screenAnswerContent.textContent = '❌ ' + msg;
    }
  } finally {
    setScreenInputsBusy(false);
    if (state.screenAbort === controller) state.screenAbort = null;
  }
}

// v0.3.2 — start a brand-new screen session. Wipes prior session state,
// attaches zero-or-more screenshots, primes the first user turn, then
// kicks off a vision call.
function startScreenSession(initialUserText, images, autoSwitch) {
  state.screenSession = {
    images: Array.isArray(images) ? images.slice() : [],
    turns:  [{ role: 'user', text: initialUserText }]
  };
  // Reset the meta badge + feedback bar for the new session.
  screenQuestionMeta.innerHTML = '';
  showFeedbackBar(false);

  state.screenActive = true;
  refreshLayout();
  if (autoSwitch) switchTab('screen');

  return runVisionTurn();
}

// Continue the current screen session with a feedback / follow-up
// message. If there's no active session (no prior turn), this falls
// through to starting a fresh text-only session.
function continueScreenSession(feedbackText) {
  if (!state.screenSession || !state.screenSession.turns || state.screenSession.turns.length === 0) {
    return startScreenSession(feedbackText, [], true);
  }
  state.screenSession.turns.push({ role: 'user', text: feedbackText });
  return runVisionTurn();
}

// =====================================================================
// v0.3.2 — Multi-screenshot queue
//
// Captures arriving via Ctrl+Shift+S or the 📸 pill button no longer
// auto-submit. They stack in state.screenshotQueue (capped at
// SCREENSHOT_QUEUE_MAX) and surface in a thumbnail strip. The Submit
// button drains the queue into screenSession.images and fires one
// multi-image vision turn; the Clear button discards the queue.
// =====================================================================

function refreshQueueUI() {
  const count = state.screenshotQueue.length;

  // 📸 capture button reflects the queue count as a small badge.
  if (btnCapture) {
    btnCapture.textContent = count > 0 ? '📸 ' + count : '📸';
  }

  // Show the queue bar only when there's something in it.
  if (screenQueueBar) {
    screenQueueBar.style.display = count > 0 ? '' : 'none';
  }

  // Submit button is the green CTA; label includes the count for
  // immediate feedback ("Submit (3)"). Disabled when queue is empty
  // — relevant only if the queue bar is somehow still visible.
  if (btnSubmitQueue) {
    btnSubmitQueue.textContent = count > 0 ? 'Submit (' + count + ')' : 'Submit';
    btnSubmitQueue.disabled    = count === 0;
  }
  if (btnClearQueue) {
    btnClearQueue.disabled = count === 0;
  }

  // Render thumbnails. Built fresh on every refresh (cheap — the
  // queue is capped at 10 and the data is already in-memory base64).
  if (screenThumbnails) {
    screenThumbnails.innerHTML = '';
    for (let i = 0; i < state.screenshotQueue.length; i++) {
      const base64 = state.screenshotQueue[i];
      const wrap = document.createElement('div');
      wrap.className = 'screen-thumb';
      wrap.dataset.index = String(i);

      const img = document.createElement('img');
      img.src = 'data:image/jpeg;base64,' + base64;
      img.alt = 'Screenshot ' + (i + 1);
      wrap.appendChild(img);

      const removeBtn = document.createElement('button');
      removeBtn.type = 'button';
      removeBtn.className = 'screen-thumb-remove';
      removeBtn.dataset.action = 'remove-thumb';
      removeBtn.textContent = '×';  // ×
      wrap.appendChild(removeBtn);

      screenThumbnails.appendChild(wrap);
    }
  }
}

function enqueueScreenshot(base64) {
  if (!base64) return;
  if (state.screenshotQueue.length >= SCREENSHOT_QUEUE_MAX) {
    // Silent cap — the badge stays at "📸 10" so the user can see
    // they hit the limit.
    return;
  }
  state.screenshotQueue.push(base64);

  // Surface the Screen tab so the user can see the thumbnail and
  // the Submit button. Same auto-switch policy as a normal capture.
  state.screenActive = true;
  refreshLayout();
  if (storeGet(LS_KEY.autoSwitchTab)) switchTab('screen');

  refreshQueueUI();
}

function submitQueuedScreenshots() {
  if (state.screenshotQueue.length === 0) return;
  // Snapshot + drain. From this point on the screen session owns
  // the images; the queue is empty and ready for the next round.
  const images = state.screenshotQueue.slice();
  state.screenshotQueue = [];
  refreshQueueUI();

  const initialText = images.length === 1
    ? 'Read this entire screenshot from top to bottom including all ' +
      'error messages, output panels, and console logs. If there is ' +
      'an error visible anywhere on screen, fix it. Otherwise extract ' +
      'the question and solve it completely.'
    : 'Read all ' + images.length + ' screenshots completely including ' +
      'all error messages, output panels, and console logs. If there ' +
      'is an error visible anywhere, fix it. Otherwise extract the ' +
      'questions and solve them completely.';
  startScreenSession(initialText, images, true);
}

function clearScreenshotQueue() {
  if (state.screenshotQueue.length === 0) return;
  state.screenshotQueue = [];
  refreshQueueUI();
}

// Entry point from the screenshot-captured IPC. v0.3.2 — captures
// no longer fire the AI directly; they accumulate in the queue and
// wait for an explicit Submit click. No copy on disk, no console.log
// of the base64 (held only in state.screenshotQueue in memory).
async function handleScreenshot(base64) {
  if (!base64) return;
  enqueueScreenshot(base64);
}

// Pill tab click handlers (v0.3.0).
//
// Voice tab:
//   - Always switches the visible tab to Voice.
//   - If the overlay is currently in the idle pill state (no voice
//     session running AND no screen capture active), this is the same
//     as clicking Start — kick off a voice session so the panel opens
//     to the live voice view. If voice is already running OR a screen
//     capture is up, just switch tabs without restarting anything.
//
// Screen tab:
//   - Always switches the visible tab to Screen.
//   - Sets state.screenActive (if it wasn't already) so refreshLayout
//     opens the expanded panel even if no voice session is running.
//     The placeholder text inside #screen-answer-content tells the
//     user to press Ctrl+Shift+S to capture a question.
tabVoiceBtn.addEventListener('click', () => {
  switchTab('voice');
  if (!state.running && !state.screenActive) {
    startSession();   // same effect as clicking the Start button
  }
});

tabScreenBtn.addEventListener('click', () => {
  switchTab('screen');
  if (!state.screenActive) {
    state.screenActive = true;
    refreshLayout();
  }
});

// v0.3.1 — manual capture button in the pill bar. Fires the same IPC
// path the SCREEN_CAPTURE_HOTKEY uses, so the resulting flow is
// identical (handleScreenshot → startScreenSession → runVisionTurn).
if (btnCapture) {
  btnCapture.addEventListener('click', () => {
    if (window.electronAPI && window.electronAPI.triggerScreenCapture) {
      window.electronAPI.triggerScreenCapture();
    }
  });
}

// v0.3.1 — manual question input (top of Screen panel). Pressing Enter
// without Shift, or clicking Send, submits the typed text as a
// text-only vision query (no screenshot). Always switches to the
// Screen tab on submit since that's where the answer renders.
function submitManualScreenQuery() {
  const text = (screenInputEl && screenInputEl.value || '').trim();
  if (!text) return;
  screenInputEl.value = '';
  startScreenSession(text, null, true);
}
if (screenInputEl) {
  screenInputEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitManualScreenQuery();
    }
  });
}
if (btnScreenInputSend) {
  btnScreenInputSend.addEventListener('click', submitManualScreenQuery);
}

// v0.3.1 — feedback / follow-up input (bottom of Screen panel). Adds
// a user turn to the current session and reruns the vision call with
// full thread context. If somehow invoked with no session, falls
// through to a fresh text-only session (continueScreenSession handles
// the empty case internally).
function submitScreenFeedback() {
  const text = (screenFeedbackEl && screenFeedbackEl.value || '').trim();
  if (!text) return;
  screenFeedbackEl.value = '';
  continueScreenSession(text);
}
if (screenFeedbackEl) {
  screenFeedbackEl.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitScreenFeedback();
    }
  });
}
if (btnScreenFeedbackSend) {
  btnScreenFeedbackSend.addEventListener('click', submitScreenFeedback);
}

// v0.3.2 — Submit / Clear buttons for the multi-screenshot queue.
if (btnSubmitQueue) {
  btnSubmitQueue.addEventListener('click', submitQueuedScreenshots);
}
if (btnClearQueue) {
  btnClearQueue.addEventListener('click', clearScreenshotQueue);
}

// v0.3.2 — thumbnail removal via event delegation. Clicking the small
// × button on a thumbnail drops that specific screenshot from the
// queue without affecting the others.
if (screenThumbnails) {
  screenThumbnails.addEventListener('click', (e) => {
    const btn = e.target.closest('button[data-action="remove-thumb"]');
    if (!btn) return;
    const wrap = btn.closest('.screen-thumb');
    if (!wrap) return;
    const idx = parseInt(wrap.dataset.index, 10);
    if (Number.isFinite(idx) && idx >= 0 && idx < state.screenshotQueue.length) {
      state.screenshotQueue.splice(idx, 1);
      refreshQueueUI();
    }
  });
}

// Copy-button delegation for code blocks rendered into #screen-answer-content.
screenAnswerContent.addEventListener('click', (e) => {
  const btn = e.target.closest('.copy-btn');
  if (btn) copyCodeFromButton(btn);
});

// IPC subscriptions for screen capture, mirroring the onHotkeyMute pattern.
if (window.electronAPI && window.electronAPI.onScreenshotCaptured) {
  window.electronAPI.onScreenshotCaptured((base64) => {
    handleScreenshot(base64);
  });
}
if (window.electronAPI && window.electronAPI.onScreenshotError) {
  window.electronAPI.onScreenshotError((msg) => {
    state.screenActive = true;
    refreshLayout();
    if (storeGet(LS_KEY.autoSwitchTab)) switchTab('screen');
    screenStatus.textContent       = '';
    screenQuestionMeta.textContent = '';
    screenAnswerContent.textContent = '❌ ' + (msg || 'Screen capture failed.');
  });
}

// Global hotkey from main process
if (window.electronAPI && window.electronAPI.onHotkeyMute) {
  state.unhookMute = window.electronAPI.onHotkeyMute(() => {
    toggleMute();
  });
}

// Alt+9 — submit hotkey from main process. The renderer talks to main only
// through the electronAPI preload bridge (ipcRenderer is not exposed here),
// so this mirrors the onHotkeyMute pattern. Clicks the first available
// submit-style button: documented ids first, then the real ones that exist
// (Submit-queue, then the typed screen-query Send).
if (window.electronAPI && window.electronAPI.onHotkeySubmit) {
  state.unhookSubmit = window.electronAPI.onHotkeySubmit(() => {
    const btn = document.querySelector('#btn-send') ||
                document.querySelector('#btn-submit') ||
                document.querySelector('[data-action="submit"]') ||
                document.querySelector('#btn-submit-queue:not([disabled])') ||
                document.querySelector('#btn-screen-input-send:not([disabled])');
    if (btn) btn.click();
  });
}

// ---------- Boot ----------
// Async because the disk store has to hydrate the in-memory cache
// before any storeGet() in loadSettings / renderQaBank / renderDocList
// can return real values. While we wait, the pill UI already shows the
// idle state from its inline class, so there's no perceptible delay.
(async () => {
  try {
    await loadStore();
  } catch (err) {
    console.error('Store hydrate failed — falling back to defaults:', err);
  }
  loadSettings();
  renderQaBank();
  renderDocList();
  renderStorageBar();
  refreshQueueUI();   // v0.3.2 — initialize 📸 badge, Submit label, queue bar hidden
  setStatus('gray', 'Idle');
  refreshLayout();
})();
