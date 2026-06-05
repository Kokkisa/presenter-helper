# CLAUDE.md — LiveCallAssistant v0.3.0

## ⚠️ Safety First — Read Before Any Code Change

v0.2.0 must be tagged and pushed to Git before ANY v0.3.0 work begins.
If git tag v0.2.0 is not yet pushed, stop and do this first:

```bash
git add .
git commit -m "v0.2.0 stable — three-tier system, PDF RAG, Claude API, semantic cache"
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

Recovery from any crash: `git checkout v0.2.0 && npm install && npm start`

## Context

This is a continuation of v0.2.0. The app is already built and working.
Before writing a single line of code, read ALL existing source files thoroughly.

Repo: https://github.com/Kokkisa/livecallassistant (private)
Current state: v0.2.0 tagged, app runs with `npm start`

### v0.1.0 recap
- Electron transparent overlay, always on top, hidden from screen share
- System audio capture via desktopCapturer (Windows WASAPI loopback)
- Auto VAD — RMS threshold detects speech end, no manual trigger needed
- Whisper transcription → GPT-4o streaming answer
- Conversation history (last 8 messages for GPT context)
- Conversational tone — first person, ~120 words, plain text, no filler openers
- Settings: API key, background context, model — persisted in localStorage

### v0.2.0 recap
- Three-tier answer system: Tier 1 (Q&A bank) → Tier 2 (semantic cache) → Tier 3 (RAG + AI)
- PDF document upload with RAG over real documents
- Embedding-based semantic matching for Q&A and cache
- Claude API as alternate LLM provider alongside OpenAI
- Source badges: Q&A / Cached / AI
- Sensitivity slider for matching threshold
- Desktop shortcut via electron-builder
- All data persists in localStorage across restarts

DO NOT refactor or rewrite v0.1.0 or v0.2.0 code unless a bug forces it.
Build v0.3.0 features cleanly on top of what exists.

---

## What v0.3.0 adds

1. **Screen Capture** — hotkey-triggered screenshot of the active screen, stealth mode (excluded from screen share)
2. **Coding Mode** — dedicated answer mode for coding/DSA/SQL/system-design questions detected from screenshot
3. **Vision Pipeline** — screenshot sent to GPT-4o Vision or Claude Vision to extract the question, then answered in code
4. **Answer Tabs** — within the overlay, toggle between "Voice Answer" and "Screen Answer" tabs
5. **Code Display** — syntax-highlighted code block in the overlay, copyable with one click

---

## Feature 1: Screen Capture (Stealth)

### How it works
- User presses a hotkey (default: `Ctrl+Shift+S`) while looking at a coding question
- The app captures a screenshot of the PRIMARY screen only (not the overlay window itself)
- The screenshot is processed locally — NEVER saved to disk, NEVER sent anywhere except the AI API
- The overlay stays hidden from screen share throughout — no flicker, no reveal

### Hotkey is a single constant — easy to change later

Define the hotkey ONCE at the very top of main.js, before any other code:

```javascript
// ─── CONFIGURABLE HOTKEYS ─────────────────────────────────────────────────
// To change the screen capture hotkey: edit ONLY this one line, save, restart.
const SCREEN_CAPTURE_HOTKEY = 'CommandOrControl+Shift+S';
// ──────────────────────────────────────────────────────────────────────────
```

Use `SCREEN_CAPTURE_HOTKEY` everywhere in main.js — never hardcode the string again.
Changing the hotkey in the future = change one line, restart app. Nothing else touched.

### Implementation — main process (main.js)

```javascript
// Register global hotkey for screen capture
const { globalShortcut, desktopCapturer, screen } = require('electron');

globalShortcut.register(SCREEN_CAPTURE_HOTKEY, async () => {
  const primaryDisplay = screen.getPrimaryDisplay();
  const sources = await desktopCapturer.getSources({
    types: ['screen'],
    thumbnailSize: {
      width: primaryDisplay.size.width,
      height: primaryDisplay.size.height
    }
  });

  // Take screenshot of primary screen only
  const primarySource = sources[0];
  const screenshot = primarySource.thumbnail; // NativeImage

  // Convert to base64 for sending to renderer
  const base64 = screenshot.toJPEG(85).toString('base64');

  // Send to renderer process
  mainWindow.webContents.send('screenshot-captured', base64);
});
```

### Critical: Exclude overlay from screenshot
The overlay window must be excluded from the screenshot so it doesn't appear in the capture.
This is done via `setContentProtection(true)` on the overlay window — already set in v0.1.0.
The desktopCapturer will skip windows with content protection enabled.
Verify: `overlayWindow.setContentProtection(true)` is present in main.js. If not, add it.

### Hotkey conflict check
Before registering, check if it's already taken:
```javascript
if (!globalShortcut.isRegistered(SCREEN_CAPTURE_HOTKEY)) {
  globalShortcut.register(SCREEN_CAPTURE_HOTKEY, captureScreen);
} else {
  console.warn(`Hotkey ${SCREEN_CAPTURE_HOTKEY} already registered by another app.`);
}
```

---

## Feature 2: Vision Pipeline — Extract Question from Screenshot

After screenshot is captured and sent to renderer via IPC, the renderer does:

### Step 1: Show "Analyzing screen..." state in overlay
- Switch overlay to a "Screen" tab (see Feature 4)
- Show a spinner with text "Reading your screen..."

### Step 2: Send to Vision API

```javascript
async function analyzeScreenshot(base64Image) {
  const provider = getSetting('provider'); // 'openai' or 'claude'

  if (provider === 'openai') {
    return await analyzeWithOpenAI(base64Image);
  } else {
    return await analyzeWithClaude(base64Image);
  }
}

async function analyzeWithOpenAI(base64Image) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${getSetting('openaiKey')}`
    },
    body: JSON.stringify({
      model: 'gpt-4o',
      messages: [
        {
          role: 'system',
          content: `You are an expert software engineer and interview coach.
The user has shared a screenshot of a coding/technical question from their screen.

Your job:
1. Extract the EXACT question text from the screenshot (copy it precisely)
2. Identify the question type: DSA | SQL | System Design | Behavioral | General Coding
3. Provide a complete, working solution

Answer format:
- Start with the question type on line 1: TYPE: [DSA/SQL/System Design/etc]
- Then QUESTION: [exact question text]
- Then SOLUTION: [your complete answer]

For DSA/SQL/General Coding — provide working code with comments.
For System Design — provide a structured breakdown (components, data flow, trade-offs).
For Behavioral — provide a STAR-format answer.

Use the background context below if provided:
${getSetting('backgroundContext') || 'No background context set.'}`
        },
        {
          role: 'user',
          content: [
            {
              type: 'image_url',
              image_url: {
                url: `data:image/jpeg;base64,${base64Image}`,
                detail: 'high'
              }
            },
            {
              type: 'text',
              text: 'Extract the question from this screenshot and solve it completely.'
            }
          ]
        }
      ],
      max_tokens: 1500,
      stream: true
    })
  });
  return response; // streaming response
}

async function analyzeWithClaude(base64Image) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'x-api-key': getSetting('claudeKey'),
      'anthropic-version': '2023-06-01'
    },
    body: JSON.stringify({
      model: 'claude-opus-4-5',
      max_tokens: 1500,
      stream: true,
      system: `You are an expert software engineer and interview coach.
Extract the coding/technical question from the screenshot and provide a complete solution.

Format:
TYPE: [DSA/SQL/System Design/General Coding/Behavioral]
QUESTION: [exact question text]
SOLUTION: [complete working answer with code and comments]

Background context: ${getSetting('backgroundContext') || 'None provided.'}`,
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/jpeg',
                data: base64Image
              }
            },
            {
              type: 'text',
              text: 'Extract the question from this screenshot and solve it completely.'
            }
          ]
        }
      ]
    })
  });
  return response;
}
```

---

## Feature 3: Coding Mode Answer Display

### Answer parsing
After streaming completes, parse the structured response:

```javascript
function parseCodeAnswer(rawText) {
  const lines = rawText.split('\n');
  let type = '', question = '', solution = '';
  let inSolution = false;

  for (const line of lines) {
    if (line.startsWith('TYPE:')) type = line.replace('TYPE:', '').trim();
    else if (line.startsWith('QUESTION:')) question = line.replace('QUESTION:', '').trim();
    else if (line.startsWith('SOLUTION:')) { inSolution = true; continue; }
    else if (inSolution) solution += line + '\n';
  }

  return { type, question, solution: solution.trim() };
}
```

### Code syntax highlighting
Use **Prism.js** (already available via CDN, no install needed):

```html
<!-- In renderer HTML head -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-sql.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-java.min.js"></script>
```

```javascript
function renderCodeBlock(solutionText) {
  // Detect language from solution
  const lang = detectLanguage(solutionText);

  // Extract code from markdown fences if present
  const codeMatch = solutionText.match(/```[\w]*\n([\s\S]*?)```/);
  const code = codeMatch ? codeMatch[1] : solutionText;

  return `
    <div class="code-block-wrapper">
      <div class="code-header">
        <span class="lang-badge">${lang}</span>
        <button class="copy-btn" onclick="copyCode(this)">Copy</button>
      </div>
      <pre class="code-pre"><code class="language-${lang}">${escapeHtml(code)}</code></pre>
    </div>
  `;
}

function detectLanguage(text) {
  if (/def |import |print\(|:\n/.test(text)) return 'python';
  if (/SELECT|FROM|WHERE|JOIN/i.test(text)) return 'sql';
  if (/function |const |let |var |=>/.test(text)) return 'javascript';
  if (/public class|void |System\.out/.test(text)) return 'java';
  return 'python'; // default
}

function copyCode(btn) {
  const code = btn.closest('.code-block-wrapper').querySelector('code').innerText;
  navigator.clipboard.writeText(code);
  btn.textContent = 'Copied!';
  setTimeout(() => btn.textContent = 'Copy', 2000);
}
```

---

## Feature 4: Pill Tab UI — Voice & Screen

### Critical rule: Voice tab code is UNTOUCHED
All existing voice answer HTML, JS, and CSS from v0.1.0 and v0.2.0 stays exactly where it is.
The ONLY change is wrapping the existing voice panel content inside a `<div id="panel-voice">` wrapper.
Do not move, reorder, or touch any voice-related logic.

### Pill tab design

```
┌──────────────────────────────────────────┐
│   🎙 Voice      🖥 Screen                │  ← pill switcher
│  ───────────────────────────────────── │
│   [active panel content]                 │
└──────────────────────────────────────────┘
```

Two ways to switch tabs:
1. **Click the pill** — instant switch, no hotkey needed
2. **Press the hotkey** — captures screen AND auto-switches to Screen tab

### Tab HTML (add to existing overlay HTML — DO NOT restructure existing content)

```html
<!-- Add this pill bar ABOVE the existing answer content -->
<div class="pill-bar">
  <div class="pill-wrapper">
    <button class="pill-btn active" id="tab-voice" onclick="switchTab('voice')">🎙 Voice</button>
    <button class="pill-btn" id="tab-screen" onclick="switchTab('screen')">🖥 Screen</button>
  </div>
</div>

<!-- Wrap existing voice content in this div — content itself unchanged -->
<div id="panel-voice" class="tab-panel active">
  <!-- ALL existing voice answer HTML goes here — zero changes to its content -->
</div>

<!-- New screen panel — completely separate -->
<div id="panel-screen" class="tab-panel">
  <div id="screen-status"></div>
  <div id="screen-question-meta"></div>
  <div id="screen-answer-content"></div>
</div>
```

```javascript
function switchTab(tab) {
  document.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  document.getElementById(`panel-${tab}`).classList.add('active');
}
```

### Pill CSS (match existing overlay dark theme)

```css
.pill-bar {
  display: flex;
  justify-content: center;
  padding: 8px 8px 4px;
}
.pill-wrapper {
  display: flex;
  background: rgba(255,255,255,0.08);
  border-radius: 20px;
  padding: 3px;
  gap: 2px;
}
.pill-btn {
  background: transparent;
  border: none;
  color: rgba(255,255,255,0.5);
  font-size: 12px;
  padding: 4px 14px;
  border-radius: 16px;
  cursor: pointer;
  transition: all 0.18s ease;
  white-space: nowrap;
}
.pill-btn.active {
  background: rgba(79,142,247,0.85);
  color: #fff;
  font-weight: 500;
}
.pill-btn:hover:not(.active) {
  color: rgba(255,255,255,0.8);
  background: rgba(255,255,255,0.1);
}
.tab-panel { display: none; }
.tab-panel.active { display: block; }
```

---

## IPC wiring — main.js ↔ renderer.js

In main.js, after screenshot is captured:
```javascript
mainWindow.webContents.send('screenshot-captured', base64);
```

In renderer.js, listen for it:
```javascript
const { ipcRenderer } = require('electron');

ipcRenderer.on('screenshot-captured', async (event, base64) => {
  switchTab('screen');
  document.getElementById('screen-status').textContent = '🔍 Reading screen...';
  document.getElementById('screen-answer-content').innerHTML = '';
  document.getElementById('screen-question-meta').textContent = '';

  try {
    const stream = await analyzeScreenshot(base64);
    let fullText = '';

    // Stream the response
    const reader = stream.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      // Parse SSE chunks (same as existing voice streaming logic)
      const lines = chunk.split('\n').filter(l => l.startsWith('data: '));
      for (const line of lines) {
        const data = line.replace('data: ', '');
        if (data === '[DONE]') continue;
        try {
          const parsed = JSON.parse(data);
          // OpenAI format:
          const delta = parsed.choices?.[0]?.delta?.content || '';
          // Claude format fallback:
          // const delta = parsed.delta?.text || '';
          fullText += delta;
          renderStreamingAnswer(fullText);
        } catch {}
      }
    }

    // Final render after stream completes
    const parsed = parseCodeAnswer(fullText);
    document.getElementById('screen-status').textContent = '';
    document.getElementById('screen-question-meta').textContent =
      `${parsed.type} • Captured just now`;
    document.getElementById('screen-answer-content').innerHTML =
      renderCodeBlock(parsed.solution);

    // Re-run Prism highlighting
    Prism.highlightAll();

  } catch (err) {
    document.getElementById('screen-status').textContent = `❌ Error: ${err.message}`;
  }
});

function renderStreamingAnswer(text) {
  // During streaming, show raw text. Final render applies code highlighting.
  document.getElementById('screen-answer-content').textContent = text;
}
```

---

## Settings additions (v0.3.0)

Add to the existing Settings panel:

| Setting | Type | Default | Description |
|---|---|---|---|
| Screen Capture Hotkey | Text input (display only) | Ctrl+Shift+S | Shows user the registered hotkey — not editable in v0.3.0 |
| Coding Language Preference | Dropdown | Python | Hints the AI to prefer this language when multiple are valid |
| Auto-switch to Screen tab | Toggle | ON | Automatically switches tab on screenshot capture |
| Vision Model | Dropdown | gpt-4o | gpt-4o \| claude-opus-4-5 |

Store new settings in localStorage under keys: `codingLang`, `autoSwitchTab`, `visionModel`.

In the system prompt for vision calls, inject language preference:
```
Preferred coding language: ${getSetting('codingLang') || 'Python'}
Use this language for all code unless the question specifies otherwise.
```

---

## Edge cases to handle

| Scenario | Handling |
|---|---|
| Screenshot captures no text (blank screen, image-only) | Show: "No question detected — try again on a screen with text" |
| API key missing | Show: "Set your API key in Settings before using Screen Capture" |
| Network error during vision call | Show: "Network error — check connection and try again" |
| Hotkey conflicts with another app | Log warning to console, notify user in settings: "Hotkey may be in use by another app" |
| Very long question (>500 words) | Truncate to first 500 words before sending, prepend: "Question (truncated): ..." |
| User captures overlay itself | Overlay is protected via setContentProtection — will appear blank in capture — safe |
| Question is not in English | Model handles multilingual by default — no special handling needed |

---

## Stealth Checklist (verify before shipping)

- [ ] Overlay window has `setContentProtection(true)` set in main.js
- [ ] desktopCapturer excludes overlay window (verified by content protection)
- [ ] Hotkey `Ctrl+Shift+S` fires silently — no system sound, no taskbar flash
- [ ] Screenshot base64 is NEVER written to disk — only held in memory and sent to API
- [ ] base64 string is cleared from memory after API call completes (set variable to null)
- [ ] No console.log prints the base64 string (would expose screenshot data in DevTools)
- [ ] API call uses HTTPS only — no plain HTTP fallback
- [ ] Overlay does not resize or move during screenshot capture

---

## Definition of done (v0.3.0)

- [ ] `Ctrl+Shift+S` captures primary screen without revealing overlay
- [ ] Screenshot sent to vision model (GPT-4o or Claude vision, based on settings)
- [ ] Question type detected and displayed as meta badge (DSA / SQL / System Design / etc.)
- [ ] Solution streamed and rendered in Screen tab with syntax highlighting
- [ ] Code copy button works with one click
- [ ] Voice tab and existing v0.2.0 behavior completely unaffected
- [ ] Language preference from settings injected into vision prompt
- [ ] All edge cases handled gracefully (no crashes, clear error messages)
- [ ] No screenshot data written to disk
- [ ] Hotkey cleaned up on app quit (`globalShortcut.unregisterAll()` in `app.on('will-quit')`)
- [ ] `SCREEN_CAPTURE_HOTKEY` constant used everywhere — no hardcoded strings
- [ ] Tagged v0.3.0 pushed to Kokkisa/livecallassistant

---

## Files to modify

| File | Change |
|---|---|
| `main.js` | Add globalShortcut registration, desktopCapturer call, IPC send |
| `renderer.js` | Add IPC listener, vision API calls, streaming handler, tab switch logic |
| `index.html` | Add tab bar HTML, screen panel HTML, Prism.js CDN links |
| `styles.css` | Add tab styles, code block styles |
| `package.json` | No new dependencies — Prism.js via CDN, all Electron APIs already available |

DO NOT create new files unless absolutely necessary.
Keep all changes within the 4 existing source files.
