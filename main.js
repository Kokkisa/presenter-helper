const { app, BrowserWindow, ipcMain, globalShortcut, desktopCapturer, dialog, screen } = require('electron');
const path = require('path');
const fs = require('fs').promises;
const pdfParse = require('pdf-parse');

// ─── CONFIGURABLE HOTKEYS ─────────────────────────────────────────────────
// To change the screen capture hotkey: edit ONLY this one line, save, restart.
const SCREEN_CAPTURE_HOTKEY = 'Alt+7';
// ──────────────────────────────────────────────────────────────────────────

// Pin the app name BEFORE any app.getPath('userData') call. This makes the
// userData directory the same in dev (`npm start`) and in the packaged
// installer (which builds with productName=LiveCallAssistant), and matches
// what setup.js / embed.js use — so offline data prep lands where the
// running app reads.
app.setName('Presenter Helper');

let mainWindow = null;

// v0.3.0 — HEIGHT_PILL grew from 64 to 100 to make room for the always-
// visible Voice/Screen pill tab switcher that now lives between the
// header pill and the (optional) expanded panel. Header is 56+8 margins
// = 64, pill-bar adds ~36, total ~100.
const HEIGHT_PILL = 100;
const HEIGHT_EXPANDED = 480;
const HEIGHT_SETTINGS = 480;

function createWindow() {
  mainWindow = new BrowserWindow({
    title: 'Presenter Helper',
    show: false,
    width: 540,
    height: HEIGHT_PILL,
    minWidth: 320,
    minHeight: HEIGHT_PILL,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    resizable: true,
    acceptFirstMouse: true,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  mainWindow.setContentProtection(true);
  mainWindow.setAlwaysOnTop(true, 'screen-saver');
  mainWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });

  // v0.3.1 — swallow the screen-capture hotkey at the renderer level too.
  // globalShortcut intercepts at the OS-level when no app is hooking the
  // raw keystream, but if the overlay itself happens to be focused (e.g.
  // after the user clicked a text input inside it), the keystroke would
  // otherwise echo into the renderer as a regular key event. This handler
  // preventDefault's it on the renderer side so it never reaches input
  // boxes or any focused element.
  mainWindow.webContents.on('before-input-event', (event, input) => {
    if (input.type === 'keyDown' && input.control && input.shift && input.key && input.key.toUpperCase() === 'S') {
      event.preventDefault();
    }
  });

  if (process.platform === 'darwin' && app.dock) app.dock.hide();

  // Forward renderer console messages to main process stdout for debugging
  mainWindow.webContents.on('console-message', (_e, level, message, line, sourceId) => {
    // Electron levels: 0=verbose, 1=info, 2=warning, 3=error
    const tag = ['verbose', 'info', 'warn', 'error'][level] || 'log';
    process.stdout.write(`[renderer:${tag}] ${message}  (${sourceId}:${line})\n`);
  });

  // Surface maximize state to the renderer so the pill's chrome button
  // can flip between maximize/restore. Double-clicking the drag region
  // also fires these, so the icon stays in sync without us having to
  // intercept those events.
  mainWindow.on('maximize',   () => mainWindow.webContents.send('win-maximize-changed', true));
  mainWindow.on('unmaximize', () => mainWindow.webContents.send('win-maximize-changed', false));

  mainWindow.loadFile('index.html');
}

ipcMain.handle('get-desktop-sources', async () => {
  const sources = await desktopCapturer.getSources({ types: ['screen'] });
  return sources.map(s => ({ id: s.id, name: s.name }));
});

ipcMain.on('set-win-height', (_event, px) => {
  if (!mainWindow) return;
  const allowed = [HEIGHT_PILL, HEIGHT_EXPANDED, HEIGHT_SETTINGS];
  const target = allowed.includes(px) ? px : Math.max(HEIGHT_PILL, Math.min(720, Number(px) || HEIGHT_PILL));
  const [w] = mainWindow.getSize();
  mainWindow.setSize(w, target, false);
});

ipcMain.on('set-overlay', (_event, enable) => {
  if (!mainWindow) return;
  if (enable) {
    mainWindow.setIgnoreMouseEvents(true, { forward: true });
  } else {
    mainWindow.setIgnoreMouseEvents(false);
  }
});

// v0.3.1 — manual screen capture trigger (used by the 📸 button in the
// pill bar). Mirrors the SCREEN_CAPTURE_HOTKEY's effect: runs
// captureScreen() which sends 'screenshot-captured' back to the renderer.
ipcMain.on('trigger-screen-capture', () => {
  captureScreen();
});

// ---- Window chrome (pill has no native frame; we wire our own) ----
ipcMain.on('win-minimize', () => {
  if (mainWindow) mainWindow.minimize();
});

ipcMain.on('win-maximize', () => {
  if (!mainWindow) return;
  if (mainWindow.isMaximized()) mainWindow.unmaximize();
  else mainWindow.maximize();
});

// ---- PDF upload + parsing (used by Tab 2 Documents) ----------------
// The dialog runs in main; pdf-parse needs Node fs + Buffer so it also
// runs in main. Renderer just gets the extracted text back.

ipcMain.handle('open-pdf-dialog', async () => {
  if (!mainWindow) return [];
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select PDF documents',
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'PDF', extensions: ['pdf'] }]
  });
  if (result.canceled) return [];
  return result.filePaths;
});

ipcMain.handle('parse-pdf', async (_event, filePath) => {
  if (typeof filePath !== 'string' || !filePath.toLowerCase().endsWith('.pdf')) {
    return { error: 'Not a PDF file' };
  }
  try {
    const buf = await fs.readFile(filePath);
    const data = await pdfParse(buf);
    return {
      filename: path.basename(filePath),
      text: data.text || '',
      pages: data.numpages || 0,
      size: buf.length
    };
  } catch (err) {
    return { error: err.message || String(err) };
  }
});

// ---- .txt upload + parsing (used by Tab 3 Q&A Bank bulk import) -----

ipcMain.handle('open-txt-dialog', async () => {
  if (!mainWindow) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: 'Select Q&A text file',
    properties: ['openFile'],
    filters: [{ name: 'Text', extensions: ['txt'] }]
  });
  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle('read-txt', async (_event, filePath) => {
  if (typeof filePath !== 'string' || !filePath.toLowerCase().endsWith('.txt')) {
    return { error: 'Not a .txt file' };
  }
  try {
    const text = await fs.readFile(filePath, 'utf8');
    return {
      filename: path.basename(filePath),
      text,
      size: Buffer.byteLength(text, 'utf8')
    };
  } catch (err) {
    return { error: err.message || String(err) };
  }
});

// =====================================================================
// Disk-backed store
//
// Replaces localStorage. Renderer talks via two IPCs — get-store(key) and
// set-store(key, value). Keys map onto four JSON files in userData:
//   settings.json — ca_key, ca_claude_key, ca_groq_key, ca_provider, ca_model, ca_threshold
//   tier1.json    — ca_tier1 (array)
//   tier2.json    — ca_tier2 (array)
//   chunks.json   — ca_doc_chunks (array)
// Settings is the only multi-key file (one object with five scalar keys);
// the others are bare arrays so they round-trip cleanly with the offline
// setup.js / embed.js scripts.
//
// Writes are atomic (tmp + rename) and serialized through a single Promise
// chain so two near-simultaneous writes to the same file can't interleave.
// Reads always come straight off disk — the renderer maintains its own
// in-memory cache for synchronous access.
// =====================================================================

const KEY_TO_FILE = {
  ca_key:              'settings',
  ca_claude_key:       'settings',
  ca_groq_key:         'settings',  // Groq Whisper transcription key
  ca_provider:         'settings',
  ca_model:            'settings',
  ca_threshold:        'settings',
  ca_coding_lang:      'settings',  // v0.3.0 — Coding Mode language preference
  ca_auto_switch_tab:  'settings',  // v0.3.0 — auto-switch to Screen tab on capture
  ca_vision_model:     'settings',  // v0.3.0 — vision API model
  ca_tier1:            'tier1',
  ca_tier2:            'tier2',
  ca_doc_chunks:       'chunks'
};

const STORE_DEFAULT = {
  ca_key:              '',
  ca_claude_key:       '',
  ca_groq_key:         '',
  ca_provider:         'openai',
  ca_model:            '',
  ca_threshold:        0.82,
  ca_coding_lang:      'Python',
  ca_auto_switch_tab:  true,
  ca_vision_model:     'gpt-4o',
  ca_tier1:            [],
  ca_tier2:            [],
  ca_doc_chunks:       []
};

function storePath(name) {
  return path.join(app.getPath('userData'), name + '.json');
}

async function ensureUserData() {
  await fs.mkdir(app.getPath('userData'), { recursive: true });
}

async function readJsonOrNull(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch (err) {
    if (err.code === 'ENOENT') return null;
    throw err;
  }
}

async function writeJsonAtomic(filePath, data) {
  const tmp = filePath + '.tmp';
  await fs.writeFile(tmp, JSON.stringify(data), 'utf8');
  await fs.rename(tmp, filePath);
}

// Single global write chain. Serializes every set-store so concurrent
// writes to the same file don't read-modify-write past each other (the
// settings file is the only multi-key file but the chain costs nothing
// for the array files either).
let writeChain = Promise.resolve();
function serializeWrite(op) {
  const next = writeChain.then(op, op);
  writeChain = next.catch(() => {});
  return next;
}

ipcMain.handle('get-store', async (_event, key) => {
  const file = KEY_TO_FILE[key];
  if (!file) return null;
  const data = await readJsonOrNull(storePath(file));
  if (file === 'settings') {
    if (data && Object.prototype.hasOwnProperty.call(data, key)) return data[key];
    return STORE_DEFAULT[key];
  }
  // Array files: the file content IS the value of the single key.
  return data === null ? STORE_DEFAULT[key] : data;
});

ipcMain.handle('set-store', async (_event, key, value) => {
  const file = KEY_TO_FILE[key];
  if (!file) throw new Error('Unknown store key: ' + key);
  return serializeWrite(async () => {
    await ensureUserData();
    if (file === 'settings') {
      const cur = (await readJsonOrNull(storePath(file))) || {};
      cur[key] = value;
      await writeJsonAtomic(storePath(file), cur);
    } else {
      await writeJsonAtomic(storePath(file), value);
    }
    return true;
  });
});

// =====================================================================
// Screen capture (v0.3.0 — Coding Mode)
//
// Stealth requirements:
//   - Overlay itself is excluded via setContentProtection(true) (already
//     set in createWindow), so it won't appear in the captured frame.
//   - Screenshot bytes never touch disk — JPEG is produced in memory,
//     base64-encoded, and shipped to the renderer for an API call.
//   - We send the JPEG over IPC and leave the in-process Buffer to the
//     GC after the send (Electron's structured-clone copies it across).
// =====================================================================
async function captureScreen() {
  if (!mainWindow) return;
  try {
    const primaryDisplay = screen.getPrimaryDisplay();
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: {
        width:  primaryDisplay.size.width,
        height: primaryDisplay.size.height
      }
    });

    if (!sources || sources.length === 0) {
      mainWindow.webContents.send('screenshot-error', 'No screen sources available.');
      return;
    }

    // On multi-monitor setups desktopCapturer doesn't guarantee primary
    // is sources[0] — match by display_id when available.
    let primarySource = sources[0];
    const targetId = String(primaryDisplay.id);
    for (const s of sources) {
      if (s.display_id === targetId) { primarySource = s; break; }
    }

    const screenshot = primarySource.thumbnail; // NativeImage
    // JPEG 85 keeps file size manageable for the vision API while
    // preserving text legibility. PNG would be lossless but ~3-5x bigger
    // and the model only needs to read pixels, not pixel-perfect copies.
    const base64 = screenshot.toJPEG(85).toString('base64');
    mainWindow.webContents.send('screenshot-captured', base64);
  } catch (err) {
    mainWindow.webContents.send('screenshot-error', err && err.message ? err.message : String(err));
  }
}

function registerShortcuts() {
  globalShortcut.register('Alt+1', () => {
    if (!mainWindow) return;
    if (mainWindow.isVisible()) mainWindow.hide();
    else mainWindow.show();
  });

  globalShortcut.register('Alt+0', () => {
    if (!mainWindow) return;
    mainWindow.webContents.send('hotkey-mute');
  });

  globalShortcut.register('Alt+9', () => {
    if (mainWindow) mainWindow.webContents.send('hotkey-submit');
  });

  // v0.3.0 screen-capture hotkey. Guard against another app holding it
  // already — register only if free, else surface a warning in the log.
  if (!globalShortcut.isRegistered(SCREEN_CAPTURE_HOTKEY)) {
    globalShortcut.register(SCREEN_CAPTURE_HOTKEY, captureScreen);
  } else {
    console.warn(`Hotkey ${SCREEN_CAPTURE_HOTKEY} already registered by another app — screen capture will not fire.`);
  }
}

app.whenReady().then(() => {
  createWindow();
  registerShortcuts();
});

app.on('will-quit', () => {
  globalShortcut.unregisterAll();
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
