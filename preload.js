const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  getDesktopSources: () => ipcRenderer.invoke('get-desktop-sources'),
  setWinHeight: (px) => ipcRenderer.send('set-win-height', px),
  setOverlay: (enable) => ipcRenderer.send('set-overlay', !!enable),
  onHotkeyMute: (callback) => {
    const listener = () => callback();
    ipcRenderer.on('hotkey-mute', listener);
    return () => ipcRenderer.removeListener('hotkey-mute', listener);
  },
  onHotkeySubmit: (callback) => {
    const listener = () => callback();
    ipcRenderer.on('hotkey-submit', listener);
    return () => ipcRenderer.removeListener('hotkey-submit', listener);
  },
  openPdfDialog: () => ipcRenderer.invoke('open-pdf-dialog'),
  parsePdf: (filePath) => ipcRenderer.invoke('parse-pdf', filePath),
  openTxtDialog: () => ipcRenderer.invoke('open-txt-dialog'),
  readTxt: (filePath) => ipcRenderer.invoke('read-txt', filePath),
  getStore: (key) => ipcRenderer.invoke('get-store', key),
  setStore: (key, value) => ipcRenderer.invoke('set-store', key, value),
  winMinimize: () => ipcRenderer.send('win-minimize'),
  winMaximize: () => ipcRenderer.send('win-maximize'),
  onMaximizeChanged: (callback) => {
    const listener = (_e, isMax) => callback(!!isMax);
    ipcRenderer.on('win-maximize-changed', listener);
    return () => ipcRenderer.removeListener('win-maximize-changed', listener);
  },
  // v0.3.0 — screen capture
  onScreenshotCaptured: (callback) => {
    const listener = (_e, base64) => callback(base64);
    ipcRenderer.on('screenshot-captured', listener);
    return () => ipcRenderer.removeListener('screenshot-captured', listener);
  },
  onScreenshotError: (callback) => {
    const listener = (_e, msg) => callback(msg);
    ipcRenderer.on('screenshot-error', listener);
    return () => ipcRenderer.removeListener('screenshot-error', listener);
  },
  // v0.3.1 — manually trigger the same capture flow as the hotkey
  // (used by the 📸 button in the pill bar).
  triggerScreenCapture: () => ipcRenderer.send('trigger-screen-capture')
});
