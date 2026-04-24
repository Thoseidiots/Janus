const { contextBridge, ipcRenderer } = require('electron');

// Expose a safe API to the renderer process
contextBridge.exposeInMainWorld('electronAPI', {
  getDaemonStatus: () => ipcRenderer.invoke('daemon-status'),
  isElectron: true,
});
