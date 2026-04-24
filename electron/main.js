const { app, BrowserWindow, shell, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');

// ── Config ────────────────────────────────────────────────────────────────────
const DAEMON_PORT = 8006;
const DEV_MODE = process.env.NODE_ENV === 'development';
const VITE_DEV_URL = 'http://localhost:3000';

let mainWindow = null;
let daemonProcess = null;

// ── Find the bundled Python daemon ────────────────────────────────────────────
function getDaemonPath() {
  if (DEV_MODE) {
    // In dev mode, use python directly
    return null;
  }
  // In packaged app, look for the bundled daemon executable
  const exeName = process.platform === 'win32' ? 'janus_daemon.exe' : 'janus_daemon';
  const candidates = [
    path.join(process.resourcesPath, 'daemon', exeName),
    path.join(app.getAppPath(), '..', 'daemon', exeName),
    path.join(__dirname, '..', 'daemon', exeName),
  ];
  for (const p of candidates) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

// ── Start the Python daemon ───────────────────────────────────────────────────
function startDaemon() {
  const daemonPath = getDaemonPath();

  if (DEV_MODE) {
    // Dev: run python directly
    console.log('[Electron] Starting daemon via python (dev mode)...');
    daemonProcess = spawn('python', ['janus_daemon.py', '--run'], {
      cwd: path.join(__dirname, '..'),
      stdio: 'pipe',
      windowsHide: true,
    });
  } else if (daemonPath) {
    console.log('[Electron] Starting bundled daemon:', daemonPath);
    daemonProcess = spawn(daemonPath, ['--run'], {
      cwd: path.dirname(daemonPath),
      stdio: 'pipe',
      windowsHide: true,
    });
  } else {
    console.warn('[Electron] Daemon executable not found — UI will run without backend');
    return;
  }

  daemonProcess.stdout?.on('data', (d) => console.log('[Daemon]', d.toString().trim()));
  daemonProcess.stderr?.on('data', (d) => console.error('[Daemon ERR]', d.toString().trim()));
  daemonProcess.on('exit', (code) => console.log('[Daemon] exited with code', code));
}

// ── Stop the daemon ───────────────────────────────────────────────────────────
function stopDaemon() {
  if (daemonProcess) {
    console.log('[Electron] Stopping daemon...');
    daemonProcess.kill();
    daemonProcess = null;
  }
}

// ── Wait for daemon to be ready ───────────────────────────────────────────────
async function waitForDaemon(maxWaitMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    try {
      const http = require('http');
      await new Promise((resolve, reject) => {
        const req = http.get(`http://localhost:${DAEMON_PORT}/health`, (res) => {
          resolve(res.statusCode === 200);
        });
        req.on('error', reject);
        req.setTimeout(1000, () => { req.destroy(); reject(new Error('timeout')); });
      });
      console.log('[Electron] Daemon is ready');
      return true;
    } catch {
      await new Promise(r => setTimeout(r, 500));
    }
  }
  console.warn('[Electron] Daemon did not respond in time — continuing anyway');
  return false;
}

// ── Create the main window ────────────────────────────────────────────────────
async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    title: 'Janus',
    backgroundColor: '#060d17',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    // Remove default menu bar
    autoHideMenuBar: true,
  });

  // Open external links in the system browser, not Electron
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  if (DEV_MODE) {
    mainWindow.loadURL(VITE_DEV_URL);
    mainWindow.webContents.openDevTools();
  } else {
    mainWindow.loadFile(path.join(__dirname, '..', 'dist', 'index.html'));
  }

  mainWindow.on('closed', () => { mainWindow = null; });
}

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  startDaemon();

  // Give daemon a moment to start before showing the window
  if (!DEV_MODE) {
    await waitForDaemon(10000);
  }

  await createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  stopDaemon();
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  stopDaemon();
});

// ── IPC: daemon status ────────────────────────────────────────────────────────
ipcMain.handle('daemon-status', () => {
  return { running: daemonProcess !== null && !daemonProcess.killed };
});
