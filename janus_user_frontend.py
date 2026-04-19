"""
janus_user_frontend.py — Janus Network user-facing frontend server.

Serves a single-page application that lets users earn JC by contributing
compute, spend JC on tasks, view transaction history, and monitor network
statistics. Communicates with the JC API running on port 8004.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# ---------------------------------------------------------------------------
# HTML single-page application
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Janus Network</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:        #060d17;
      --surface:   #0d1b2a;
      --surface2:  #112236;
      --border:    #1e3a5f;
      --accent:    #00d4ff;
      --accent2:   #7b61ff;
      --success:   #00ff88;
      --warning:   #ffaa00;
      --danger:    #ff4466;
      --text:      #c8d8e8;
      --text-dim:  #5a7a9a;
      --font:      'Space Mono', monospace;
    }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      line-height: 1.6;
      min-height: 100vh;
    }

    /* ── Header ── */
    header {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 0 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      height: 60px;
      position: sticky;
      top: 0;
      z-index: 100;
    }

    .logo {
      font-size: 18px;
      font-weight: 700;
      letter-spacing: 4px;
      color: var(--accent);
      text-transform: uppercase;
    }

    .balance-pill {
      background: var(--surface2);
      border: 1px solid var(--accent);
      border-radius: 20px;
      padding: 6px 16px;
      font-size: 13px;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .balance-pill .label { color: var(--text-dim); }
    .balance-pill .value { color: var(--accent); font-weight: 700; }

    /* ── Layout ── */
    main {
      max-width: 1100px;
      margin: 0 auto;
      padding: 32px 16px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }

    @media (max-width: 720px) {
      main { grid-template-columns: 1fr; }
    }

    .full-width { grid-column: 1 / -1; }

    /* ── Cards ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 24px;
    }

    .card-title {
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: var(--accent);
      margin-bottom: 20px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }

    /* ── Forms ── */
    label {
      display: block;
      font-size: 11px;
      letter-spacing: 1px;
      color: var(--text-dim);
      margin-bottom: 6px;
      text-transform: uppercase;
    }

    input, textarea, select {
      width: 100%;
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--text);
      font-family: var(--font);
      font-size: 13px;
      padding: 10px 12px;
      margin-bottom: 16px;
      outline: none;
      transition: border-color 0.2s;
    }

    input:focus, textarea:focus, select:focus {
      border-color: var(--accent);
    }

    textarea { resize: vertical; min-height: 80px; }

    /* ── Buttons ── */
    .btn {
      display: inline-block;
      padding: 10px 20px;
      border: none;
      border-radius: 4px;
      font-family: var(--font);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }

    .btn:hover  { opacity: 0.85; }
    .btn:active { transform: scale(0.97); }

    .btn-primary  { background: var(--accent);  color: var(--bg); }
    .btn-secondary { background: var(--accent2); color: #fff; }
    .btn-success  { background: var(--success);  color: var(--bg); }
    .btn-full     { width: 100%; }

    /* ── Status messages ── */
    .msg {
      margin-top: 12px;
      padding: 10px 14px;
      border-radius: 4px;
      font-size: 12px;
      display: none;
    }

    .msg.ok    { background: rgba(0,255,136,0.1); border: 1px solid var(--success); color: var(--success); display: block; }
    .msg.err   { background: rgba(255,68,102,0.1); border: 1px solid var(--danger);  color: var(--danger);  display: block; }
    .msg.info  { background: rgba(0,212,255,0.1);  border: 1px solid var(--accent);  color: var(--accent);  display: block; }

    /* ── Stats grid ── */
    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
    }

    .stat-box {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
      text-align: center;
    }

    .stat-box .stat-val {
      font-size: 22px;
      font-weight: 700;
      color: var(--accent);
      display: block;
    }

    .stat-box .stat-lbl {
      font-size: 10px;
      letter-spacing: 2px;
      color: var(--text-dim);
      text-transform: uppercase;
      margin-top: 4px;
    }

    /* ── Task / transaction lists ── */
    .list-item {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px 16px;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      flex-wrap: wrap;
    }

    .list-item .item-main { flex: 1; min-width: 0; }
    .list-item .item-desc { font-size: 13px; word-break: break-word; }
    .list-item .item-meta { font-size: 11px; color: var(--text-dim); margin-top: 4px; }

    .badge {
      font-size: 10px;
      font-weight: 700;
      letter-spacing: 1px;
      padding: 3px 8px;
      border-radius: 10px;
      text-transform: uppercase;
      white-space: nowrap;
    }

    .badge-open      { background: rgba(0,255,136,0.15); color: var(--success); border: 1px solid var(--success); }
    .badge-claimed   { background: rgba(255,170,0,0.15);  color: var(--warning); border: 1px solid var(--warning); }
    .badge-completed { background: rgba(0,212,255,0.15);  color: var(--accent);  border: 1px solid var(--accent); }
    .badge-earn      { background: rgba(0,255,136,0.15); color: var(--success); border: 1px solid var(--success); }
    .badge-spend     { background: rgba(255,68,102,0.15); color: var(--danger);  border: 1px solid var(--danger); }
    .badge-transfer  { background: rgba(123,97,255,0.15); color: var(--accent2); border: 1px solid var(--accent2); }
    .badge-welcome   { background: rgba(0,212,255,0.15);  color: var(--accent);  border: 1px solid var(--accent); }

    .amount-pos { color: var(--success); font-weight: 700; }
    .amount-neg { color: var(--danger);  font-weight: 700; }

    .empty-state {
      text-align: center;
      color: var(--text-dim);
      padding: 24px;
      font-size: 12px;
      letter-spacing: 1px;
    }

    /* ── Compute info box ── */
    .compute-info {
      background: var(--surface2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 16px;
      margin-bottom: 20px;
      font-size: 12px;
      color: var(--text-dim);
      line-height: 1.8;
    }

    .compute-info span { color: var(--accent); font-weight: 700; }

    /* ── User ID bar ── */
    .user-bar {
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 10px 24px;
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 12px;
    }

    .user-bar label { margin: 0; }
    .user-bar input { margin: 0; width: 220px; }
    .user-bar .btn  { padding: 8px 16px; }
  </style>
</head>
<body>

<!-- Header -->
<header>
  <div class="logo">&#9670; Janus Network</div>
  <div class="balance-pill">
    <span class="label">JC Balance</span>
    <span class="value" id="header-balance">—</span>
  </div>
</header>

<!-- User identity bar -->
<div class="user-bar">
  <label for="global-user-id">User ID:</label>
  <input type="text" id="global-user-id" placeholder="your-username" value="demo_user" />
  <button class="btn btn-primary" onclick="loadAll()">Load</button>
</div>

<main>

  <!-- ── Earn JC ── -->
  <div class="card">
    <div class="card-title">&#9650; Earn JC — Contribute Compute</div>
    <div class="compute-info">
      Contribute compute to the Janus network and earn JC automatically.<br/>
      Rates: <span>Inference 0.1 JC/unit</span> &nbsp;|&nbsp;
             <span>Training 0.5 JC/unit</span> &nbsp;|&nbsp;
             <span>Storage 0.01 JC/unit</span>
    </div>
    <label for="earn-units">Compute Units</label>
    <input type="number" id="earn-units" value="10" min="0.01" step="0.01" />
    <label for="earn-type">Task Type</label>
    <select id="earn-type">
      <option value="inference">Inference</option>
      <option value="training">Training</option>
      <option value="storage">Storage</option>
    </select>
    <button class="btn btn-success btn-full" onclick="earnJC()">Contribute Compute</button>
    <div class="msg" id="earn-msg"></div>
  </div>

  <!-- ── Spend JC / Post Task ── -->
  <div class="card">
    <div class="card-title">&#9660; Spend JC — Submit Task</div>
    <label for="task-desc">Task Description</label>
    <textarea id="task-desc" placeholder="Describe what you need Janus to do..."></textarea>
    <label for="task-reward">JC Reward</label>
    <input type="number" id="task-reward" value="5" min="0.01" step="0.01" />
    <label for="task-type">Task Type</label>
    <select id="task-type">
      <option value="inference">Inference</option>
      <option value="training">Training</option>
      <option value="storage">Storage</option>
    </select>
    <button class="btn btn-secondary btn-full" onclick="postTask()">Post Task</button>
    <div class="msg" id="spend-msg"></div>
  </div>

  <!-- ── Network Stats ── -->
  <div class="card full-width">
    <div class="card-title">&#9670; Network Statistics</div>
    <div class="stats-grid">
      <div class="stat-box">
        <span class="stat-val" id="stat-circulation">—</span>
        <div class="stat-lbl">JC in Circulation</div>
      </div>
      <div class="stat-box">
        <span class="stat-val" id="stat-accounts">—</span>
        <div class="stat-lbl">Active Accounts</div>
      </div>
      <div class="stat-box">
        <span class="stat-val" id="stat-transactions">—</span>
        <div class="stat-lbl">Total Transactions</div>
      </div>
      <div class="stat-box">
        <span class="stat-val" id="stat-compute">—</span>
        <div class="stat-lbl">Compute Contributed (JC)</div>
      </div>
    </div>
  </div>

  <!-- ── Open Tasks ── -->
  <div class="card full-width">
    <div class="card-title">&#9670; Open Tasks</div>
    <div id="open-tasks-list"><div class="empty-state">Loading...</div></div>
  </div>

  <!-- ── Transaction History ── -->
  <div class="card full-width">
    <div class="card-title">&#9670; Transaction History</div>
    <div id="tx-history-list"><div class="empty-state">Loading...</div></div>
  </div>

</main>

<script>
  const API = 'http://localhost:8004';

  function userId() {
    return document.getElementById('global-user-id').value.trim() || 'demo_user';
  }

  function showMsg(id, text, type) {
    const el = document.getElementById(id);
    el.textContent = text;
    el.className = 'msg ' + type;
  }

  async function apiFetch(path, opts = {}) {
    const res = await fetch(API + path, {
      headers: { 'Content-Type': 'application/json' },
      ...opts,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    return data;
  }

  // ── Balance ──
  async function loadBalance() {
    try {
      const data = await apiFetch('/jc/balance/' + encodeURIComponent(userId()));
      const bal = parseFloat(data.balance_jc).toFixed(4);
      document.getElementById('header-balance').textContent = bal + ' JC';
    } catch (e) {
      document.getElementById('header-balance').textContent = 'ERR';
    }
  }

  // ── Network stats ──
  async function loadNetworkStats() {
    try {
      const d = await apiFetch('/jc/network');
      document.getElementById('stat-circulation').textContent =
        parseFloat(d.total_jc_in_circulation).toFixed(2);
      document.getElementById('stat-accounts').textContent = d.total_accounts;
      document.getElementById('stat-transactions').textContent = d.total_transactions;
      document.getElementById('stat-compute').textContent =
        parseFloat(d.total_compute_contributed).toFixed(4);
    } catch (e) {
      console.error('Stats error:', e);
    }
  }

  // ── Transaction history ──
  async function loadHistory() {
    const el = document.getElementById('tx-history-list');
    try {
      const data = await apiFetch('/jc/history/' + encodeURIComponent(userId()) + '?limit=10');
      const txs = data.transactions || [];
      if (!txs.length) {
        el.innerHTML = '<div class="empty-state">No transactions yet.</div>';
        return;
      }
      el.innerHTML = txs.map(tx => {
        const isIncoming = tx.to_user === userId();
        const amtClass = isIncoming ? 'amount-pos' : 'amount-neg';
        const amtSign  = isIncoming ? '+' : '-';
        const badgeClass = 'badge-' + (tx.tx_type === 'welcome_bonus' ? 'welcome' : tx.tx_type);
        const ts = tx.timestamp ? tx.timestamp.replace('T', ' ').slice(0, 19) : '';
        return \`
          <div class="list-item">
            <div class="item-main">
              <div class="item-desc">\${escHtml(tx.description || tx.tx_type)}</div>
              <div class="item-meta">\${ts} &nbsp;|&nbsp; \${tx.tx_id ? tx.tx_id.slice(0,8) + '…' : ''}</div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px">
              <span class="\${amtClass}">\${amtSign}\${parseFloat(tx.amount_jc).toFixed(4)} JC</span>
              <span class="badge \${badgeClass}">\${tx.tx_type}</span>
            </div>
          </div>\`;
      }).join('');
    } catch (e) {
      el.innerHTML = '<div class="empty-state">Could not load history.</div>';
    }
  }

  // ── Open tasks ──
  async function loadOpenTasks() {
    const el = document.getElementById('open-tasks-list');
    try {
      const data = await apiFetch('/jc/tasks');
      const tasks = data.tasks || [];
      if (!tasks.length) {
        el.innerHTML = '<div class="empty-state">No open tasks right now.</div>';
        return;
      }
      el.innerHTML = tasks.map(t => {
        const ts = t.created_at ? t.created_at.replace('T', ' ').slice(0, 19) : '';
        return \`
          <div class="list-item">
            <div class="item-main">
              <div class="item-desc">\${escHtml(t.description)}</div>
              <div class="item-meta">Posted by \${escHtml(t.requester_id)} &nbsp;|&nbsp; \${ts} &nbsp;|&nbsp; Type: \${t.task_type}</div>
            </div>
            <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px">
              <span class="amount-pos">+\${parseFloat(t.jc_reward).toFixed(4)} JC</span>
              <span class="badge badge-open">open</span>
              <button class="btn btn-primary" style="padding:6px 12px;font-size:10px"
                onclick="claimTask('\${escHtml(t.task_id)}')">Claim</button>
            </div>
          </div>\`;
      }).join('');
    } catch (e) {
      el.innerHTML = '<div class="empty-state">Could not load tasks.</div>';
    }
  }

  // ── Earn JC ──
  async function earnJC() {
    const units = parseFloat(document.getElementById('earn-units').value);
    const type  = document.getElementById('earn-type').value;
    if (!units || units <= 0) {
      showMsg('earn-msg', 'Enter a positive compute unit value.', 'err');
      return;
    }
    try {
      const data = await apiFetch('/jc/earn', {
        method: 'POST',
        body: JSON.stringify({ user_id: userId(), compute_units: units, task_type: type }),
      });
      showMsg('earn-msg',
        \`Earned \${parseFloat(data.earned_jc).toFixed(4)} JC! New balance: \${parseFloat(data.new_balance).toFixed(4)} JC\`,
        'ok');
      loadAll();
    } catch (e) {
      showMsg('earn-msg', 'Error: ' + e.message, 'err');
    }
  }

  // ── Post task ──
  async function postTask() {
    const desc   = document.getElementById('task-desc').value.trim();
    const reward = parseFloat(document.getElementById('task-reward').value);
    const type   = document.getElementById('task-type').value;
    if (!desc) {
      showMsg('spend-msg', 'Please enter a task description.', 'err');
      return;
    }
    if (!reward || reward <= 0) {
      showMsg('spend-msg', 'Enter a positive JC reward.', 'err');
      return;
    }
    try {
      const data = await apiFetch('/jc/tasks', {
        method: 'POST',
        body: JSON.stringify({
          requester_id: userId(),
          task_description: desc,
          jc_reward: reward,
          task_type: type,
        }),
      });
      showMsg('spend-msg',
        \`Task posted! ID: \${data.task_id.slice(0, 8)}…\`,
        'ok');
      document.getElementById('task-desc').value = '';
      loadAll();
    } catch (e) {
      showMsg('spend-msg', 'Error: ' + e.message, 'err');
    }
  }

  // ── Claim task ──
  async function claimTask(taskId) {
    try {
      await apiFetch('/jc/tasks/' + encodeURIComponent(taskId) + '/claim', {
        method: 'POST',
        body: JSON.stringify({ worker_id: userId() }),
      });
      loadAll();
    } catch (e) {
      alert('Could not claim task: ' + e.message);
    }
  }

  // ── Helpers ──
  function escHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function loadAll() {
    loadBalance();
    loadNetworkStats();
    loadHistory();
    loadOpenTasks();
  }

  // Auto-refresh every 30 seconds
  loadAll();
  setInterval(loadAll, 30000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Janus User Frontend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse, summary="Janus Network user dashboard")
def index() -> HTMLResponse:
    """Serve the Janus Network single-page application."""
    return HTMLResponse(content=HTML_PAGE)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("janus_user_frontend:app", host="0.0.0.0", port=8005, reload=False)
