/* Live Match Prediction — vanilla JS, connects to /api/live and /ws/live */

const LIVE_API = '/api/live';
const LIVE_WS_PROTOCOL = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const LIVE_WS = `${LIVE_WS_PROTOCOL}//${window.location.host}/ws/live`;

let liveWs = null;
let currentMatchId = null;
let timelineData = [];

// --- Navigation ---
function initLivePage() {
  document.getElementById('livePageLink').addEventListener('click', (e) => {
    e.preventDefault();
    showLivePage();
  });
  document.getElementById('draftPageLink').addEventListener('click', (e) => {
    e.preventDefault();
    showDraftPage();
  });
}

function showLivePage() {
  document.getElementById('draftArea').style.display = 'none';
  document.getElementById('liveArea').style.display = '';
  document.getElementById('livePageLink').classList.add('active');
  document.getElementById('draftPageLink').classList.remove('active');
  fetchLiveMatches();
}

function showDraftPage() {
  document.getElementById('liveArea').style.display = 'none';
  document.getElementById('draftArea').style.display = '';
  document.getElementById('draftPageLink').classList.add('active');
  document.getElementById('livePageLink').classList.remove('active');
}

// --- Fetch live matches ---
async function fetchLiveMatches() {
  const list = document.getElementById('liveMatchList');
  list.innerHTML = '<div class="loading-text">Loading live matches...</div>';

  try {
    const res = await fetch(`${LIVE_API}/matches`);
    const data = await res.json();

    if (!data.matches || data.matches.length === 0) {
      list.innerHTML = '<div class="empty-text">No live matches currently. Try again in a few minutes.</div>';
      return;
    }

    list.innerHTML = '';
    data.matches.forEach(m => {
      const card = document.createElement('div');
      card.className = 'live-match-card';
      card.innerHTML = `
        <div class="live-match-teams">
          <span class="team-radiant">${escapeHtml(m.radiant_team || 'Radiant')}</span>
          <span class="vs">VS</span>
          <span class="team-dire">${escapeHtml(m.dire_team || 'Dire')}</span>
        </div>
        <div class="live-match-meta">
          <span>Match #${m.match_id}</span>
          <span>${formatDuration(m.duration)}</span>
          <span>${m.spectators || 0} viewers</span>
        </div>
      `;
      card.addEventListener('click', () => startLivePrediction(m.match_id, m.radiant_team, m.dire_team));
      list.appendChild(card);
    });
  } catch (e) {
    list.innerHTML = '<div class="error-text">Failed to load matches</div>';
  }
}

// --- Live prediction display ---
function startLivePrediction(matchId, radiantName, direName) {
  currentMatchId = matchId;
  timelineData = [];

  // Update panel headers
  document.getElementById('liveRadiantName').textContent = radiantName || 'Radiant';
  document.getElementById('liveDireName').textContent = direName || 'Dire';

  // Connect WebSocket
  if (liveWs) liveWs.close();
  liveWs = new WebSocket(LIVE_WS);

  liveWs.onopen = () => {
    liveWs.send(JSON.stringify({ match_id: matchId, interval: 10 }));
    document.getElementById('liveStatus').textContent = 'Connected';
    document.getElementById('liveStatus').className = 'live-status connected';
  };

  liveWs.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'prediction') {
      updatePredictionDisplay(data);
    } else if (data.type === 'error') {
      document.getElementById('liveStatus').textContent = 'Error: ' + data.detail;
      document.getElementById('liveStatus').className = 'live-status error';
    }
  };

  liveWs.onclose = () => {
    document.getElementById('liveStatus').textContent = 'Disconnected';
    document.getElementById('liveStatus').className = 'live-status';
  };

  // Show prediction panel
  document.getElementById('livePredictionPanel').style.display = '';
  document.getElementById('liveMatchList').style.display = 'none';
}

function updatePredictionDisplay(data) {
  const prob = data.radiant_win_probability;

  // Win probability gauge
  const gaugeFill = document.getElementById('liveWinGaugeFill');
  const probText = document.getElementById('liveWinProb');

  if (gaugeFill) {
    gaugeFill.style.width = `${prob * 100}%`;
    const color = prob >= 0.55 ? 'var(--green)' : prob <= 0.45 ? 'var(--red)' : 'var(--gold)';
    gaugeFill.style.background = color;
  }
  if (probText) {
    probText.textContent = `${(prob * 100).toFixed(1)}%`;
    probText.style.color = prob >= 0.55 ? 'var(--green)' : prob <= 0.45 ? 'var(--red)' : 'var(--gold)';
  }

  // Team percentages (liveDirePct exists; liveWinProb already shows Radiant %)
  const dPct = document.getElementById('liveDirePct');
  if (dPct) dPct.textContent = `${((1 - prob) * 100).toFixed(0)}%`;

  // Game time
  const timeEl = document.getElementById('liveGameTime');
  if (timeEl) timeEl.textContent = `${data.minute}:00`;

  // Timeline data point
  addTimelinePoint(data.minute, prob);

  // Feature contributions
  const featEl = document.getElementById('liveFeatures');
  if (featEl && data.features) {
    featEl.innerHTML = `
      <div class="feature-row">
        <span>Gold Advantage</span>
        <span class="${data.features.radiant_gold_adv > 0 ? 'pos' : 'neg'}">
          ${data.features.radiant_gold_adv > 0 ? '+' : ''}${formatGold(data.features.radiant_gold_adv)}
        </span>
      </div>
      <div class="feature-row">
        <span>XP Advantage</span>
        <span class="${data.features.radiant_xp_adv > 0 ? 'pos' : 'neg'}">
          ${data.features.radiant_xp_adv > 0 ? '+' : ''}${formatGold(data.features.radiant_xp_adv)}
        </span>
      </div>
      <div class="feature-row">
        <span>Kill Diff</span>
        <span class="${data.features.kill_diff > 0 ? 'pos' : 'neg'}">
          ${data.features.kill_diff > 0 ? '+' : ''}${data.features.kill_diff}
        </span>
      </div>
      <div class="feature-row">
        <span>Tower Diff</span>
        <span class="${data.features.tower_diff > 0 ? 'pos' : 'neg'}">
          ${data.features.tower_diff > 0 ? '+' : ''}${data.features.tower_diff}
        </span>
      </div>
      <div class="feature-row">
        <span>Teamfight Diff</span>
        <span class="${data.features.tf_diff > 0 ? 'pos' : 'neg'}">
          ${data.features.tf_diff > 0 ? '+' : ''}${data.features.tf_diff}
        </span>
      </div>
    `;
  }
}

// --- Timeline chart (canvas) ---
function addTimelinePoint(minute, probability) {
  const idx = timelineData.findIndex(p => p.minute === minute);
  if (idx >= 0) timelineData[idx] = { minute, probability };
  else timelineData.push({ minute, probability });
  timelineData.sort((a, b) => a.minute - b.minute);
  drawTimeline();
}

function drawTimeline() {
  const canvas = document.getElementById('liveTimeline');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;

  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#161b22';
  ctx.fillRect(0, 0, W, H);

  // 50% line
  ctx.strokeStyle = '#30363d';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, H / 2);
  ctx.lineTo(W, H / 2);
  ctx.stroke();
  ctx.setLineDash([]);

  if (timelineData.length < 2) return;

  const maxMin = Math.max(30, timelineData[timelineData.length - 1].minute);

  // Gradient stroke
  const grad = ctx.createLinearGradient(0, 0, W, 0);
  grad.addColorStop(0, '#3b82f6');
  grad.addColorStop(0.5, '#a855f7');
  grad.addColorStop(1, '#ef4444');
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2;

  ctx.beginPath();
  timelineData.forEach((p, i) => {
    const x = (p.minute / maxMin) * W;
    const y = (1 - p.probability) * H;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Fill under line
  const lastP = timelineData[timelineData.length - 1];
  ctx.lineTo((lastP.minute / maxMin) * W, H);
  ctx.lineTo(0, H);
  ctx.closePath();
  ctx.fillStyle = 'rgba(59, 130, 246, 0.1)';
  ctx.fill();

  // Current point
  const x = (lastP.minute / maxMin) * W;
  const y = (1 - lastP.probability) * H;
  ctx.fillStyle = lastP.probability > 0.5 ? '#3b82f6' : '#ef4444';
  ctx.beginPath();
  ctx.arc(x, y, 5, 0, Math.PI * 2);
  ctx.fill();

  // Labels
  ctx.fillStyle = '#8b949e';
  ctx.font = '10px monospace';
  ctx.fillText('100% Radiant', 4, 12);
  ctx.fillText('50%', 4, H / 2 - 2);
  ctx.fillText('100% Dire', 4, H - 4);
  ctx.fillText(`${maxMin} min`, W - 50, H - 4);
}

// --- Helpers ---
function formatDuration(seconds) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function formatGold(n) {
  return Math.abs(Math.round(n)).toLocaleString();
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function backToMatchList() {
  if (liveWs) liveWs.close();
  document.getElementById('livePredictionPanel').style.display = 'none';
  document.getElementById('liveMatchList').style.display = '';
  timelineData = [];
}

// --- Init ---
document.addEventListener('DOMContentLoaded', () => {
  initLivePage();
});
