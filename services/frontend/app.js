/* ================================================================
   Dota 2 Draft Predictor — single-file vanilla JS
   Connects to ML API at :8080 for win-rate predictions
   ================================================================ */

const API_BASE = '/api';
const WS_BASE = `ws://${window.location.host}/ws/draft`;

let draftWs = null;
let draftWsTurnId = 0;
let draftWsResolvers = {};

// ── Draft order for patch 60 (from dotaconstants) ────────────────
// Pattern: B1 B1 B0 B0 B1 B0 B0 P1 P0 B1 B1 B0 P0 P1 P1 P0 P0 P1 B1 B0 B1 B0 P1 P0
// After normalization (0 = first-pick team): B0 B0 B1 B1 B0 B1 B1 P0 P1 ...
const PATCH_60_ORDER = [
  ['ban',0],['ban',0],['ban',1],['ban',1],['ban',0],['ban',1],
  ['ban',1],['pick',0],['pick',1],
  ['ban',0],['ban',0],['ban',1],
  ['pick',1],['pick',0],['pick',0],['pick',1],['pick',1],['pick',0],
  ['ban',0],['ban',1],['ban',0],['ban',1],
  ['pick',0],['pick',1],
];

let firstPickTeam = 0; // 0=Radiant, 1=Dire

function getDraftPhases() {
  return PATCH_60_ORDER.map(([type, team0]) => ({
    type,
    team: team0 === 0
      ? (firstPickTeam === 0 ? 'radiant' : 'dire')
      : (firstPickTeam === 0 ? 'dire' : 'radiant')
  }));
}

let DRAFT_PHASES = getDraftPhases();

// ── State ────────────────────────────────────────────────────────
let HEROES_DATA = [];
let TEAMS_DATA = [];
let heroes = [];
let heroMap = {};
let draftIndex = 0;
let draftHistory = [];
let apiOk = false;
let predictAbort = null;
let tooltipDebounce = null;
let recommendations = [];
let recoAbort = null;
let recoRadiant = [];
let recoDire = [];
let recoPickRadiant = [];
let recoPickDire = [];
let isLoadingRecs = false;

// ── Init ─────────────────────────────────────────────────────────
async function init() {
  try {
    const [heroesRes, teamsRes] = await Promise.all([
      fetch('/heroes.json'),
      fetch('/teams.json')
    ]);
    HEROES_DATA = await heroesRes.json();
    TEAMS_DATA = await teamsRes.json();
  } catch (e) {
    console.error('Failed to load data files:', e);
  }

  heroes = HEROES_DATA;
  heroes.sort((a, b) => a.localized_name.localeCompare(b.localized_name));
  heroes.forEach(h => {
    h.img = `img/${h.name.replace('npc_dota_hero_', '')}.png`;
    heroMap[h.id] = h;
  });
  document.getElementById('footerInfo').textContent = `${heroes.length} heroes loaded`;

  checkApi();
  initTeamSelectors();
  document.getElementById('loading').classList.add('hidden');

  document.getElementById('search').addEventListener('input', e => renderGrid(e.target.value));
}

function initTeamSelectors() {
  const rSel = document.getElementById('radiant-team');
  const dSel = document.getElementById('dire-team');
  TEAMS_DATA.forEach(t => {
    const optR = document.createElement('option');
    optR.value = t.team_id;
    optR.textContent = `${t.name} (${t.tag}) — ${t.rating}`;
    rSel.appendChild(optR);
    const optD = document.createElement('option');
    optD.value = t.team_id;
    optD.textContent = `${t.name} (${t.tag}) — ${t.rating}`;
    dSel.appendChild(optD);
  });
}

// ── WebSocket connection for real-time MCTS streaming ─────────────
let draftWsConnected = null;  // Promise that resolves when WS is OPEN

function ensureWsConnected() {
  if (draftWs && draftWs.readyState === WebSocket.OPEN) return Promise.resolve();

  // If already connecting, wait for it
  if (draftWsConnected) return draftWsConnected;

  draftWsConnected = new Promise((resolve) => {
    draftWs = new WebSocket(WS_BASE);

    draftWs.onopen = () => {
      console.log('WS connected');
      resolve();
    };

    draftWs.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'error') return;

      if (data.type === 'mcts_progress') {
        const progressFill = document.getElementById('mctsProgressFill');
        const progressText = document.getElementById('mctsProgressText');
        const topPicks = document.getElementById('mctsTopPicks');
        if (progressFill && data.total > 0) {
          progressFill.style.width = `${(data.iteration / data.total) * 100}%`;
        }
        if (progressText) {
          progressText.textContent = `Evaluating hero ${data.hero_id} (${data.iteration}/${data.total})`;
        }
        if (topPicks && data.top_picks) {
          topPicks.innerHTML = data.top_picks.map(p =>
            `<li><span>${p.action} Hero ${p.hero_id}</span><span>${(p.win_rate * 100).toFixed(1)}%</span></li>`
          ).join('');
        }
      }

      if (data.type === 'mcts_complete') {
        const key = `${data.turn_id}_0`;
        const keyD = `${data.turn_id}_1`;
        if (draftWsResolvers[key]) {
          draftWsResolvers[key](data);
          delete draftWsResolvers[key];
        }
        if (draftWsResolvers[keyD]) {
          draftWsResolvers[keyD](data);
          delete draftWsResolvers[keyD];
        }
      }
    };

    draftWs.onclose = () => {
      console.log('WS disconnected');
      draftWsConnected = null;
      draftWs = null;
    };
  });

  return draftWsConnected;
}

function onTeamChange() {
  const rVal = document.getElementById('radiant-team').value;
  const dVal = document.getElementById('dire-team').value;
  const rSel = document.getElementById('radiant-team');
  const dSel = document.getElementById('dire-team');

  // Disable same team in the other dropdown
  Array.from(dSel.options).forEach(opt => {
    if (opt.value !== '0' && opt.value === rVal) {
      opt.disabled = true;
    } else {
      opt.disabled = false;
    }
  });
  Array.from(rSel.options).forEach(opt => {
    if (opt.value !== '0' && opt.value === dVal) {
      opt.disabled = true;
    } else {
      opt.disabled = false;
    }
  });

  // Enable Start button when both teams selected
  const startBtn = document.getElementById('startBtn');
  startBtn.disabled = !(rVal !== '0' && dVal !== '0');
}

function startDraft() {
  document.getElementById('draftArea').style.display = '';
  document.getElementById('startBtn').disabled = true;
  document.getElementById('startBtn').textContent = 'DRAFT IN PROGRESS';
  draftIndex = 0;
  draftHistory = [];
  recoRadiant = [];
  recoDire = [];
  recoPickRadiant = [];
  recoPickDire = [];
  DRAFT_PHASES = getDraftPhases();
  renderDraftBar();
  renderGrid();
  updatePhaseInfo();
  renderRecoPanel();
  fetchRecommendations();
}

function setFirstPick(team) {
  firstPickTeam = team;
  DRAFT_PHASES = getDraftPhases();
  draftIndex = 0;
  draftHistory = [];
  recoRadiant = [];
  recoDire = [];
  recoPickRadiant = [];
  recoPickDire = [];
  document.getElementById('fp-radiant').className = 'first-pick-btn' + (team === 0 ? ' active' : '');
  document.getElementById('fp-dire').className = 'first-pick-btn' + (team === 1 ? ' active' : '');
  renderDraftBar();
  renderGrid(document.getElementById('search').value);
  updatePhaseInfo();
  renderRecoPanel();
  fetchRecommendations();
}

async function fetchRecommendations() {
  if (!apiOk || draftIndex >= DRAFT_PHASES.length) {
    setDraftLock(false);
    recoRadiant = [];
    recoDire = [];
    recoPickRadiant = [];
    recoPickDire = [];
    renderRecoPanel();
    return;
  }
  const draftArray = buildDraftArray();
  const baseBody = {
    patch_id: 60,
    draft: draftArray,
    first_pick_team: firstPickTeam,
    radiant_team_id: parseInt(document.getElementById('radiant-team').value) || null,
    dire_team_id: parseInt(document.getElementById('dire-team').value) || null,
  };

  // Clear stale recommendations and lock UI
  recoRadiant = [];
  recoDire = [];
  recoPickRadiant = [];
  recoPickDire = [];
  setDraftLock(true);
  renderRecoPanel();

  document.getElementById('recoPanel').style.display = '';

  // Show MCTS overlay
  const overlay = document.getElementById('mctsOverlay');
  const progressFill = document.getElementById('mctsProgressFill');
  const progressText = document.getElementById('mctsProgressText');
  const topPicks = document.getElementById('mctsTopPicks');
  if (overlay) overlay.classList.remove('hidden');
  if (progressFill) progressFill.style.width = '0%';
  if (progressText) progressText.textContent = 'Starting search...';
  if (topPicks) topPicks.innerHTML = '';

  draftWsTurnId++;
  const turnId = draftWsTurnId;

  // Ensure WebSocket is connected before sending
  await ensureWsConnected();

  // Send requests for both teams via WebSocket
  const promises = [0, 1].map(for_team => {
    return new Promise((resolve) => {
      const key = `${turnId}_${for_team}`;
      draftWsResolvers[key] = resolve;
      draftWs.send(JSON.stringify({ ...baseBody, for_team, turn_id: turnId }));
    });
  });

  try {
    const [dR, dD] = await Promise.all(promises);

    recoRadiant = dR?.recommendations || [];
    recoDire = dD?.recommendations || [];
    recoPickRadiant = dR?.recommendations || [];
    recoPickDire = dD?.recommendations || [];

    setDraftLock(false);
    renderRecoPanel();
    renderGrid(document.getElementById('search').value);
  } catch (e) {
    recoRadiant = [];
    recoDire = [];
    recoPickRadiant = [];
    recoPickDire = [];
    setDraftLock(false);
    renderRecoPanel();
  } finally {
    if (overlay) setTimeout(() => overlay.classList.add('hidden'), 800);
  }
}

function renderRecoPanel() {
  const panel = document.getElementById('recoPanel');
  const bansSection = panel.querySelector('.reco-section');
  const picksSection = panel.querySelectorAll('.reco-section')[1];

  if (!recoRadiant.length && !recoDire.length && !recoPickRadiant.length && !recoPickDire.length) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = '';

  const phase = draftIndex < DRAFT_PHASES.length ? DRAFT_PHASES[draftIndex] : null;
  const currentTeam = phase ? phase.team : null;
  const isBan = phase ? phase.type === 'ban' : true;

  // Show bans section only during ban phases, picks only during pick phases
  if (bansSection) bansSection.style.display = isBan ? '' : 'none';
  if (picksSection) picksSection.style.display = isBan ? 'none' : '';

  // BANS section
  const titleD = document.getElementById('recoTitleD');
  const listD = document.getElementById('recoListD');
  titleD.textContent = `Best bans for Dire`;
  listD.innerHTML = '';
  recoDire.slice(0, 5).forEach((rec, i) => {
    listD.appendChild(createRecoChip(rec, i, 'dire', currentTeam === 'dire' && isBan));
  });

  const titleR = document.getElementById('recoTitleR');
  const listR = document.getElementById('recoListR');
  titleR.textContent = `Best bans for Radiant`;
  listR.innerHTML = '';
  recoRadiant.slice(0, 5).forEach((rec, i) => {
    listR.appendChild(createRecoChip(rec, i, 'radiant', currentTeam === 'radiant' && isBan));
  });

  // PICKS section
  const pickTitleD = document.getElementById('recoPickTitleD');
  const pickListD = document.getElementById('recoPickListD');
  pickTitleD.textContent = `Best picks for Dire`;
  pickListD.innerHTML = '';
  recoPickDire.slice(0, 5).forEach((rec, i) => {
    pickListD.appendChild(createRecoChip(rec, i, 'dire', currentTeam === 'dire' && !isBan));
  });

  const pickTitleR = document.getElementById('recoPickTitleR');
  const pickListR = document.getElementById('recoPickListR');
  pickTitleR.textContent = `Best picks for Radiant`;
  pickListR.innerHTML = '';
  recoPickRadiant.slice(0, 5).forEach((rec, i) => {
    pickListR.appendChild(createRecoChip(rec, i, 'radiant', currentTeam === 'radiant' && !isBan));
  });
}

function createRecoChip(rec, index, side, active) {
  const hero = heroMap[rec.hero_id];
  if (!hero) return document.createElement('div');
  // Disable if hero already drafted OR predictions are loading
  const alreadyDrafted = draftHistory.some(a => a.heroId === rec.hero_id);
  const clickable = active && !alreadyDrafted && !isLoadingRecs;
  const chip = document.createElement('div');
  chip.className = `reco-chip rank-${index + 1}` + (clickable ? '' : ' disabled');
  const winPct = rec.win_probability ? (rec.win_probability * 100).toFixed(1) : '—';
  const score = rec.score ? rec.score.toFixed(4) : '—';
  const phase = draftIndex < DRAFT_PHASES.length ? DRAFT_PHASES[draftIndex] : null;
  const teamName = side === 'radiant' ? 'Radiant' : 'Dire';
  const slotNum = draftIndex + 1;
  const actionType = phase ? phase.type : '';
  const teamInfo = rec.team_games > 0
    ? ` | team: ${rec.team_games}g ${rec.team_win_rate ? (rec.team_win_rate * 100).toFixed(0) + '% WR' : ''}`
    : '';
  const boostTag = rec.boosted ? ' <span style="color:var(--green);font-weight:700">★</span>' : '';
  chip.innerHTML = `
    <img src="${hero.img}" alt="">
    <div>
      <div class="rc-name">${hero.localized_name}${boostTag}</div>
      <div class="rc-score">${teamName} slot ${slotNum} ${actionType} | WR ${winPct}% | score ${score}${teamInfo}</div>
    </div>`;
  if (active) {
    chip.addEventListener('click', () => pickHero(rec.hero_id));
  }
  return chip;
}

// ── API health check ─────────────────────────────────────────────
async function checkApi() {
  const el = document.getElementById('apiStatus');
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    el.className = 'api-url ok';
    el.textContent = `API: connected (patch ${data.patch_models_loaded?.[0] || '?'})`;
    apiOk = true;
  } catch {
    el.className = 'api-url err';
    el.textContent = 'API: unreachable';
    apiOk = false;
  }
}

// ── Draft Bar Rendering ──────────────────────────────────────────
function renderDraftBar() {
  const radiantSlots = document.getElementById('radiantSlots');
  const direSlots = document.getElementById('direSlots');
  radiantSlots.innerHTML = '';
  direSlots.innerHTML = '';

  const rSlots = [];
  const dSlots = [];

  DRAFT_PHASES.forEach((phase, i) => {
    const slot = document.createElement('div');
    slot.className = `slot ${phase.type} ${phase.team}`;
    slot.dataset.index = i;

    if (phase.team === 'radiant') {
      if (i < draftIndex) {
        const action = draftHistory[i];
        const hero = heroMap[action.heroId];
        if (hero) {
          slot.classList.add('filled');
          slot.innerHTML = `<img src="${hero.img}" alt="" title="${hero.localized_name}">`;
        }
      } else if (i === draftIndex) {
        slot.classList.add('active');
        slot.textContent = '+';
      }
      rSlots.push(slot);
    } else {
      if (i < draftIndex) {
        const action = draftHistory[i];
        const hero = heroMap[action.heroId];
        if (hero) {
          slot.classList.add('filled');
          slot.innerHTML = `<img src="${hero.img}" alt="" title="${hero.localized_name}">`;
        }
      } else if (i === draftIndex) {
        slot.classList.add('active');
        slot.textContent = '+';
      }
      dSlots.push(slot);
    }
  });

  // Add phase dividers for Radiant
  let prevType = null;
  rSlots.forEach(slot => {
    const phase = DRAFT_PHASES[+slot.dataset.index];
    if (prevType && phase.type !== prevType) {
      const div = document.createElement('div');
      div.className = 'phase-divider';
      radiantSlots.appendChild(div);
    }
    radiantSlots.appendChild(slot);
    prevType = phase.type;
  });

  // Add phase dividers for Dire
  prevType = null;
  dSlots.forEach(slot => {
    const phase = DRAFT_PHASES[+slot.dataset.index];
    if (prevType && phase.type !== prevType) {
      const div = document.createElement('div');
      div.className = 'phase-divider';
      direSlots.appendChild(div);
    }
    direSlots.appendChild(slot);
    prevType = phase.type;
  });

  // Update center indicator
  if (draftIndex < DRAFT_PHASES.length) {
    const phase = DRAFT_PHASES[draftIndex];
    document.getElementById('draftPhase').textContent = `Phase ${getPhaseNumber()}`;
    document.getElementById('draftAction').textContent = (draftIndex + 1);
    document.getElementById('draftTurn').textContent = `${phase.type.toUpperCase()} — ${phase.team.toUpperCase()}`;
  } else {
    document.getElementById('draftPhase').textContent = 'Draft Complete';
    document.getElementById('draftAction').textContent = '✓';
    document.getElementById('draftTurn').textContent = 'All slots filled';
  }
}

function getPhaseNumber() {
  if (draftIndex < 4) return 1;
  if (draftIndex < 10) return 2;
  if (draftIndex < 14) return 3;
  return 4;
}

// ── Hero Grid ────────────────────────────────────────────────────
function renderGrid(filter = '') {
  const grid = document.getElementById('heroGrid');
  const used = new Set(draftHistory.map(a => a.heroId));
  const term = filter.toLowerCase();

  // Build recommendation rank map: heroId -> rank (1-5) from CURRENT team's perspective
  const recoRank = {};
  const currentTeam = draftIndex < DRAFT_PHASES.length ? DRAFT_PHASES[draftIndex].team : null;
  const activeRecos = currentTeam === 'dire' ? recoDire : recoRadiant;
  activeRecos.slice(0, 5).forEach((rec, i) => {
    recoRank[rec.hero_id] = i + 1;
  });

  const filtered = heroes.filter(h => {
    if (term && !h.localized_name.toLowerCase().includes(term) &&
        !h.name.toLowerCase().includes(term) &&
        !(h.roles || []).some(r => r.toLowerCase().includes(term))) return false;
    return true;
  });

  grid.innerHTML = '';
  filtered.forEach(hero => {
    const card = document.createElement('div');
    card.className = 'hero-card';
    if (used.has(hero.id)) card.classList.add('used');

    // Mark by draft role
    draftHistory.forEach(a => {
      if (a.heroId === hero.id) {
        card.classList.add(`${a.team}-${a.type}`);
      }
    });

    // Apply recommendation highlight
    if (!used.has(hero.id) && recoRank[hero.id]) {
      card.classList.add(`reco-${recoRank[hero.id]}`);
    }

    const badge = (!used.has(hero.id) && recoRank[hero.id])
      ? `<div class="reco-badge">${recoRank[hero.id]}</div>` : '';

    card.innerHTML = `
      ${badge}
      <img class="hero-img" src="${hero.img}" alt=""
           onerror="this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 60 52%22><rect fill=%22%2321262d%22 width=%2260%22 height=%2252%22/><text x=%2230%22 y=%2230%22 text-anchor=%22middle%22 fill=%22%238b949e%22 font-size=%2210%22>?</text></svg>'">
      <div class="hero-name">${hero.localized_name}</div>
    `;

    card.addEventListener('click', () => pickHero(hero.id));
    card.addEventListener('mouseenter', e => showTooltip(e, hero.id));
    card.addEventListener('mousemove', e => moveTooltip(e));
    card.addEventListener('mouseleave', hideTooltip);

    grid.appendChild(card);
  });
}

// ── UI Lock State ────────────────────────────────────────────────
function setDraftLock(locked) {
  isLoadingRecs = locked;
  const gridContainer = document.querySelector('.hero-grid');
  const recoPanel = document.getElementById('recoPanel');

  if (locked) {
    gridContainer.classList.add('locked');
    recoPanel.classList.add('locked');
  } else {
    gridContainer.classList.remove('locked');
    recoPanel.classList.remove('locked');
  }
}

// ── Pick / Ban ───────────────────────────────────────────────────
function pickHero(heroId) {
  if (isLoadingRecs) return;
  if (draftIndex >= DRAFT_PHASES.length) return;
  if (draftHistory.some(a => a.heroId === heroId)) return;

  const phase = DRAFT_PHASES[draftIndex];
  draftHistory.push({
    heroId,
    type: phase.type,
    team: phase.team,
    phaseIndex: draftIndex
  });
  draftIndex++;

  renderDraftBar();
  renderGrid(document.getElementById('search').value);
  updatePhaseInfo();
  fetchRecommendations();
  updateMatchForecast();
}

function undoDraft() {
  if (draftIndex === 0) return;
  if (recoAbort) recoAbort.abort();
  setDraftLock(false);

  draftIndex--;
  draftHistory.pop();
  renderDraftBar();
  renderGrid(document.getElementById('search').value);
  updatePhaseInfo();
  fetchRecommendations();
  updateMatchForecast();
}

function resetDraft() {
  if (recoAbort) recoAbort.abort();
  setDraftLock(false);

  draftIndex = 0;
  draftHistory = [];
  recoRadiant = [];
  recoDire = [];
  recoPickRadiant = [];
  recoPickDire = [];
  renderDraftBar();
  renderGrid(document.getElementById('search').value);
  updatePhaseInfo();
  renderRecoPanel();
  fetchRecommendations();
  updateMatchForecast();
}

function updatePhaseInfo() {
  const el = document.getElementById('phaseInfo');
  const bans = draftHistory.filter(a => a.type === 'ban').length;
  const picks = draftHistory.filter(a => a.type === 'pick').length;
  el.innerHTML = `Bans: <strong>${bans}/14</strong> &bull; Picks: <strong>${picks}/10</strong>`;
}

// ── Build draft array for API ────────────────────────────────────
function buildApiDraft(heroId, team, isPick) {
  const order = draftIndex + 1;
  const teamInt = team === 'radiant' ? 0 : 1;
  return {
    hero_id: heroId,
    is_pick: isPick,
    team: teamInt,
    order: order,
  };
}

function buildDraftArray() {
  return draftHistory.map((a, i) => ({
    hero_id: a.heroId,
    is_pick: a.type === 'pick',
    team: a.team === 'radiant' ? 0 : 1,
    order: i + 1,
  }));
}

// ── 5v5 Victory Forecast ────────────────────────────────────────
async function updateMatchForecast() {
  const forecastEl = document.getElementById('matchForecast');
  if (draftIndex < DRAFT_PHASES.length) {
    forecastEl.style.display = 'none';
    return;
  }

  const radPicks = draftHistory.filter(a => a.team === 'radiant' && a.type === 'pick').map(a => a.heroId);
  const direPicks = draftHistory.filter(a => a.team === 'dire' && a.type === 'pick').map(a => a.heroId);

  if (radPicks.length !== 5 || direPicks.length !== 5) {
    forecastEl.style.display = 'none';
    return;
  }

  forecastEl.style.display = 'block';
  document.getElementById('forecastRadiantPct').textContent = 'Loading...';
  document.getElementById('forecastDirePct').textContent = '';
  document.getElementById('forecastRadiant').style.width = '50%';
  document.getElementById('forecastDire').style.width = '50%';

  try {
    const res = await fetch(`${API_BASE}/predict-match`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patch_id: 60,
        radiant_heroes: radPicks,
        dire_heroes: direPicks,
        radiant_team_id: parseInt(document.getElementById('radiant-team').value) || null,
        dire_team_id: parseInt(document.getElementById('dire-team').value) || null,
      })
    });

    if (!res.ok) throw new Error('Forecast failed');
    const data = await res.json();

    const rP = data.radiant_win_probability;
    const dP = data.dire_win_probability;
    document.getElementById('forecastRadiant').style.width = `${rP * 100}%`;
    document.getElementById('forecastDire').style.width = `${dP * 100}%`;
    document.getElementById('forecastRadiantPct').textContent = `${(rP * 100).toFixed(1)}%`;
    document.getElementById('forecastDirePct').textContent = `${(dP * 100).toFixed(1)}%`;
  } catch (e) {
    forecastEl.style.display = 'none';
    console.error('Match forecast failed:', e);
  }
}

// ── Prediction Tooltip ───────────────────────────────────────────
function showTooltip(e, heroId) {
  const tooltip = document.getElementById('tooltip');
  const hero = heroMap[heroId];
  if (!hero) return;

  const nextAction = draftIndex < DRAFT_PHASES.length ? DRAFT_PHASES[draftIndex] : null;
  const isRadiantTurn = nextAction?.team === 'radiant';
  const suggestedTeam = isRadiantTurn ? 'radiant' : 'dire';

  // Show hero info immediately
  tooltip.innerHTML = `
    <div class="tooltip-hero">
      <img src="${hero.img}" alt=""
           onerror="this.style.display='none'">
      <div>
        <div class="tooltip-hero-name">${hero.localized_name}</div>
        <div class="tooltip-hero-attr">${hero.primary_attr || '?'} &bull; ${(hero.roles||[]).join(', ') || 'Unknown'}</div>
      </div>
    </div>
    <div class="tooltip-pred loading">Loading prediction...</div>
  `;
  tooltip.classList.add('show');
  moveTooltip(e);

  if (!apiOk) {
    tooltip.querySelector('.tooltip-pred').className = 'tooltip-pred error';
    tooltip.querySelector('.tooltip-pred').textContent = 'ML API not connected';
    return;
  }

  if (draftHistory.some(a => a.heroId === heroId)) {
    tooltip.querySelector('.tooltip-pred').className = 'tooltip-pred error';
    tooltip.querySelector('.tooltip-pred').textContent = 'Already drafted';
    return;
  }

  if (!nextAction) {
    tooltip.querySelector('.tooltip-pred').className = 'tooltip-pred error';
    tooltip.querySelector('.tooltip-pred').textContent = 'Draft complete';
    return;
  }

  // Debounce API call — only fire after 300ms of no mouse movement
  if (tooltipDebounce) clearTimeout(tooltipDebounce);
  tooltipDebounce = setTimeout(() => fetchTooltipPrediction(heroId, nextAction, suggestedTeam), 300);
}

async function fetchTooltipPrediction(heroId, nextAction, suggestedTeam) {
  const tooltip = document.getElementById('tooltip');
  if (!tooltip.classList.contains('show')) return;

  // Use current draft state (don't append the hovered hero — it would be "taken")
  const draftArray = buildDraftArray();

  try {
    if (predictAbort) predictAbort.abort();
    predictAbort = new AbortController();

    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patch_id: 60,
        draft: draftArray,
        first_pick_team: firstPickTeam,
        radiant_team_id: parseInt(document.getElementById('radiant-team').value) || null,
        dire_team_id: parseInt(document.getElementById('dire-team').value) || null,
        for_team: nextAction.team === 'radiant' ? 0 : 1,
        num_recommendations: 50,
      }),
      signal: predictAbort.signal
    });

    if (!res.ok) {
      const errBody = await res.text();
      throw new Error(`API ${res.status}: ${errBody.slice(0, 100)}`);
    }
    const data = await res.json();
    const predEl = tooltip.querySelector('.tooltip-pred');
    if (!predEl) return;

    // Find this hero in recommendations
    const rec = data.recommendations.find(r => r.hero_id === heroId);

    if (!rec) {
      predEl.className = 'tooltip-pred';
      predEl.innerHTML = `
        <div class="pred-row">
          <span class="pred-label">Win Rate for ${suggestedTeam === 'radiant' ? 'Radiant' : 'Dire'}</span>
          <span class="pred-value mid">N/A</span>
        </div>
        <div class="pred-row" style="margin-top:4px">
          <span class="pred-label" style="font-size:11px;color:var(--text2)">Not recommended for this turn by the model</span>
        </div>
        <div class="pred-row" style="margin-top:6px">
          <span class="pred-label" style="font-size:10px;color:var(--text3)">If ${nextAction.type} by ${suggestedTeam}</span>
        </div>`;
      return;
    }

    const winRate = rec.win_probability ?? rec.score ?? 0.5;
    const winPct = (winRate * 100).toFixed(1);
    const cls = winRate >= 0.55 ? 'high' : winRate >= 0.45 ? 'mid' : 'low';
    const barColor = winRate >= 0.55 ? 'var(--green)' :
                     winRate >= 0.45 ? 'var(--gold)' : 'var(--red)';

    const rankLabel = `Rank #${data.recommendations.indexOf(rec) + 1} of ${data.recommendations.length}`;
    const teamGames = rec.team_games || 0;
    const teamWr = rec.team_win_rate;
    const teamInfo = teamGames > 0
      ? `<div class="pred-row" style="margin-top:2px">
           <span class="pred-label">Team Record</span>
           <span class="pred-value" style="font-size:11px;color:var(--text2)">${teamGames}g, ${teamWr ? (teamWr * 100).toFixed(0) + '% WR' : '—'}</span>
         </div>`
      : '';
    const boostTag = rec.boosted ? ' <span style="color:var(--green);font-weight:700">★</span>' : '';

    predEl.className = 'tooltip-pred';
    predEl.innerHTML = `
      <div class="pred-row">
        <span class="pred-label">Win Rate for ${suggestedTeam === 'radiant' ? 'Radiant' : 'Dire'}</span>
        <span class="pred-value ${cls}">${winPct}%${boostTag}</span>
      </div>
      <div class="pred-bar">
        <div class="pred-bar-fill" style="width:${winPct}%;background:${barColor}"></div>
      </div>
      ${rec.pick_probability ? `<div class="pred-row" style="margin-top:4px">
        <span class="pred-label">Pick Probability</span>
        <span class="pred-value" style="font-size:11px;color:var(--text2)">${(rec.pick_probability * 100).toFixed(0)}%</span>
      </div>` : ''}
      ${teamInfo}
      <div class="pred-row" style="margin-top:2px">
        <span class="pred-label">Rank</span>
        <span class="pred-value" style="font-size:11px;color:var(--text2)">${rankLabel}</span>
      </div>
      <div class="pred-row" style="margin-top:6px">
        <span class="pred-label" style="font-size:10px;color:var(--text3)">If ${nextAction.type} by ${suggestedTeam}</span>
      </div>
    `;
  } catch (err) {
    if (err.name === 'AbortError') return;
    const predEl = tooltip.querySelector('.tooltip-pred');
    if (predEl) {
      predEl.className = 'tooltip-pred error';
      predEl.textContent = `Prediction unavailable: ${err.message}`;
    }
  }
}

function moveTooltip(e) {
  const tooltip = document.getElementById('tooltip');
  const pad = 12;
  let x = e.clientX + pad;
  let y = e.clientY + pad;

  // Prevent overflow
  const rect = tooltip.getBoundingClientRect();
  if (x + rect.width > window.innerWidth) x = e.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight) y = e.clientY - rect.height - pad;

  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}

function hideTooltip() {
  document.getElementById('tooltip').classList.remove('show');
  if (predictAbort) predictAbort.abort();
}

// ── Start ────────────────────────────────────────────────────────
init();
