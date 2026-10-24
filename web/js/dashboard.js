/* ═══════════════════════════════════════════════════════════════════════════
   Retail Store — Dashboard JavaScript
   Chart.js 4.x · Dark theme · Filter-driven interactivity · Groq AI chat
   ═════════════════════════════════════════════════════════════════════════ */

'use strict';

/* ── Palette ────────────────────────────────────────────────────────────── */
const C = {
  teal:   '#26d4aa',
  gold:   '#f5a623',
  purple: '#7c4dff',
  cyan:   '#26c6da',
  red:    '#ef5350',
  green:  '#66bb6a',
  blue:   '#42a5f5',
  text:   '#e2e8f0',
  text2:  '#94a3b8',
  text3:  '#64748b',
  bg:     '#0d1117',
  surface:'#161b2a',
  border: '#253050',
};

const TYPE_COLOR = { A: C.gold, B: C.cyan, C: C.purple };
const REGION_COLOR = { East: C.teal, North: C.gold, South: C.purple, West: C.cyan };
const HOLIDAY_COLORS = [C.gold, C.red, C.teal, C.green, C.purple];
const SEASON_COLOR = { Spring: C.green, Summer: C.gold, Fall: C.red, Winter: C.cyan };

/* ── Global Chart defaults ──────────────────────────────────────────────── */
Chart.defaults.color        = C.text2;
Chart.defaults.font.family  = "'Inter', system-ui, sans-serif";
Chart.defaults.font.size    = 11;
Chart.defaults.borderColor  = C.border;

const BASE_GRID = {
  color: 'rgba(37,48,80,0.5)',
  drawBorder: false,
};

/* ── Utility ────────────────────────────────────────────────────────────── */
const fmt = (v, decimals = 0) => new Intl.NumberFormat('en-US', {
  style: 'currency', currency: 'USD',
  minimumFractionDigits: decimals, maximumFractionDigits: decimals,
}).format(v);

const fmtShort = (v) => {
  const sign = v < 0 ? '-' : '';
  const a = Math.abs(v);
  if (a >= 1e9) return `${sign}$${(a / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `${sign}$${(a / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `${sign}$${(a / 1e3).toFixed(1)}K`;
  return fmt(v);
};

const alpha = (hex, a) => {
  const r = parseInt(hex.slice(1,3),16);
  const g = parseInt(hex.slice(3,5),16);
  const b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
};

/* ── Filter state ───────────────────────────────────────────────────────── */
const filters = { year: 'all', region: 'all', type: 'all', season: 'all' };

/* ── Chart registry ─────────────────────────────────────────────────────── */
const charts = {};

/* ══════════════════════════════ INITIALISATION ════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  fetch('data/dashboard_data.json')
    .then(r => r.json())
    .then(data => {
      window.__dashData = data;
      initFilterChips();
      renderAll(data);
    })
    .catch(err => console.error('Failed to load dashboard data:', err));
});

/* ── Filter chip wiring ─────────────────────────────────────────────────── */
function initFilterChips() {
  document.querySelectorAll('.filter-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const group = chip.dataset.group;
      const val   = chip.dataset.value;
      // deactivate siblings
      document.querySelectorAll(`.filter-chip[data-group="${group}"]`).forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      filters[group] = val;
      updateFilteredCharts(window.__dashData);
    });
  });
}

/* ── Apply filters to store_weekly ──────────────────────────────────────── */
function filterStoreWeekly(data) {
  return data.store_weekly.filter(row => {
    if (filters.year   !== 'all' && row.year      !== parseInt(filters.year))   return false;
    if (filters.region !== 'all' && row.region    !== filters.region)           return false;
    if (filters.type   !== 'all' && row.store_type !== filters.type)            return false;
    if (filters.season !== 'all' && row.season    !== filters.season)           return false;
    return true;
  });
}

/* ══════════════════════════════ RENDER ALL ════════════════════════════════ */
function renderAll(data) {
  renderKPIs(data.kpis);
  renderMonthlyRevenue(data.monthly_revenue);
  renderTypeDonut(data.by_type);
  renderRegionBar(data.by_region);
  renderWeeklyTrend(data.weekly_type_trend);
  renderHolidayImpact(data.holiday_impact, data.non_holiday_baseline);
  renderSeasonalRadar(data.seasonal_pattern);
  renderFeatureImportance(data.feature_importance);
  renderPearsonCorr(data.pearson_correlation);
  renderMarkdownCorr(data.markdown_correlation);
  renderTopDepts(data.top_departments);
  renderEvaluation(data.evaluation);
  renderHoldoutSeries(data.evaluation.weekly);
  renderWeeklyError(data.evaluation.weekly);
  renderResiduals(data.evaluation.residuals);
  renderHorizons(data.evaluation.horizons);
  renderForecast(data.forecast);
}

/* ── Update charts driven by filters ───────────────────────────────────── */
function updateFilteredCharts(data) {
  const filtered = filterStoreWeekly(data);
  const activeFilters = Object.entries(filters).filter(([,v]) => v !== 'all').map(([k,v]) => `${k}: ${v}`);
  const badgeText = activeFilters.length ? activeFilters.join(' · ') : 'All Data';

  document.querySelectorAll('.chart-badge').forEach(b => (b.textContent = badgeText));

  if (!filtered.length) {
    console.warn('No data matches current filters.');
    return;
  }

  // ── KPIs from filtered data
  const total = filtered.reduce((s, r) => s + r.revenue, 0);
  const avg   = total / filtered.length;
  const holRows    = filtered.filter(r => r.is_holiday === 1);
  const nonHolRows = filtered.filter(r => r.is_holiday === 0);
  const holAvg  = holRows.length ? holRows.reduce((s,r) => s+r.revenue, 0) / holRows.length : 0;
  const nolAvg  = nonHolRows.length ? nonHolRows.reduce((s,r) => s+r.revenue, 0) / nonHolRows.length : 1;
  const lift    = ((holAvg - nolAvg) / nolAvg) * 100;

  document.getElementById('kpi-total-sales').textContent = fmtShort(total);
  document.getElementById('kpi-avg-weekly').textContent  = fmtShort(avg);
  document.getElementById('kpi-holiday-lift').textContent = (holRows.length ? `+${lift.toFixed(1)}%` : 'N/A');
  document.getElementById('kpi-holiday-lift-sub').textContent = 'Raw, for the filtered data';

  // ── Monthly revenue
  const byMonth = {};
  filtered.forEach(r => {
    const m = r.date.slice(0, 7);
    byMonth[m] = (byMonth[m] || 0) + r.revenue;
  });
  const labels  = Object.keys(byMonth).sort();
  const revenue = labels.map(k => byMonth[k]);
  if (charts.monthly) {
    charts.monthly.data.labels = labels;
    charts.monthly.data.datasets[0].data = revenue;
    charts.monthly.update('none');
  }

  // ── Revenue by type
  const byType = {};
  filtered.forEach(r => {
    byType[r.store_type] = byType[r.store_type] || { rev: 0, cnt: 0 };
    byType[r.store_type].rev += r.revenue;
    byType[r.store_type].cnt += 1;
  });
  const types    = Object.keys(byType).sort();
  const typeRevs = types.map(t => byType[t].rev);
  if (charts.typeDonut) {
    charts.typeDonut.data.labels = types.map(t => `Type ${t}`);
    charts.typeDonut.data.datasets[0].data = typeRevs;
    charts.typeDonut.data.datasets[0].backgroundColor = types.map(t => TYPE_COLOR[t] || C.text3);
    charts.typeDonut.update('none');
  }
  buildTypeTable(types.map(t => ({
    type: t,
    revenue: byType[t].rev,
    avg_week: byType[t].rev / byType[t].cnt,
    num_stores: data.by_type.find(x => x.type === t)?.num_stores || '—',
  })));

  // ── Revenue by region
  const byRegion = {};
  filtered.forEach(r => {
    byRegion[r.region] = (byRegion[r.region] || 0) + r.revenue;
  });
  const regions   = Object.keys(byRegion).sort((a,b) => byRegion[b] - byRegion[a]);
  const regionRevs = regions.map(r => byRegion[r]);
  if (charts.regionBar) {
    charts.regionBar.data.labels = regions;
    charts.regionBar.data.datasets[0].data = regionRevs;
    charts.regionBar.data.datasets[0].backgroundColor = regions.map(r => REGION_COLOR[r] || C.text3);
    charts.regionBar.update('none');
  }
  buildRegionDetail(regions, byRegion, data.by_region);
}

/* ══════════════════════════════ KPI RENDERING ═════════════════════════════ */
function renderKPIs(kpis) {
  document.getElementById('kpi-total-sales').textContent   = fmtShort(kpis.total_sales);
  document.getElementById('kpi-avg-weekly').textContent    = fmtShort(kpis.avg_weekly_sales);
  document.getElementById('kpi-holiday-lift').textContent  = `+${kpis.holiday_lift_pct.toFixed(1)}%`;
  document.getElementById('kpi-holiday-lift-sub').textContent =
    `Raw · +${kpis.holiday_lift_controlled_pct.toFixed(1)}% within the same store-dept`;
  document.getElementById('kpi-max-store').textContent     = fmtShort(kpis.max_store_revenue);
}

/* ══════════════════════════ MONTHLY REVENUE LINE ══════════════════════════ */
function renderMonthlyRevenue(d) {
  const ctx = document.getElementById('chart-monthly-revenue').getContext('2d');

  const gradient = ctx.createLinearGradient(0, 0, 0, 300);
  gradient.addColorStop(0, alpha(C.gold, 0.25));
  gradient.addColorStop(1, alpha(C.gold, 0.0));

  charts.monthly = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.labels,
      datasets: [{
        data: d.revenue,
        borderColor: C.gold,
        backgroundColor: gradient,
        borderWidth: 2,
        pointRadius: 4,
        pointBackgroundColor: C.gold,
        pointBorderColor: C.bg,
        pointBorderWidth: 2,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface,
        borderColor: C.border, borderWidth: 1,
        callbacks: { label: ctx => fmtShort(ctx.parsed.y) },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { maxTicksLimit: 12, maxRotation: 45 } },
        y: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });
}

/* ════════════════════════════ TYPE DONUT ══════════════════════════════════ */
function renderTypeDonut(byType) {
  const ctx = document.getElementById('chart-type-donut').getContext('2d');
  charts.typeDonut = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: byType.map(t => `Type ${t.type}`),
      datasets: [{
        data: byType.map(t => t.revenue),
        backgroundColor: byType.map(t => TYPE_COLOR[t.type] || C.text3),
        borderColor: C.surface,
        borderWidth: 3,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: '70%',
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
          callbacks: { label: ctx => `${ctx.label}: ${fmtShort(ctx.parsed)}` },
        },
      },
    },
  });
  buildTypeTable(byType);
}

function buildTypeTable(byType) {
  const tbody = document.getElementById('type-table-body');
  if (!tbody) return;
  tbody.innerHTML = byType.map(t => `
    <tr>
      <td><span class="type-tag ${t.type}">${t.type}</span></td>
      <td>${fmtShort(t.revenue)}</td>
      <td>${fmtShort(t.avg_week)}</td>
      <td>${t.num_stores ?? '—'} stores</td>
    </tr>
  `).join('');
}

/* ════════════════════════════ REGION BAR ══════════════════════════════════ */
function renderRegionBar(byRegion) {
  const ctx = document.getElementById('chart-region-bar').getContext('2d');
  const regions = byRegion.map(r => r.region);
  const revenues = byRegion.map(r => r.revenue);

  charts.regionBar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: regions,
      datasets: [{
        data: revenues,
        backgroundColor: regions.map(r => alpha(REGION_COLOR[r] || C.text3, 0.75)),
        borderColor:     regions.map(r => REGION_COLOR[r] || C.text3),
        borderWidth: 1.5,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: { label: ctx => fmtShort(ctx.parsed.x) },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
        y: { grid: { display: false } },
      },
    },
  });
  buildRegionDetail(regions, Object.fromEntries(byRegion.map(r => [r.region, r.revenue])), byRegion);
}

function buildRegionDetail(regions, byRegion, fullByRegion) {
  const total = Object.values(byRegion).reduce((s, v) => s + v, 0);
  const el = document.getElementById('region-detail');
  if (!el) return;
  el.innerHTML = regions.map(r => {
    const rev = byRegion[r] || 0;
    const pct = total ? ((rev / total) * 100).toFixed(1) : '0';
    const orig = fullByRegion.find ? fullByRegion.find(x => x.region === r) : null;
    const storeInfo = orig ? ` · ${orig.stores} stores` : '';
    return `<div class="region-row">
      <span class="region-label" style="color:${REGION_COLOR[r]||C.text2}">${r}</span>
      <span class="region-range">${fmtShort(rev)} &nbsp; ${pct}%${storeInfo}</span>
    </div>`;
  }).join('');
}

/* ════════════════════════════ WEEKLY TREND ════════════════════════════════ */
function renderWeeklyTrend(d) {
  const ctx = document.getElementById('chart-weekly-trend').getContext('2d');
  charts.weeklyTrend = new Chart(ctx, {
    type: 'line',
    data: {
      labels: d.labels,
      datasets: [
        {
          label: 'Type A', data: d.type_A,
          borderColor: C.gold, backgroundColor: alpha(C.gold, 0.0),
          borderWidth: 2, pointRadius: 0, tension: 0.3,
        },
        {
          label: 'Type B', data: d.type_B,
          borderColor: C.cyan, backgroundColor: alpha(C.cyan, 0.0),
          borderWidth: 2, pointRadius: 0, tension: 0.3,
        },
        {
          label: 'Type C', data: d.type_C,
          borderColor: C.purple, backgroundColor: alpha(C.purple, 0.0),
          borderWidth: 2, pointRadius: 0, tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: { label: ctx => `${ctx.dataset.label}: ${fmtShort(ctx.parsed.y)}` },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { maxTicksLimit: 10, maxRotation: 45 } },
        y: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });
}

/* ════════════════════════════ HOLIDAY IMPACT ══════════════════════════════ */
function renderHolidayImpact(holidays, baseline) {
  const ctx = document.getElementById('chart-holiday-impact').getContext('2d');
  // include baseline
  const allLabels = ['Non-Holiday', ...holidays.map(h => h.holiday)];
  const allValues = [baseline,     ...holidays.map(h => h.avg_sales)];
  const allColors = [C.text3,      ...HOLIDAY_COLORS];

  charts.holidayImpact = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: allLabels,
      datasets: [{
        data: allValues,
        backgroundColor: allColors.map(c => alpha(c, 0.75)),
        borderColor:     allColors,
        borderWidth: 1.5,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: { label: ctx => fmtShort(ctx.parsed.y) },
      }},
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 30 } },
        y: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });
}

/* ════════════════════════════ SEASONAL RADAR ══════════════════════════════ */
function renderSeasonalRadar(seasonal) {
  const ctx = document.getElementById('chart-seasonal-radar').getContext('2d');
  const labels = seasonal.map(s => s.season);
  const values = seasonal.map(s => s.avg_sales);

  charts.seasonalRadar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: C.teal,
        backgroundColor: alpha(C.teal, 0.15),
        borderWidth: 2,
        pointBackgroundColor: C.teal,
        pointBorderColor: C.bg,
        pointRadius: 5,
        pointBorderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: { label: ctx => fmtShort(ctx.parsed.r) },
      }},
      scales: {
        r: {
          grid: { color: 'rgba(37,48,80,0.6)' },
          angleLines: { color: 'rgba(37,48,80,0.6)' },
          pointLabels: { color: C.text2, font: { size: 12, weight: '600' } },
          beginAtZero: true,
          ticks: { display: false },
        },
      },
    },
  });
}

/* ═══════════════════════ PERMUTATION IMPORTANCE ═══════════════════════════ */
function renderFeatureImportance(features) {
  const top = features.filter(f => f.wmae_increase > 0).slice(0, 12);
  const ctx = document.getElementById('chart-feat-importance').getContext('2d');

  const gradient = ctx.createLinearGradient(300, 0, 0, 0);
  gradient.addColorStop(0,   C.gold);
  gradient.addColorStop(0.5, C.teal);
  gradient.addColorStop(1,   C.purple);

  charts.featImportance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top.map(f => f.feature.replace(/_/g, ' ')),
      datasets: [{
        data: top.map(f => f.wmae_increase),
        backgroundColor: gradient,
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: {
          label: c => `+${fmtShort(c.parsed.x)} WMAE when shuffled ` +
                      `(${(top[c.dataIndex].importance * 100).toFixed(1)}% of total)`,
        },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) },
             title: { display: true, text: 'WMAE increase', color: C.text3 } },
        y: { grid: { display: false }, ticks: { font: { size: 10 } } },
      },
    },
  });
}

/* ════════════════════════════ PEARSON CORRELATION ═════════════════════════ */
function renderPearsonCorr(pearson) {
  const sorted = [...pearson].sort((a, b) => b.correlation - a.correlation);
  const ctx = document.getElementById('chart-pearson').getContext('2d');

  charts.pearsonCorr = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(f => f.feature.replace(/_/g, ' ')),
      datasets: [{
        data: sorted.map(f => f.correlation),
        backgroundColor: sorted.map(f => f.correlation >= 0 ? alpha(C.teal, 0.75) : alpha(C.red, 0.75)),
        borderColor:     sorted.map(f => f.correlation >= 0 ? C.teal : C.red),
        borderWidth: 1.5,
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: 'y',
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: { label: ctx => ctx.parsed.x.toFixed(4) },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { callback: v => v.toFixed(2) } },
        y: { grid: { display: false }, ticks: { font: { size: 10 } } },
      },
    },
  });
}

/* ════════════════════════════ MARKDOWN CORR ═══════════════════════════════ */
function renderMarkdownCorr(markdown) {
  const ctx = document.getElementById('chart-markdown-corr').getContext('2d');

  charts.markdownCorr = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: markdown.map(m => m.markdown),
      datasets: [
        {
          label: 'All weeks', data: markdown.map(m => m.correlation),
          backgroundColor: alpha(C.gold, 0.75), borderColor: C.gold, borderWidth: 1.5,
          borderRadius: 6, borderSkipped: false,
        },
        {
          label: 'Non-holiday weeks', data: markdown.map(m => m.correlation_non_holiday),
          backgroundColor: alpha(C.teal, 0.75), borderColor: C.teal, borderWidth: 1.5,
          borderRadius: 6, borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: { label: c => `${c.dataset.label}: r = ${c.parsed.y.toFixed(4)}` },
      }},
      scales: {
        x: { grid: { display: false } },
        y: { grid: BASE_GRID, ticks: { callback: v => v.toFixed(2) } },
      },
    },
  });
}

/* ════════════════════════════ TOP DEPARTMENTS ═════════════════════════════ */
function renderTopDepts(depts) {
  const ctx = document.getElementById('chart-top-depts').getContext('2d');

  const grad = ctx.createLinearGradient(0, 0, 0, 300);
  grad.addColorStop(0, alpha(C.purple, 0.85));
  grad.addColorStop(1, alpha(C.purple, 0.35));

  charts.topDepts = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: depts.map(d => d.dept),
      datasets: [{
        data: depts.map(d => d.revenue),
        backgroundColor: grad,
        borderColor: C.purple,
        borderWidth: 1,
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: { label: ctx => fmtShort(ctx.parsed.y) },
      }},
      scales: {
        x: { grid: { display: false } },
        y: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });
}

/* ═══════════════════════════ MODEL EVALUATION ═════════════════════════════ */
const pct = (v, digits = 1, signed = false) =>
  `${signed && v > 0 ? '+' : ''}${(v * 100).toFixed(digits)}%`;

function renderEvaluation(ev) {
  const best  = ev.models.find(m => m.model === 'LightGBM + lags');
  const naive = ev.models.find(m => m.model.startsWith('Seasonal naive'));
  const h     = ev.holdout;

  document.getElementById('eval-wmae').textContent     = fmtShort(best.wmae);
  document.getElementById('eval-skill').textContent    = pct(best.skill_vs_naive, 1, true);
  document.getElementById('eval-wape').textContent     = pct(best.wape);
  document.getElementById('eval-bias-sub').textContent = `Bias ${pct(best.bias, 1, true)} (+ = over-forecast)`;
  document.getElementById('eval-coverage').textContent = pct(ev.interval.coverage);
  document.getElementById('eval-protocol').textContent =
    `One-week-ahead forecasts · trained on data before ${h.test_start}, ` +
    `scored on ${h.test_start} → ${h.test_end} (${h.test_rows.toLocaleString('en-US')} store-dept-weeks)`;

  const bestWmae = Math.min(...ev.models.map(m => m.wmae));
  document.getElementById('model-table-body').innerHTML = ev.models.map(m => {
    const cls = m.kind === 'baseline' ? 'baseline' : (m.wmae === bestWmae ? 'best' : '');
    const skillCls = m.skill_vs_naive < 0 ? 'skill-neg' : '';
    return `<tr class="${cls}">
      <td>${m.model}${m.kind === 'baseline' ? '<span class="kind-tag">baseline</span>' : ''}</td>
      <td>${fmtShort(m.wmae)}</td>
      <td>${pct(m.wape)}</td>
      <td>${pct(m.bias, 1, true)}</td>
      <td>${fmtShort(m.mae)}</td>
      <td>${fmtShort(m.rmse)}</td>
      <td class="${skillCls}">${pct(m.skill_vs_naive, 1, true)}</td>
    </tr>`;
  }).join('');

  document.getElementById('model-table-notes').innerHTML =
    `WMAE weights holiday weeks 5&times;, as in the original Kaggle competition. ` +
    `Skill is measured against the seasonal naive forecast (same week last year, ${fmtShort(naive.wmae)} WMAE). ` +
    `Adding temperature, fuel price, CPI, unemployment and markdowns back changes WMAE from ` +
    `${fmt(ev.ablation.without_exogenous_wmae)} to ${fmt(ev.ablation.with_exogenous_wmae)}, ` +
    `so they are left out of the model.`;
}

/* Shades holiday weeks behind a category-axis chart. */
const holidayShading = (flags) => ({
  id: 'holidayShading',
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea, scales: { x } } = chart;
    const half = (x.getPixelForValue(1) - x.getPixelForValue(0)) / 2;
    ctx.save();
    ctx.fillStyle = alpha(C.red, 0.10);
    flags.forEach((f, i) => {
      if (!f) return;
      const cx = x.getPixelForValue(i);
      ctx.fillRect(cx - half, chartArea.top, half * 2, chartArea.bottom - chartArea.top);
    });
    ctx.restore();
  },
});

const weekLabel = (w, i) => (w.holiday[i] ? `${w.dates[i]} · ${w.holiday[i]}` : w.dates[i]);

function renderHoldoutSeries(w) {
  const ctx = document.getElementById('chart-holdout-series').getContext('2d');
  const line = (label, data, color, extra = {}) => ({
    label, data, borderColor: color, backgroundColor: color,
    borderWidth: 2, pointRadius: 0, pointHoverRadius: 4, tension: 0.25, ...extra,
  });
  charts.holdoutSeries = new Chart(ctx, {
    type: 'line',
    data: {
      labels: w.dates,
      datasets: [
        line('Actual', w.actual, C.text, { borderWidth: 2.5 }),
        line('LightGBM', w.lightgbm, C.teal),
        line('Seasonal naive', w.seasonal_naive, C.gold, { borderDash: [5, 4], borderWidth: 1.5 }),
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: {
          title: items => weekLabel(w, items[0].dataIndex),
          label: c => `${c.dataset.label}: ${fmtShort(c.parsed.y)}`,
        },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { maxTicksLimit: 10, maxRotation: 0 } },
        y: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
      },
    },
    plugins: [holidayShading(w.is_holiday)],
  });
}

function renderWeeklyError(w) {
  const ctx = document.getElementById('chart-weekly-error').getContext('2d');
  charts.weeklyError = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: w.dates,
      datasets: [
        {
          label: 'LightGBM', data: w.mae_lightgbm,
          backgroundColor: w.is_holiday.map(h => alpha(h ? C.red : C.teal, 0.8)),
          borderRadius: 3, borderSkipped: false,
        },
        {
          label: 'Seasonal naive', data: w.mae_seasonal_naive, type: 'line',
          borderColor: C.gold, borderWidth: 1.5, borderDash: [5, 4], pointRadius: 0, tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: {
          title: items => weekLabel(w, items[0].dataIndex),
          label: c => `${c.dataset.label} MAE: ${fmtShort(c.parsed.y)}`,
        },
      }},
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 10, maxRotation: 0 } },
        y: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });
}

function renderResiduals(r) {
  const ctx = document.getElementById('chart-residuals').getContext('2d');
  const mids = r.bin_start.map((s, i) => (s + r.bin_end[i]) / 2);
  charts.residuals = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: mids.map(m => fmtShort(m)),
      datasets: [{
        data: r.count,
        backgroundColor: mids.map(m => alpha(m < 0 ? C.gold : C.teal, 0.75)),
        borderRadius: 2, barPercentage: 1, categoryPercentage: 0.95,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: {
          title: items => {
            const i = items[0].dataIndex;
            return `${fmtShort(r.bin_start[i])} to ${fmtShort(r.bin_end[i])}`;
          },
          label: c => `${c.parsed.y.toLocaleString('en-US')} store-dept-weeks`,
        },
      }},
      scales: {
        x: { grid: { display: false }, ticks: { maxTicksLimit: 8, maxRotation: 0 } },
        y: { grid: BASE_GRID, ticks: { callback: v => v.toLocaleString('en-US') } },
      },
    },
  });
}

function renderHorizons(hz) {
  const ctx = document.getElementById('chart-horizons').getContext('2d');
  charts.horizons = new Chart(ctx, {
    type: 'line',
    data: {
      labels: hz.map(h => `${h.horizon}w`),
      datasets: [
        {
          label: 'LightGBM', data: hz.map(h => h.wmae),
          borderColor: C.teal, backgroundColor: C.teal, borderWidth: 2, pointRadius: 4,
        },
        {
          label: 'Seasonal naive', data: hz.map(h => h.naive_wmae),
          borderColor: C.gold, borderDash: [5, 4], borderWidth: 1.5, pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        callbacks: {
          title: items => `${hz[items[0].dataIndex].horizon}-week-ahead`,
          label: c => `${c.dataset.label} WMAE: ${fmtShort(c.parsed.y)}`,
          afterBody: items => `Skill vs naive: ${pct(hz[items[0].dataIndex].skill_vs_naive, 1, true)}`,
        },
      }},
      scales: {
        x: { grid: BASE_GRID, title: { display: true, text: 'Weeks ahead', color: C.text3 } },
        y: { grid: BASE_GRID, beginAtZero: true, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });
}

/* ═══════════════════════════════ FORECAST ═════════════════════════════════ */
function renderForecast(f) {
  const ctx = document.getElementById('chart-forecast').getContext('2d');
  const hist = f.history;
  const labels = [...hist.dates, ...f.dates];
  const pad = new Array(hist.dates.length - 1).fill(null);
  const lastActual = hist.actual[hist.actual.length - 1];
  // Forecast and band start at the last actual week so the lines join up.
  const fc    = [...pad, lastActual, ...f.forecast];
  const lower = [...pad, lastActual, ...f.lower];
  const upper = [...pad, lastActual, ...f.upper];
  const label = d => (f.holidays[d] ? `${d} · ${f.holidays[d]}` : d);

  charts.forecast = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Upper', data: upper, borderWidth: 0, pointRadius: 0, fill: false },
        {
          label: '80% band', data: lower, borderWidth: 0, pointRadius: 0,
          backgroundColor: alpha(C.teal, 0.18), fill: '-1',
        },
        {
          label: 'Actual', data: [...hist.actual, ...new Array(f.dates.length).fill(null)],
          borderColor: C.text, borderWidth: 2.5, pointRadius: 0, tension: 0.25,
        },
        {
          label: 'Forecast', data: fc, borderColor: C.teal, borderWidth: 2.5,
          borderDash: [6, 4], pointRadius: fc.map((v, i) => (i >= hist.dates.length ? 4 : 0)),
          pointBackgroundColor: C.teal, tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
        filter: item => item.raw !== null && item.dataset.label !== 'Upper'
          && !(item.dataset.label !== 'Actual' && item.dataIndex < hist.dates.length),
        callbacks: {
          title: items => label(labels[items[0].dataIndex]),
          label: c => {
            if (c.dataset.label === '80% band') {
              const j = c.dataIndex - hist.dates.length;
              return `80% band: ${fmtShort(f.lower[j])} – ${fmtShort(f.upper[j])}`;
            }
            return `${c.dataset.label}: ${fmtShort(c.parsed.y)}`;
          },
        },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { maxTicksLimit: 12, maxRotation: 0 } },
        y: { grid: BASE_GRID, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });

  document.getElementById('forecast-sub').textContent =
    `${f.dates[0]} → ${f.dates[f.dates.length - 1]} · one direct LightGBM model per horizon, refit on all history`;

  const types = Object.keys(f.by_store_type).sort();
  document.getElementById('forecast-table-head').innerHTML =
    `<tr><th>Week</th><th>Total</th><th>80% band</th>${types.map(t => `<th>Type ${t}</th>`).join('')}</tr>`;
  document.getElementById('forecast-table-body').innerHTML = f.dates.map((d, i) => `
    <tr>
      <td>${d}${f.holidays[d] ? `<span class="kind-tag">${f.holidays[d]}</span>` : ''}</td>
      <td>${fmtShort(f.forecast[i])}</td>
      <td>${fmtShort(f.lower[i])} – ${fmtShort(f.upper[i])}</td>
      ${types.map(t => `<td>${fmtShort(f.by_store_type[t][i])}</td>`).join('')}
    </tr>`).join('');
}

/* ══════════════════════════════ FILTER PANEL ══════════════════════════════ */
function toggleFilters() {
  const panel   = document.getElementById('filter-panel');
  const overlay = document.getElementById('filter-overlay');
  panel.classList.toggle('open');
  overlay.classList.toggle('open');
}

function resetFilters() {
  ['year', 'region', 'type', 'season'].forEach(g => { filters[g] = 'all'; });
  document.querySelectorAll('.filter-chip').forEach(c => {
    c.classList.toggle('active', c.dataset.value === 'all');
  });

  if (!window.__dashData) return;
  const data = window.__dashData;

  // KPIs
  renderKPIs(data.kpis);

  // Monthly chart
  if (charts.monthly) {
    charts.monthly.data.labels = data.monthly_revenue.labels;
    charts.monthly.data.datasets[0].data = data.monthly_revenue.revenue;
    charts.monthly.update('none');
  }

  // Type donut
  if (charts.typeDonut) {
    charts.typeDonut.data.labels = data.by_type.map(t => `Type ${t.type}`);
    charts.typeDonut.data.datasets[0].data = data.by_type.map(t => t.revenue);
    charts.typeDonut.data.datasets[0].backgroundColor = data.by_type.map(t => TYPE_COLOR[t.type] || C.text3);
    charts.typeDonut.update('none');
  }
  buildTypeTable(data.by_type);

  // Region bar
  const regions = data.by_region.map(r => r.region);
  if (charts.regionBar) {
    charts.regionBar.data.labels = regions;
    charts.regionBar.data.datasets[0].data = data.by_region.map(r => r.revenue);
    charts.regionBar.data.datasets[0].backgroundColor = regions.map(r => alpha(REGION_COLOR[r] || C.text3, 0.75));
    charts.regionBar.data.datasets[0].borderColor = regions.map(r => REGION_COLOR[r] || C.text3);
    charts.regionBar.update('none');
  }
  buildRegionDetail(
    regions,
    Object.fromEntries(data.by_region.map(r => [r.region, r.revenue])),
    data.by_region
  );

  document.querySelectorAll('.chart-badge').forEach(b => (b.textContent = 'All Data'));
}

/* ══════════════════════════════ SCROLL HELPERS ════════════════════════════ */
function scrollToChat() {
  document.getElementById('ai-assistant-section').scrollIntoView({ behavior: 'smooth' });
}

/* ══════════════════════════════ GROQ AI CHAT ══════════════════════════════ */
let groqApiKey = '';

function activateChat() {
  const key = document.getElementById('groq-api-key').value.trim();
  if (!key || !key.startsWith('gsk_')) {
    alert('Please enter a valid Groq API key starting with gsk_');
    return;
  }
  groqApiKey = key;
  document.getElementById('api-key-section').classList.add('hidden');
  document.getElementById('chat-interface').classList.remove('hidden');
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg   = input.value.trim();
  if (!msg) return;
  input.value = '';

  appendMsg('user', msg);
  const status = document.getElementById('chat-status');
  status.textContent = 'Thinking…';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, apiKey: groqApiKey }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    appendMsg('assistant', data.reply);
    status.textContent = '';
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
    appendMsg('assistant', `Sorry, I encountered an error: ${err.message}`);
  }
}

function appendMsg(role, text) {
  const msgs = document.getElementById('chat-messages');
  const div  = document.createElement('div');
  div.className = `chat-msg ${role}`;
  div.innerHTML = `<strong>${role === 'user' ? 'You' : 'Assistant'}:</strong> ${escapeHtml(text)}`;
  msgs.appendChild(div);
  msgs.scrollTop = msgs.scrollHeight;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}
