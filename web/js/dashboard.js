/* ═══════════════════════════════════════════════════════════════════════════
   Retil Store — Dashboard JavaScript
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
  if (Math.abs(v) >= 1e9) return '$' + (v / 1e9).toFixed(2) + 'B';
  if (Math.abs(v) >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
  if (Math.abs(v) >= 1e3) return '$' + (v / 1e3).toFixed(1) + 'K';
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
  renderMLModels(data.ml_models);
  renderActualVsPred(data.actual_vs_predicted);
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
          ticks: { display: false },
        },
      },
    },
  });
}

/* ════════════════════════════ FEATURE IMPORTANCE ══════════════════════════ */
function renderFeatureImportance(features) {
  const top12 = features.slice(0, 12);
  const ctx   = document.getElementById('chart-feat-importance').getContext('2d');

  const gradient = ctx.createLinearGradient(300, 0, 0, 0);
  gradient.addColorStop(0,   C.gold);
  gradient.addColorStop(0.5, C.teal);
  gradient.addColorStop(1,   C.purple);

  charts.featImportance = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top12.map(f => f.feature.replace(/_/g, ' ')),
      datasets: [{
        data: top12.map(f => (f.importance * 100).toFixed(2)),
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
        callbacks: { label: ctx => `${ctx.parsed.x.toFixed(2)}%` },
      }},
      scales: {
        x: { grid: BASE_GRID, ticks: { callback: v => v + '%' } },
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
  const blues = ['#42a5f5','#5c6bc0','#7e57c2','#ec407a','#26c6da'];

  charts.markdownCorr = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: markdown.map(m => m.markdown),
      datasets: [{
        data: markdown.map(m => m.correlation),
        backgroundColor: blues.map(c => alpha(c, 0.75)),
        borderColor: blues,
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
        callbacks: { label: ctx => ctx.parsed.y.toFixed(4) },
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

/* ════════════════════════════ ML METRICS ══════════════════════════════════ */
function renderMLModels(ml) {
  const rf = ml.random_forest;
  const gb = ml.gradient_boosting;

  document.getElementById('rf-r2-badge').textContent  = `R² ${rf.r2.toFixed(3)}`;
  document.getElementById('rf-r2').textContent        = rf.r2.toFixed(3);
  document.getElementById('rf-mae').textContent       = fmtShort(rf.mae);
  document.getElementById('rf-rmse').textContent      = fmtShort(rf.rmse);

  document.getElementById('gb-r2-badge').textContent  = `R² ${gb.r2.toFixed(3)}`;
  document.getElementById('gb-r2').textContent        = gb.r2.toFixed(3);
  document.getElementById('gb-mae').textContent       = fmtShort(gb.mae);
  document.getElementById('gb-rmse').textContent      = fmtShort(gb.rmse);
}

/* ════════════════════════ ACTUAL VS PREDICTED ═════════════════════════════ */
function renderActualVsPred(avp) {
  const ctx = document.getElementById('chart-actual-vs-pred').getContext('2d');

  const scatter = avp.actual.map((a, i) => ({ x: a, y: avp.predicted_rf[i] }));
  const maxVal  = Math.max(...avp.actual, ...avp.predicted_rf);
  const perfect = [{ x: 0, y: 0 }, { x: maxVal, y: maxVal }];

  charts.actualVsPred = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'Actual vs Predicted',
          data: scatter,
          backgroundColor: alpha(C.gold, 0.55),
          borderColor: C.gold,
          borderWidth: 1,
          pointRadius: 4,
          pointHoverRadius: 6,
        },
        {
          label: 'Perfect Fit',
          data: perfect,
          type: 'line',
          borderColor: C.teal,
          borderWidth: 2,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          tension: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: C.surface, borderColor: C.border, borderWidth: 1,
          callbacks: {
            label: ctx => ctx.dataset.label === 'Perfect Fit'
              ? 'Perfect Fit Line'
              : `Actual: ${fmtShort(ctx.parsed.x)}  Pred: ${fmtShort(ctx.parsed.y)}`,
          },
        },
      },
      scales: {
        x: { grid: BASE_GRID, title: { display: true, text: 'Actual Weekly Sales ($)', color: C.text3 }, ticks: { callback: v => fmtShort(v) } },
        y: { grid: BASE_GRID, title: { display: true, text: 'Predicted Weekly Sales ($)', color: C.text3 }, ticks: { callback: v => fmtShort(v) } },
      },
    },
  });
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
