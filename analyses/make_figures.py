"""Draw Figures 2-5 and S1 from the JSON outputs of parts A-C."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from soilloop import Params, equilibria, state_at, simulate
c = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']; ink = '#52514e'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': ink, 'xtick.color': ink, 'ytick.color': ink,
                     'axes.spines.top': False, 'axes.spines.right': False})
A = json.load(open('part_a.json')); B = json.load(open('part_b_anchored.json')); C = json.load(open('part_c.json'))
W = json.load(open('part_b_wide.json'))
P = Params(); e = equilibria(P); trap = state_at(e[0], P)
def grid(ax):
    for a in np.ravel(ax): a.grid(alpha=.25, lw=.5)
# ---- Fig 2
fig, ax = plt.subplots(2, 2, figsize=(10, 7.2))
a = ax[0, 0]; st, un = [], []
for p in np.linspace(0.01, 0.12, 300):
    r = equilibria(P.with_(p=p))
    for i, s in enumerate(r): (un if (len(r) == 3 and i == 1) else st).append((p, s))
st, un = np.array(st), np.array(un)
a.plot(st[:, 0], st[:, 1], '.', ms=2.5, color=c[0], label='Stable state'); a.plot(un[:, 0], un[:, 1], '.', ms=2.5, color='#9b9a94', label='Unstable threshold')
a.set_xlabel('Degradation pressure on uncared land, p (1/yr)'); a.set_ylabel('Soil condition at equilibrium, S*'); a.legend(frameon=False)
a.set_title('A  Two stable states and a tipping zone', loc='left', fontweight='bold')
a = ax[0, 1]
for i, f in enumerate([0.15, 0.19, 0.23, 0.35]):
    t, y = simulate(P, [trap[0] + f * (1 - trap[0]), trap[1], trap[2]], years=300, points=900)
    a.plot(t, y[0], lw=2, color=c[i], label=f'Cleanup closes {int(f * 100)}% of the gap')
a.set_xlabel('Years after cleanup'); a.set_ylabel('Soil condition, S'); a.legend(frameon=False, fontsize=8)
a.set_title('B  Below the threshold, cleanup relapses', loc='left', fontweight='bold')
a = ax[1, 0]; per = [(r, g, t) for r, g, t in A['persistence']]
rs = [r for r, g, t in per if g == 'bistable' and t is not None and not np.isnan(t)]; ts = [t * 100 for r, g, t in per if g == 'bistable' and t is not None and not np.isnan(t)]
trap_max = max(r for r, g, t in per if g == 'trap only'); heal_min = min(r for r, g, t in per if g == 'healthy only')
a.axvspan(0.008, trap_max + 0.00025, color='#e34948', alpha=.10, lw=0); a.axvspan(heal_min - 0.00025, 0.026, color='#1baf7a', alpha=.12, lw=0)
a.plot(rs, ts, '-o', ms=5, lw=2, color=c[0]); a.set_xlim(0.008, 0.026); a.set_ylim(0, 105)
a.text(0.0105, 52, 'Trap only:\nno one-off\ncleanup escapes', ha='center', fontsize=8); a.text(0.0233, 52, 'No trap:\nthe system\nrecovers alone', ha='center', fontsize=8)
a.set_xlabel('Natural soil recovery rate, r (1/yr)  ← longer residence'); a.set_ylabel('Minimum cleanup to escape (% of gap)')
a.set_title('C  Contaminant persistence deepens the trap', loc='left', fontweight='bold')
a = ax[1, 1]; cm = np.array(A['community']); a.plot(cm[:, 0], cm[:, 1] * 100, '-o', ms=5, lw=2, color=c[1])
a.set_xlabel('Community recovery time, 1/d (years)'); a.set_ylabel('Minimum cleanup to escape (% of gap)')
a.set_title('D  Slow community recovery raises the bar', loc='left', fontweight='bold'); grid(ax)
fig.tight_layout(); fig.savefig('fig2.png', dpi=160, facecolor='#fcfcfb')
# ---- Fig 3
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2)); a = ax[0]; dev = {}
for i, tau in enumerate(['5', '10', '20']):
    f = np.array([(x, u) for x, u in A['frontier'][tau]], float); m = ~np.isnan(f[:, 1])
    a.plot(f[m, 0] * 100, f[m, 1] * 100, '-o', ms=3.5, lw=2, color=c[i], label=f'Pressure reduced for {tau} years')
    if tau != '5':
        u0 = f[0, 1]; x0 = f[np.argmax(f[:, 1] == 0), 0]; a.plot([0, x0 * 100], [u0 * 100, 0], '--', lw=1, color=c[i], alpha=.6)
        dev[tau] = float(np.nanmax((u0 * (1 - f[:, 0] / x0) - f[:, 1])[f[:, 0] < x0]))
a.set_xlabel('Cleanup (% of the soil gap closed)'); a.set_ylabel('Pressure reduction needed (%)'); a.legend(frameon=False, fontsize=8)
a.text(14, 80, 'Escape', fontsize=9); a.text(1, 8, 'Relapse', fontsize=9)
a.set_title('A  Trade-off between cleanup and land care', loc='left', fontweight='bold')
a = ax[1]
for (x, u, lab, col) in [(0.125, 0, 'Cleanup 12.5% alone', c[0]), (0, 0.45, 'Pressure −45% for 10 yr alone', c[1]), (0.125, 0.45, 'Both together', c[2])]:
    t, y = simulate(P, [trap[0] + x * (1 - trap[0]), trap[1], trap[2]], years=300, land_care=u, land_care_years=10, points=900)
    a.plot(t, y[0], lw=2, color=col, label=lab)
a.axhline(e[1], color='#9b9a94', lw=1, ls=':'); a.text(300, e[1] + 0.005, 'threshold', ha='right', fontsize=8, color=ink)
a.set_xlabel('Years after intervention'); a.set_ylabel('Soil condition, S'); a.legend(frameon=False, fontsize=8)
a.set_title('B  Each alone fails, together they succeed', loc='left', fontweight='bold'); grid(ax)
fig.tight_layout(); fig.savefig('fig3.png', dpi=160, facecolor='#fcfcfb')
json.dump(dev, open('fig3_additivity_deviation.json', 'w'))
# ---- Fig 4
col = {'healthy only': '#1baf7a', 'bistable': '#eda100', 'trap only': '#e34948'}; lab = {'healthy only': 'Healthy state only', 'bistable': 'Two stable states', 'trap only': 'Trap only'}
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2)); a = ax[0]; grp = ['Long residence\n(> 250 yr)', 'Intermediate\n(70–250 yr)', 'Short residence\n(< 70 yr)']; bot = np.zeros(3)
for k in ['trap only', 'bistable', 'healthy only']:
    v = np.array([B[g][k] * 100 for g in ('long', 'intermediate', 'short')]); a.bar(grp, v, bottom=bot, color=col[k], label=lab[k], edgecolor='#fcfcfb', linewidth=2, width=.6); bot += v
a.set_ylim(0, 118); a.set_yticks(range(0, 101, 20)); a.set_ylabel('Share of parameter sets (%)'); a.legend(frameon=False, fontsize=8, ncol=3, loc='upper center')
a.set_title('A  The longer contaminants stay, the likelier the trap', loc='left', fontweight='bold', fontsize=9.5)
a = ax[1]; tau = [0, 20, 40, 80]
ne = [B['cleanup']['none'] * 100] + [B[f'cleanup_with_lc{t}']['none'] * 100 for t in (20, 40, 80)]
md = [B['cleanup']['median'] * 100] + [B[f'cleanup_with_lc{t}']['median'] * 100 for t in (20, 40, 80)]
a.plot(tau, ne, '-o', lw=2, ms=6, color='#e34948', label='No cleanup is enough (% of cases)'); a.plot(tau, md, '-o', lw=2, ms=6, color=c[0], label='Median cleanup needed (% of gap)')
a.set_ylim(0, 70); a.set_xlabel('Years of land care halving the pressure, alongside cleanup'); a.set_ylabel('%'); a.legend(frameon=False, fontsize=8, loc='lower left')
a.set_title('B  Land care must last decades to matter', loc='left', fontweight='bold', fontsize=9.5)
for a in ax: a.grid(alpha=.25, lw=.5, axis='y')
fig.tight_layout(); fig.savefig('fig4.png', dpi=160, facecolor='#fcfcfb')
# ---- Fig 5
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3)); a = ax[0]; xs = np.array(C['xs']) * 100
for i, (s, l) in enumerate([('0.0', 'No shocks'), ('0.01', 'Weak shocks'), ('0.02', 'Moderate shocks'), ('0.04', 'Strong shocks')]):
    a.plot(xs, np.array(C['escape'][s]) * 100, '-o', ms=3.5, lw=2, color=c[i], label=l)
a.set_xlabel('Cleanup (% of the soil gap closed)'); a.set_ylabel('Probability of escaping the trap (%)'); a.legend(frameon=False, fontsize=8)
a.set_title('A  Shocks turn the threshold into odds', loc='left', fontweight='bold')
a = ax[1]; ps = np.array(C['ps'])
for i, (k, l) in enumerate([('0_0_0', 'Uniform, no shocks'), ('0.3_0.05_0.01', 'Heterogeneous, coupled, weak shocks'), ('0.3_0.05_0.03', 'Heterogeneous, coupled, stronger shocks')]):
    r = C['sweeps'][k]; a.plot(ps, r['up'], '-', lw=2, color=c[i], label=f"{l} (gap {r['gap']:.2f})"); a.plot(ps, r['down'], '--', lw=2, color=c[i])
a.set_xlabel('Degradation pressure, p (1/yr)'); a.set_ylabel('Mean soil condition, 144 neighbourhoods'); a.legend(frameon=False, fontsize=7.5, loc='lower left')
a.text(0.118, 0.93, 'solid: rising pressure\ndashed: falling pressure', ha='right', va='top', fontsize=7.5, color=ink)
a.set_title('B  City-scale hysteresis shrinks but persists', loc='left', fontweight='bold')
a = ax[2]; pp = sorted(float(p) for p in C['sync'])
a.plot(pp, [C['sync'][str(p)]['isolated'] * 100 for p in pp], '-o', lw=2, ms=5, color=c[2], label='Isolated neighbourhoods')
a.plot(pp, [C['sync'][str(p)]['connected'] * 100 for p in pp], '-o', lw=2, ms=5, color=c[1], label='Socially connected neighbourhoods')
a.set_ylim(0, 100); a.set_xlabel('Degradation pressure, p (1/yr)'); a.set_ylabel('Neighbourhoods in the trap (%)'); a.legend(frameon=False, fontsize=8, loc='upper left')
a.set_title('C  Connection synchronises neighbourhoods', loc='left', fontweight='bold'); grid(ax)
fig.tight_layout(); fig.savefig('fig5.png', dpi=160, facecolor='#fcfcfb')
# ---- Fig S1
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2)); a = ax[0]; bins = np.linspace(0.005, 0.05, 7); mid = (bins[:-1] + bins[1:]) / 2; bot = np.zeros(6)
for k in ['trap only', 'bistable', 'healthy only']:
    v = np.array(W['by_r'][k]) * 100; a.bar(mid, v, width=(bins[1] - bins[0]) * .9, bottom=bot, color=col[k], label=lab[k], edgecolor='#fcfcfb', linewidth=2); bot += v
a.set_ylim(0, 118); a.set_yticks(range(0, 101, 20)); a.legend(frameon=False, fontsize=8, ncol=3, loc='upper center')
a.set_xlabel('Natural soil recovery rate, r (1/yr)  ← longer residence'); a.set_ylabel('Share of 4,000 parameter sets (%)')
a.set_title('A  Persistence makes traps more likely', loc='left', fontweight='bold')
a = ax[1]; v = np.array(W['cleanup_values']); n_none = int(np.isnan(v).sum()); v = v[~np.isnan(v)]
a.hist(v * 100, bins=20, color=c[0], edgecolor='#fcfcfb', linewidth=1.5); a.bar(105, n_none, width=4.5, color='#e34948', edgecolor='#fcfcfb')
a.text(105, n_none + 3, 'no\nescape', ha='center', fontsize=8)
a.set_xlabel('Minimum cleanup to escape the trap (% of the soil gap)'); a.set_ylabel('Parameter sets with two stable states')
a.set_title('B  How much cleanup is enough?', loc='left', fontweight='bold')
for a in ax: a.grid(alpha=.25, lw=.5, axis='y')
fig.tight_layout(); fig.savefig('figS1.png', dpi=160, facecolor='#fcfcfb')
print('figures done', dev)
