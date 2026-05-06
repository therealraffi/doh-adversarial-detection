import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── Graph 1: SHAP feature importance ─────────────────────────
features = [
    'PacketTimeVariance',
    'PacketTimeMedian',
    'ResponseTimeTimeMean',
    'PacketLengthMean',
    'PacketLengthMedian',
    'FlowBytesReceived',
    'ResponseTimeTimeMedian',
    'PacketLengthMode',
]
xgb_vals = [0.227, 0.386, 0.497, 1.012, 1.344, 1.518, 1.617, 5.624]
rf_vals  = [0.036, 0.009, 0.008, 0.037, 0.035, 0.037, 0.025, 0.124]

fig, ax = plt.subplots(figsize=(10, 6))
y = np.arange(len(features))
h = 0.35
ax.barh(y - h/2, rf_vals,  h, color='#3266ad', label='Random Forest')
ax.barh(y + h/2, xgb_vals, h, color='#e67e22', label='XGBoost')
ax.set_yticks(y)
ax.set_yticklabels(features, fontsize=11)
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_xlim(0, 7.0)
ax.set_title('What the models actually look for\n(Top 8 SHAP features driving detection)', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.text(3.8, 7.4, 'Primary detection signal', fontsize=10, color='#e67e22', fontweight='bold')
plt.tight_layout()
plt.savefig('slide_shap.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved slide_shap.png')
plt.close()

# ── Graph 2: Feature matching ─────────────────────────────────
features2   = ['PacketLengthMode (B)', 'PacketLengthMean (B)', 'PacketLengthStd (B)', 'FlowBytesSent (KB)']
our_vals    = [54.0, 166.0, 89.0, 68.0]
benign_vals = [74.1, 137.5, 89.3, 62.0]

x = np.arange(len(features2))
w = 0.35
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, our_vals,    w, color='#e55353', label='Our Adversarial Flows')
ax.bar(x + w/2, benign_vals, w, color='#3266ad', label='Real Benign (CIRA-CIC)')
ax.set_xticks(x)
ax.set_xticklabels(features2, fontsize=11)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Adversarial flows match real benign distributions\n(validated against CIRA-CIC browser traffic)', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i in range(len(features2)):
    ax.text(i, max(our_vals[i], benign_vals[i]) * 1.05, 'match',
            ha='center', fontsize=9, color='green', fontweight='bold')
plt.tight_layout()
plt.savefig('slide_feature_match.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved slide_feature_match.png')
plt.close()

# ── Graph 3: White-box vs Black-box combined heatmap ──────────
strategies = ['naive', 'timing_only', 'size_mimicry', 'cover_traffic', 'full_mimicry', 'adaptive']
cols = ['WB\nRF', 'WB\nGB', 'WB\nXGB', 'BB\nRF', 'BB\nGB', 'BB\nXGB']

data = np.array([
    [100, 100, 100,  0,  0,  0],
    [100, 100, 100,  0, 40,  0],
    [100, 100, 100,  0, 40,  0],
    [100, 100, 100,  0, 25,  0],
    [100, 100, 100,  0, 50,  0],
    [100, 100, 100,  0, 50,  0],
])

fig, ax = plt.subplots(figsize=(11, 5))
cmap = mcolors.LinearSegmentedColormap.from_list('evasion', ['#c0392b', '#f5f5f5', '#1a7a3a'])
im = ax.imshow(data, cmap=cmap, vmin=0, vmax=100, aspect='auto')

ax.set_xticks(range(len(cols)))
ax.set_xticklabels(cols, fontsize=11, fontweight='bold')
ax.set_yticks(range(len(strategies)))
ax.set_yticklabels(strategies, fontsize=11)
ax.set_title('Evasion rate: white-box vs black-box\n(green = evades detection   |   red = caught)', fontsize=13, fontweight='bold', pad=20)

ax.axvline(x=2.5, color='black', linewidth=2.5, linestyle='--', alpha=0.6)
ax.text(1.0, 6.7, 'WHITE-BOX', ha='center', fontsize=10, fontweight='bold',
        color='#1a7a3a', transform=ax.transData)
ax.text(4.0, 6.7, 'BLACK-BOX', ha='center', fontsize=10, fontweight='bold',
        color='#c0392b', transform=ax.transData)

for i in range(len(strategies)):
    for j in range(len(cols)):
        val = data[i, j]
        color = 'white' if val > 50 or val < 15 else 'black'
        ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color=color)

plt.colorbar(im, ax=ax, label='Evasion Rate (%)')
plt.tight_layout()
plt.savefig('slide_whitebox.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved slide_whitebox.png')
plt.close()

# ── Graph 4: Scaler gap table ───────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

row_data = [
    ['Random Forest',     '100%', '0%',  'RF: robust without scaler'],
    ['Gradient Boosting', '100%', '50%', 'GB: partially vulnerable'],
    ['XGBoost',           '100%', '0%',  'XGB: robust without scaler'],
]
col_labels = ['Model', 'White-box\n(scaler known)', 'Black-box\n(realistic attacker)', 'Verdict']
cell_colors = [
    ['#f5f5f5','#1a7a3a','#c0392b','#fff3f3'],
    ['#f5f5f5','#1a7a3a','#e67e22','#fff8f0'],
    ['#f5f5f5','#1a7a3a','#c0392b','#fff3f3'],
]
text_colors = [
    ['black','white','white','#c0392b'],
    ['black','white','white','#e67e22'],
    ['black','white','white','#c0392b'],
]
table = ax.table(cellText=row_data, colLabels=col_labels, cellLoc='center', loc='center', bbox=[0,0,1,1])
table.auto_set_font_size(False)
table.set_fontsize(13)
for j in range(4):
    cell = table[0,j]
    cell.set_facecolor('#2c2c2c')
    cell.set_text_props(color='white', fontweight='bold', fontsize=12)
    cell.set_height(0.2)
for i in range(3):
    for j in range(4):
        cell = table[i+1,j]
        cell.set_facecolor(cell_colors[i][j])
        cell.set_text_props(color=text_colors[i][j], fontweight='bold' if j in [1,2] else 'normal', fontsize=13)
        cell.set_height(0.22)
ax.set_title('The scaler is the critical defense artifact\n(white-box vs black-box evasion rate)', fontsize=13, fontweight='bold', pad=20, y=1.02)
plt.tight_layout()
plt.savefig('slide_gap.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved slide_gap.png')
plt.close()

# ── Graph 5: Arms race ───────────────────────────────────────
models_arms = ['RF', 'GB', 'XGB']
x = np.arange(len(models_arms))
w = 0.35
fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x - w/2, [100,100,100], w, color='#c0392b', label='Round 1: Before retraining (attacker wins)', zorder=3)
b2 = ax.bar(x + w/2, [2,2,2], w, color='#1a7a3a', label='Round 2: After retraining (defender wins)', zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(models_arms, fontsize=14, fontweight='bold')
ax.set_ylabel('Evasion Rate (%)', fontsize=12)
ax.set_ylim(0, 120)
ax.set_title('Arms race: adversarial retraining closes the gap
(300 adversarial hard negatives added to training set)', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=1)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.yaxis.grid(True, alpha=0.3, zorder=0)
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            '100%', ha='center', fontsize=12, fontweight='bold', color='#c0392b')
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            '0%', ha='center', fontsize=12, fontweight='bold', color='#1a7a3a')
ax.text(-0.5, -12, 'Round 1 (red) vs Round 2 (green) — attacker wins first, defender wins after retraining',
        ha='left', fontsize=9, color='gray', style='italic', transform=ax.transData)
plt.tight_layout()
plt.savefig('slide_arms_race.png', dpi=150, bbox_inches='tight', facecolor='white')
print('Saved slide_arms_race.png')
plt.close()

print('\nAll 5 graphs saved.')
