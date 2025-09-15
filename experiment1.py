import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Data
fuel_volumes = [125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
groove_widths = [3, 4, 5, 6, 7, 8, 9, 10]

# Score matrix (2=complete success, 1=partial, 0=failure)
score_matrix = {
    3: [2,2,2,2,2,2,2,2,2,1,1,1,2,2,1,1],
    4: [2,1,1,2,2,2,2,2,2,2,2,1,2,1,2,1],
    5: [2,2,2,1,2,2,2,2,1,2,2,1,2,2,1,2],
    6: [2,2,1,1,1,2,1,1,0,1,2,1,1,1,0,0],
    7: [1,2,2,1,1,2,2,0,0,1,2,1,2,1,1,2],
    8: [0,1,1,1,1,0,1,0,0,0,1,1,1,1,0,0],
    9: [1,2,2,1,1,0,1,0,0,0,1,0,0,1,1,0],
    10: [0,0,0,1,1,1,0,0,1,0,1,1,1,1,1,0]
}

# Create figure with subplots
fig = plt.figure(figsize=(18, 14))

# ====================
# Chart 1: Heat Map
# ====================
ax1 = plt.subplot(2, 2, (1, 2))

# Convert score_matrix to numpy array for heatmap
heatmap_data = np.array([score_matrix[g] for g in groove_widths])

# Create custom colormap (red=0, yellow=1, green=2)
colors = ['#e74c3c', '#f39c12', '#27ae60']
n_bins = 3
cmap = plt.cm.colors.ListedColormap(colors)
bounds = [0, 0.67, 1.33, 2]
norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

# Create heatmap
im = ax1.imshow(heatmap_data, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

# Set ticks and labels
ax1.set_xticks(np.arange(len(fuel_volumes)))
ax1.set_yticks(np.arange(len(groove_widths)))
ax1.set_xticklabels(fuel_volumes)
ax1.set_yticklabels([f'{g}mm' for g in groove_widths])

# Add value annotations
for i in range(len(groove_widths)):
    for j in range(len(fuel_volumes)):
        value = heatmap_data[i, j]
        color = 'white' if value != 1 else 'black'
        text = ax1.text(j, i, int(value), ha="center", va="center", 
                       color=color, fontweight='bold', fontsize=9)

# Labels and title
ax1.set_xlabel('Fuel Volume (μL)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Groove Width (mm)', fontsize=12, fontweight='bold')
ax1.set_title('Flame Propagation Stability Heat Map', fontsize=14, fontweight='bold', pad=20)

# Create custom legend
legend_elements = [
    mpatches.Patch(color='#27ae60', label='Complete Rotation (2)'),
    mpatches.Patch(color='#f39c12', label='Partial Rotation (1)'),
    mpatches.Patch(color='#e74c3c', label='No Propagation (0)')
]
ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
          ncol=3, frameon=False, fontsize=10)

# Rotate x labels for better readability
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ====================
# Chart 2: Success Rate vs Groove Width
# ====================
ax2 = plt.subplot(2, 2, 3)

# Calculate success rates by groove width
groove_success_rates = []
groove_partial_rates = []
groove_failure_rates = []

for groove in groove_widths:
    scores = score_matrix[groove]
    total = len(scores)
    success_rate = (scores.count(2) / total) * 100
    partial_rate = (scores.count(1) / total) * 100
    failure_rate = (scores.count(0) / total) * 100
    
    groove_success_rates.append(success_rate)
    groove_partial_rates.append(partial_rate)
    groove_failure_rates.append(failure_rate)

# Plot lines
x_pos = np.arange(len(groove_widths))
ax2.plot(x_pos, groove_success_rates, 'o-', color='#27ae60', linewidth=2.5, 
         markersize=8, label='Complete Success')
ax2.plot(x_pos, groove_partial_rates, 's-', color='#f39c12', linewidth=2.5, 
         markersize=8, label='Partial Success')
ax2.plot(x_pos, groove_failure_rates, '^-', color='#e74c3c', linewidth=2.5, 
         markersize=8, label='Failure')

# Formatting
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f'{g}mm' for g in groove_widths])
ax2.set_xlabel('Groove Width', fontsize=12, fontweight='bold')
ax2.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Success Rate vs Groove Width', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 100)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', frameon=True, shadow=True)

# Add percentage labels on data points
for i, (s, p, f) in enumerate(zip(groove_success_rates, groove_partial_rates, groove_failure_rates)):
    if i % 2 == 0:  # Show every other label to avoid crowding
        ax2.annotate(f'{s:.0f}%', (i, s), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8, color='#27ae60')

# ====================
# Chart 3: Success Rate vs Fuel Volume
# ====================
ax3 = plt.subplot(2, 2, 4)

# Calculate success rates by fuel volume
fuel_success_rates = []
fuel_partial_rates = []
fuel_failure_rates = []

for idx in range(len(fuel_volumes)):
    scores = [score_matrix[g][idx] for g in groove_widths]
    total = len(scores)
    success_rate = (scores.count(2) / total) * 100
    partial_rate = (scores.count(1) / total) * 100
    failure_rate = (scores.count(0) / total) * 100
    
    fuel_success_rates.append(success_rate)
    fuel_partial_rates.append(partial_rate)
    fuel_failure_rates.append(failure_rate)

# Create stacked bar chart
x_pos = np.arange(len(fuel_volumes))
width = 0.8

# Plot stacked bars
p1 = ax3.bar(x_pos, fuel_success_rates, width, color='#27ae60', label='Complete Success')
p2 = ax3.bar(x_pos, fuel_partial_rates, width, bottom=fuel_success_rates, 
             color='#f39c12', label='Partial Success')
p3 = ax3.bar(x_pos, fuel_failure_rates, width, 
             bottom=np.array(fuel_success_rates) + np.array(fuel_partial_rates),
             color='#e74c3c', label='Failure')

# Formatting
ax3.set_xticks(x_pos[::2])  # Show every other label to avoid crowding
ax3.set_xticklabels(fuel_volumes[::2])
ax3.set_xlabel('Fuel Volume (μL)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax3.set_title('Success Rate vs Fuel Volume', fontsize=14, fontweight='bold')
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(loc='best', frameon=True, shadow=True)

# Rotate x labels
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

# ====================
# Overall title and layout
# ====================
fig.suptitle('Flame Propagation Analysis in Ring-Shaped Trough\nExperimental Study of Combustion Stability Parameters', 
             fontsize=16, fontweight='bold', y=0.98)

# Add experiment info text
info_text = f'Total Experiments: {len(fuel_volumes) * len(groove_widths)} | ' \
           f'Fuel Range: {min(fuel_volumes)}-{max(fuel_volumes)} μL | ' \
           f'Groove Range: {min(groove_widths)}-{max(groove_widths)} mm'
fig.text(0.5, 0.02, info_text, ha='center', fontsize=10, style='italic', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Show the plots
plt.show()

# ====================
# Additional Analysis
# ====================
print("=" * 60)
print("FLAME PROPAGATION ANALYSIS SUMMARY")
print("=" * 60)

# Overall statistics
all_scores = []
for g in groove_widths:
    all_scores.extend(score_matrix[g])

total_experiments = len(all_scores)
complete_success = all_scores.count(2)
partial_success = all_scores.count(1)
failures = all_scores.count(0)

print(f"\nTotal Experiments: {total_experiments}")
print(f"Complete Success: {complete_success} ({complete_success/total_experiments*100:.1f}%)")
print(f"Partial Success: {partial_success} ({partial_success/total_experiments*100:.1f}%)")
print(f"Failures: {failures} ({failures/total_experiments*100:.1f}%)")

# Best configurations
print("\n" + "=" * 40)
print("OPTIMAL CONFIGURATIONS")
print("=" * 40)

best_configs = []
for g in groove_widths:
    for i, fuel in enumerate(fuel_volumes):
        if score_matrix[g][i] == 2:
            best_configs.append((g, fuel, 2))

# Find the groove width with highest success rate
best_groove = max(groove_widths, key=lambda g: score_matrix[g].count(2))
best_groove_rate = (score_matrix[best_groove].count(2) / len(score_matrix[best_groove])) * 100

print(f"\nBest Groove Width: {best_groove}mm (Success Rate: {best_groove_rate:.1f}%)")

# Find the fuel volume with highest success rate
best_fuel_idx = max(range(len(fuel_volumes)), 
                   key=lambda i: sum(score_matrix[g][i] == 2 for g in groove_widths))
best_fuel = fuel_volumes[best_fuel_idx]
best_fuel_rate = (sum(score_matrix[g][best_fuel_idx] == 2 for g in groove_widths) / len(groove_widths)) * 100

print(f"Best Fuel Volume: {best_fuel}μL (Success Rate: {best_fuel_rate:.1f}%)")

# Critical transitions
print("\n" + "=" * 40)
print("CRITICAL TRANSITIONS")
print("=" * 40)

print("\nGroove Width Stability Threshold:")
for i, g in enumerate(groove_widths):
    success_rate = (score_matrix[g].count(2) / len(score_matrix[g])) * 100
    if success_rate < 50 and i > 0:
        prev_rate = (score_matrix[groove_widths[i-1]].count(2) / len(score_matrix[groove_widths[i-1]])) * 100
        if prev_rate >= 50:
            print(f"  Transition at {groove_widths[i-1]}-{g}mm (from {prev_rate:.1f}% to {success_rate:.1f}%)")
            break

print("\nFuel Volume Critical Points:")
for i in range(1, len(fuel_volumes)):
    curr_scores = [score_matrix[g][i] for g in groove_widths]
    prev_scores = [score_matrix[g][i-1] for g in groove_widths]
    curr_success = curr_scores.count(2) / len(curr_scores)
    prev_success = prev_scores.count(2) / len(prev_scores)
    
    if abs(curr_success - prev_success) > 0.3:
        print(f"  Significant change at {fuel_volumes[i]}μL " 
              f"({prev_success*100:.1f}% → {curr_success*100:.1f}%)")