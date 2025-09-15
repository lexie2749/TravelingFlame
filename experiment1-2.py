import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-whitegrid')

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

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 7))

# Plot scatter points
ax.scatter(groove_widths, groove_success_rates, s=100, color='#27ae60', 
           alpha=0.8, edgecolors='darkgreen', linewidth=2, label='Complete Success', zorder=3)
ax.scatter(groove_widths, groove_partial_rates, s=100, color='#f39c12', 
           alpha=0.8, edgecolors='darkorange', linewidth=2, label='Partial Success', marker='s', zorder=3)
ax.scatter(groove_widths, groove_failure_rates, s=100, color='#e74c3c', 
           alpha=0.8, edgecolors='darkred', linewidth=2, label='Failure', marker='^', zorder=3)

# Add value labels on points
for i, (groove, success, partial, failure) in enumerate(zip(groove_widths, groove_success_rates, groove_partial_rates, groove_failure_rates)):
    ax.annotate(f'{success:.1f}%', (groove, success), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='#27ae60')
    ax.annotate(f'{partial:.1f}%', (groove, partial), textcoords="offset points", 
                xytext=(0,10), ha='center', fontsize=9, fontweight='bold', color='#f39c12')
    ax.annotate(f'{failure:.1f}%', (groove, failure), textcoords="offset points", 
                xytext=(0,-15), ha='center', fontsize=9, fontweight='bold', color='#e74c3c')

# Formatting
ax.set_xlabel('Groove Width (mm)', fontsize=14, fontweight='bold')
ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
ax.set_title('Success Rate vs Groove Width\nFlame Propagation in Ring-Shaped Trough', 
            fontsize=16, fontweight='bold', pad=20)

# Set x-axis
ax.set_xticks(groove_widths)
ax.set_xticklabels([f'{g}' for g in groove_widths])

# Set y-axis
ax.set_ylim(-5, 105)
ax.set_yticks(range(0, 101, 20))

# Add grid for better readability
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11, 
         markerscale=1.2, borderpad=1, columnspacing=1.5)

# Add background color zones for interpretation
ax.axhspan(0, 30, alpha=0.05, color='red', zorder=0)
ax.axhspan(30, 70, alpha=0.05, color='orange', zorder=0)
ax.axhspan(70, 100, alpha=0.05, color='green', zorder=0)

# Add zone labels
ax.text(10.3, 85, 'High Stability', fontsize=10, style='italic', color='green', alpha=0.7)
ax.text(10.3, 50, 'Transition', fontsize=10, style='italic', color='darkorange', alpha=0.7)
ax.text(10.3, 15, 'Low Stability', fontsize=10, style='italic', color='red', alpha=0.7)

# Add summary statistics box
stats_text = f'Optimal Range: 3-5 mm\nAvg Success Rate (3-5mm): {np.mean(groove_success_rates[:3]):.1f}%\nAvg Success Rate (6-10mm): {np.mean(groove_success_rates[3:]):.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray', linewidth=1.5)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Print analysis
print("=" * 50)
print("SUCCESS RATE ANALYSIS BY GROOVE WIDTH")
print("=" * 50)
print("\nGroove Width | Complete | Partial | Failure")
print("-" * 50)
for i, groove in enumerate(groove_widths):
    print(f"   {groove:2d} mm    |  {groove_success_rates[i]:5.1f}%  | {groove_partial_rates[i]:5.1f}%  | {groove_failure_rates[i]:5.1f}%")

print("\n" + "=" * 50)
print("KEY FINDINGS:")
print("=" * 50)
print(f"• Best groove width: {groove_widths[np.argmax(groove_success_rates)]} mm ({max(groove_success_rates):.1f}% success)")
print(f"• Worst groove width: {groove_widths[np.argmin(groove_success_rates)]} mm ({min(groove_success_rates):.1f}% success)")
print(f"• Critical transition: Between {groove_widths[2]} mm and {groove_widths[3]} mm")
print(f"• Stability threshold: ~6 mm (success rate drops below 50%)")