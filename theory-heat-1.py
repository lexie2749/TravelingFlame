import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Experimental Data from the provided image
# The first list is the Groove Diameter (x-axis) in mm
groove_diameters = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# The first list is the Fuel Volume (y-axis) in uL
fuel_volumes = [
    125.0, 130.0, 135.0, 140.0, 145.0, 150.0, 155.0, 160.0, 165.0,
    170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0
]

# Angular velocity data extracted from the image's heatmap
# Missing values are represented by NaN
angular_velocities = np.array([
    [569.4, 558.9, 574.8, 563.2, 613.4, 567.2, 770.2, 579.2],
    [571.6, 669.2, 509.4, 575.7, 572.4, 396.5, 531.9, np.nan],
    [713.1, 359.0, 205.8, 569.2, 585.7, 483.4, 547.1, np.nan],
    [590.0, 571.9, 132.1, 437.5, 613.9, 545.3, 471.4, np.nan],
    [np.nan, 597.8, 172.1, 589.6, 372.4, 486.7, 462.9, np.nan],
    [577.2, 564.2, 504.4, 588.3, 500.7, 562.8, 610.5, 759.6],
    [597.5, 548.4, 215.0, 596.8, 646.0, 485.6, 421.8, np.nan],
    [434.6, 585.7, 534.2, 609.0, np.nan, np.nan, np.nan, np.nan],
    [530.9, 603.1, 415.4, 692.0, np.nan, np.nan, np.nan, np.nan],
    [449.0, 583.1, 679.5, 282.6, 279.8, np.nan, np.nan, np.nan],
    [578.0, 622.1, 612.0, 702.3, 645.9, 537.1, 782.7, 830.3],
    [485.8, 584.6, 515.1, 207.4, 483.4, 367.5, np.nan, np.nan],
    [552.4, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [569.8, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [432.2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    [574.1, 482.6, 300.4, 579.5, 451.2, np.nan, np.nan, np.nan]
])

# Create the figure and axes
fig, ax = plt.subplots(figsize=(10, 8))

# Define a custom colormap from the image (similar to 'viridis')
# You can get a good approximation of the colors with this
colors = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
cmap = LinearSegmentedColormap.from_list("custom_viridis", colors, N=256)

# Create the heatmap
c = ax.imshow(
    angular_velocities,
    cmap=cmap,
    interpolation='nearest',
    extent=[
        min(groove_diameters) - 0.5,
        max(groove_diameters) + 0.5,
        max(fuel_volumes) + 2.5,
        min(fuel_volumes) - 2.5,
    ],
    aspect='auto',
)

# Label the x and y axes with the provided values
ax.set_xticks(groove_diameters)
ax.set_xticklabels([f'{d:.1f}mm' for d in groove_diameters])
ax.set_yticks(fuel_volumes)
ax.set_yticklabels([f'{v:.1f}µL' for v in fuel_volumes])

# Label the cell values
for i in range(len(fuel_volumes)):
    for j in range(len(groove_diameters)):
        val = angular_velocities[i, j]
        if not np.isnan(val):
            ax.text(
                groove_diameters[j],
                fuel_volumes[i],
                f'{val:.1f}',
                ha='center',
                va='center',
                color='white' if val < 400 else 'black',
                fontsize=8,
            )

# Add title and labels
ax.set_title(
    'Angular Velocity Heatmap\n(Darker = Faster)',
    fontsize=16,
    fontweight='bold',
    pad=20,
)
ax.set_xlabel('Groove Diameter', fontsize=12)
ax.set_ylabel('Fuel Volume', fontsize=12)

# Create color bar
cbar = plt.colorbar(c, ax=ax, shrink=0.7)
cbar.set_label('Average Angular Velocity (ω)', rotation=270, labelpad=15)
cbar.set_ticks(np.arange(100, 901, 100))

# Set the background color for missing data points
ax.set_facecolor('lightgray')

plt.tight_layout()
plt.show()

