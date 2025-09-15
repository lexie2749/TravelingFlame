import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Set up matplotlib to display English characters
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Read the Excel file
try:
    df_raw = pd.read_excel('实验数据2(1)_radians.xlsx', header=None)
    print(f"Successfully loaded Excel file with {len(df_raw)} rows")
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Function to check if a string contains Chinese characters
def contains_chinese(text):
    if pd.isna(text):
        return False
    text = str(text)
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# Function to translate Chinese experiment titles to English
def translate_title(chinese_title):
    """
    Translate Chinese experimental titles to English
    """
    translations = {
        '微升': 'μL',
        '凹槽': 'groove',
        '第': 'Trial ',
        '次实验': '',
        '实验': '',
    }
    
    english_title = chinese_title
    
    # Extract numbers and key information
    volume_pattern = r'(\d+)\s*微升'
    groove_pattern = r'(\d+mm)\s*凹槽'
    trial_pattern = r'第(\d+)次'
    
    volume_match = re.search(volume_pattern, chinese_title)
    groove_match = re.search(groove_pattern, chinese_title)
    trial_match = re.search(trial_pattern, chinese_title)
    
    parts = []
    if volume_match:
        parts.append(f"{volume_match.group(1)}μL")
    if groove_match:
        parts.append(f"{groove_match.group(1)} groove")
    if trial_match:
        parts.append(f"Trial {trial_match.group(1)}")
    
    if parts:
        english_title = " - ".join(parts)
    else:
        for ch, en in translations.items():
            english_title = english_title.replace(ch, en)
        english_title = re.sub(r'\s+', ' ', english_title).strip()
    
    return english_title

# List to store dataframes for each experiment
experiment_dataframes = []
current_experiment_info = None
current_data = []
data_started = False

print("Starting to parse experiments...")
print("NOTE: Converting all angular velocity data from degrees/s to rad/s")
print(f"Conversion factor: π/180 = {np.pi/180:.6f}\n")

# Iterate through the rows to identify experiment blocks
for index, row in df_raw.iterrows():
    row_values = row.tolist()
    
    if any(contains_chinese(cell) for cell in row_values):
        if current_experiment_info is not None and len(current_data) > 0:
            try:
                temp_df = pd.DataFrame(current_data, columns=['t', 'omega'])
                temp_df = temp_df.dropna()
                temp_df['t'] = pd.to_numeric(temp_df['t'], errors='coerce')
                temp_df['omega'] = pd.to_numeric(temp_df['omega'], errors='coerce')
                temp_df = temp_df.dropna()
                
                # Take absolute values
                temp_df['t'] = temp_df['t'].abs()
                temp_df['omega'] = temp_df['omega'].abs()
                
                # CONVERT FROM DEGREES/S TO RAD/S
                temp_df['omega'] = temp_df['omega'] * (np.pi / 180.0)
                
                if not temp_df.empty:
                    english_title = translate_title(current_experiment_info)
                    experiment_dataframes.append({
                        'info': english_title,
                        'original_info': current_experiment_info,
                        'df': temp_df
                    })
                    print(f"  Saved {len(temp_df)} data points for: {english_title}")
                    print(f"    Mean ω = {temp_df['omega'].mean():.4f} rad/s")
            except Exception as e:
                print(f"  Error saving data: {e}")
        
        chinese_cells = [str(cell) for cell in row_values if contains_chinese(cell)]
        current_experiment_info = " ".join(chinese_cells)
        current_data = []
        data_started = False
        print(f"\nFound new experiment: {current_experiment_info}")
        continue
    
    row_str = [str(x).lower() if pd.notna(x) else '' for x in row_values]
    if 't' in row_str or 'ω' in [str(x) if pd.notna(x) else '' for x in row_values] or 'omega' in row_str:
        data_started = True
        continue
    
    if data_started and current_experiment_info is not None:
        numeric_values = []
        for val in row_values:
            if pd.notna(val):
                try:
                    numeric_val = float(val)
                    numeric_values.append(numeric_val)
                except (ValueError, TypeError):
                    continue
        
        if len(numeric_values) >= 2:
            current_data.append([numeric_values[0], numeric_values[1]])

# Save the last experiment
if current_experiment_info is not None and len(current_data) > 0:
    try:
        temp_df = pd.DataFrame(current_data, columns=['t', 'omega'])
        temp_df = temp_df.dropna()
        temp_df['t'] = pd.to_numeric(temp_df['t'], errors='coerce')
        temp_df['omega'] = pd.to_numeric(temp_df['omega'], errors='coerce')
        temp_df = temp_df.dropna()
        
        # Take absolute values
        temp_df['t'] = temp_df['t'].abs()
        temp_df['omega'] = temp_df['omega'].abs()
        
        # CONVERT FROM DEGREES/S TO RAD/S
        temp_df['omega'] = temp_df['omega'] * (np.pi / 180.0)
        
        if not temp_df.empty:
            english_title = translate_title(current_experiment_info)
            experiment_dataframes.append({
                'info': english_title,
                'original_info': current_experiment_info,
                'df': temp_df
            })
            print(f"  Saved {len(temp_df)} data points for: {english_title}")
            print(f"    Mean ω = {temp_df['omega'].mean():.4f} rad/s")
    except Exception as e:
        print(f"  Error saving last experiment: {e}")

print(f"\n{'='*60}")
print(f"Total experiments detected: {len(experiment_dataframes)}")
print(f"All data has been converted from deg/s to rad/s")
print(f"{'='*60}\n")

# MAIN ANALYSIS SECTION WITH GROOVE RADIUS
print("\n" + "="*60)
print("IMPORTANT: Using only FIRST 2 SECONDS of time data for average velocity calculation")
print("Creating SCATTER PLOTS for each fuel volume (varying groove RADIUS)")
print("All plots are pure scatter plots with no connecting lines")
print("All angular velocities are in RADIANS per second")
print("="*60)

# Collect all data organized by fuel volume and groove RADIUS
data_by_fuel = {}
print("\nCollecting data from first 2 seconds of each experiment...")

for exp in experiment_dataframes:
    original_title = exp.get('original_info', exp['info'])
    df_exp = exp['df']
    
    if not df_exp.empty:
        # Extract groove diameter and convert to RADIUS
        groove_match = re.search(r'(\d+)mm', original_title)
        groove_radius = float(groove_match.group(1)) / 2 if groove_match else None  # Convert to radius
        
        # Extract fuel volume
        volume_match = re.search(r'(\d+)(?:微升|μL|ul)', original_title, re.IGNORECASE)
        fuel_volume = float(volume_match.group(1)) if volume_match else None
        
        if groove_radius is not None and fuel_volume is not None:
            if fuel_volume not in data_by_fuel:
                data_by_fuel[fuel_volume] = {}
            
            if groove_radius not in data_by_fuel[fuel_volume]:
                data_by_fuel[fuel_volume][groove_radius] = []
            
            # Use only FIRST 2 SECONDS of time data for omega values
            # Note: omega values are already in rad/s from the conversion above
            first_2s_data = df_exp[df_exp['t'] <= 2.0]
            
            # If no data in first 2 seconds, use all available data with warning
            if first_2s_data.empty:
                print(f"    WARNING: No data found in first 2 seconds for {fuel_volume}μL, {groove_radius}mm radius")
                print(f"            Using all {len(df_exp)} available data points")
                first_2s_omega = df_exp['omega'].tolist()
            else:
                first_2s_omega = first_2s_data['omega'].tolist()
                if len(first_2s_omega) < len(df_exp):
                    print(f"    Using {len(first_2s_omega)}/{len(df_exp)} data points (first 2 seconds) for {fuel_volume}μL, {groove_radius}mm radius")
            
            data_by_fuel[fuel_volume][groove_radius].extend(first_2s_omega)

# Create INDIVIDUAL SCATTER plots for each fuel volume (x-axis: groove RADIUS)
fuel_plot_count = 0
for fuel_vol in sorted(data_by_fuel.keys()):
    if data_by_fuel[fuel_vol]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        groove_radii = sorted(data_by_fuel[fuel_vol].keys())
        mean_values = []
        std_values = []
        
        for radius in groove_radii:
            omega_values = data_by_fuel[fuel_vol][radius]
            mean_values.append(np.mean(omega_values))
            std_values.append(np.std(omega_values) if len(omega_values) > 1 else 0)
        
        # PURE SCATTER PLOT with error bars
        ax.errorbar(groove_radii, mean_values, yerr=std_values,
                   fmt='o',  # Format: circle markers only, no lines
                   markersize=12,
                   capsize=6, capthick=2, alpha=0.9,
                   color='steelblue', ecolor='darkgray',
                   markeredgecolor='navy', markeredgewidth=1.5,
                   elinewidth=1.5)
        
        # Add value labels
        for r, m, s in zip(groove_radii, mean_values, std_values):
            label_text = f'{m:.3f}' if s == 0 else f'{m:.3f}±{s:.3f}'
            ax.annotate(label_text,
                       (r, m),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=9,
                       color='darkblue',
                       weight='bold')
        
        ax.set_xlabel('Groove Radius (mm)', fontsize=13)
        ax.set_ylabel('Average Angular Velocity ω (rad/s)', fontsize=13)
        ax.set_title(f'Angular Velocity vs Groove Radius (Scatter Plot)\nFuel Volume: {fuel_vol:.0f}μL', 
                    fontsize=15, weight='bold')
        
        # Enhanced grid for scatter plot
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.8)
        ax.set_axisbelow(True)  # Put grid behind the data points
        
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Set axis limits with appropriate padding
        if mean_values:
            x_padding = 0.2
            y_min = min([m - s for m, s in zip(mean_values, std_values)])
            y_max = max([m + s for m, s in zip(mean_values, std_values)])
            y_range = y_max - y_min
            y_padding = y_range * 0.15 if y_range > 0 else 0.01
            
            ax.set_xlim(min(groove_radii) - x_padding, max(groove_radii) + x_padding)
            ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        
        plt.tight_layout()
        filename = f'scatter_fuel_{fuel_vol:.0f}uL_vs_groove_radius.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()  # Display each plot
        plt.close()
        
        fuel_plot_count += 1
        print(f"  [{fuel_plot_count}] Saved: {filename}")

print(f"\nTotal fuel volume plots created: {fuel_plot_count}")

# Create individual plots for each groove RADIUS (varying fuel volume)
print("\n" + "="*60)
print("Creating SCATTER PLOTS for each groove radius (varying fuel volume)")
print("Using only FIRST 2 SECONDS of data for calculations")
print("All plots are pure scatter plots with no connecting lines")
print("="*60)

# Reorganize data by groove RADIUS
data_by_radius = {}

for exp in experiment_dataframes:
    original_title = exp.get('original_info', exp['info'])
    df_exp = exp['df']
    
    if not df_exp.empty:
        # Extract groove diameter and convert to RADIUS
        groove_match = re.search(r'(\d+)mm', original_title)
        groove_radius = float(groove_match.group(1)) / 2 if groove_match else None  # Convert to radius
        
        # Extract fuel volume
        volume_match = re.search(r'(\d+)(?:微升|μL|ul)', original_title, re.IGNORECASE)
        fuel_volume = float(volume_match.group(1)) if volume_match else None
        
        if groove_radius is not None and fuel_volume is not None:
            if groove_radius not in data_by_radius:
                data_by_radius[groove_radius] = {}
            
            if fuel_volume not in data_by_radius[groove_radius]:
                data_by_radius[groove_radius][fuel_volume] = []
            
            # Use only FIRST 2 SECONDS of time data for omega values
            # Note: omega values are already in rad/s from the conversion above
            first_2s_data = df_exp[df_exp['t'] <= 2.0]
            
            # If no data in first 2 seconds, use all available data
            if first_2s_data.empty:
                first_2s_omega = df_exp['omega'].tolist()
            else:
                first_2s_omega = first_2s_data['omega'].tolist()
            
            data_by_radius[groove_radius][fuel_volume].extend(first_2s_omega)

# Create INDIVIDUAL SCATTER plots for each groove radius (x-axis: fuel volume)
radius_plot_count = 0
for groove_rad in sorted(data_by_radius.keys()):
    if data_by_radius[groove_rad]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        fuel_volumes = sorted(data_by_radius[groove_rad].keys())
        mean_values = []
        std_values = []
        
        for fuel in fuel_volumes:
            omega_values = data_by_radius[groove_rad][fuel]
            mean_values.append(np.mean(omega_values))
            std_values.append(np.std(omega_values) if len(omega_values) > 1 else 0)
        
        # PURE SCATTER PLOT with error bars
        ax.errorbar(fuel_volumes, mean_values, yerr=std_values,
                   fmt='s',  # Format: square markers only, no lines
                   markersize=12,
                   capsize=6, capthick=2, alpha=0.9,
                   color='darkgreen', ecolor='darkgray',
                   markeredgecolor='darkgreen', markeredgewidth=1.5,
                   elinewidth=1.5)
        
        # Add value labels
        for f, m, s in zip(fuel_volumes, mean_values, std_values):
            label_text = f'{m:.3f}' if s == 0 else f'{m:.3f}±{s:.3f}'
            ax.annotate(label_text,
                       (f, m),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center',
                       fontsize=9,
                       color='darkgreen',
                       weight='bold')
        
        ax.set_xlabel('Fuel Volume (μL)', fontsize=13)
        ax.set_ylabel('Average Angular Velocity ω (rad/s)', fontsize=13)
        ax.set_title(f'Angular Velocity vs Fuel Volume (Scatter Plot)\nGroove Radius: {groove_rad:.1f}mm', 
                    fontsize=15, weight='bold')
        
        # Enhanced grid for scatter plot
        ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.8)
        ax.set_axisbelow(True)  # Put grid behind the data points
        
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        # Set axis limits with appropriate padding
        if mean_values:
            x_padding = 5  # padding for fuel volume axis
            y_min = min([m - s for m, s in zip(mean_values, std_values)])
            y_max = max([m + s for m, s in zip(mean_values, std_values)])
            y_range = y_max - y_min
            y_padding = y_range * 0.15 if y_range > 0 else 0.01
            
            ax.set_xlim(min(fuel_volumes) - x_padding, max(fuel_volumes) + x_padding)
            ax.set_ylim(max(0, y_min - y_padding), y_max + y_padding)
        
        plt.tight_layout()
        filename = f'scatter_groove_radius_{groove_rad:.1f}mm_vs_fuel.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.show()  # Display each plot
        plt.close()
        
        radius_plot_count += 1
        print(f"  [{radius_plot_count}] Saved: {filename}")

print(f"\nTotal groove radius plots created: {radius_plot_count}")

# Create summary grid plots with RADIUS
print("\n" + "="*60)
print("Creating summary SCATTER grid plots...")
print("All plots are pure scatter plots with error bars")
print("="*60)

# Grid plot for all fuel volumes (x-axis: groove RADIUS)
n_fuel_plots = len(data_by_fuel)
if n_fuel_plots > 0:
    n_cols = 4
    n_rows = (n_fuel_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    plot_idx = 0
    for fuel_vol in sorted(data_by_fuel.keys()):
        if data_by_fuel[fuel_vol]:
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 or n_cols > 1 else axes[0, 0]
            
            groove_radii = sorted(data_by_fuel[fuel_vol].keys())
            mean_values = []
            std_values = []
            
            for radius in groove_radii:
                omega_values = data_by_fuel[fuel_vol][radius]
                mean_values.append(np.mean(omega_values))
                std_values.append(np.std(omega_values) if len(omega_values) > 1 else 0)
            
            # PURE SCATTER PLOT with error bars
            ax.errorbar(groove_radii, mean_values, yerr=std_values,
                       fmt='o',  # Circle markers only, no lines
                       markersize=8,
                       capsize=3, capthick=1, alpha=0.8,
                       color='steelblue', ecolor='gray')
            
            ax.set_xlabel('Groove Radius (mm)', fontsize=9)
            ax.set_ylabel('ω (rad/s)', fontsize=9)
            ax.set_title(f'{fuel_vol:.0f}μL', fontsize=10, weight='bold')
            ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('All Fuel Volumes: Angular Velocity vs Groove Radius (Pure Scatter Plots)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('scatter_grid_all_fuel_volumes_vs_radius.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: scatter_grid_all_fuel_volumes_vs_radius.png")

# Grid plot for all groove radii
n_radius_plots = len(data_by_radius)
if n_radius_plots > 0:
    n_cols = 4
    n_rows = (n_radius_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    plot_idx = 0
    for groove_rad in sorted(data_by_radius.keys()):
        if data_by_radius[groove_rad]:
            row = plot_idx // n_cols
            col = plot_idx % n_cols
            ax = axes[row, col] if n_rows > 1 or n_cols > 1 else axes[0, 0]
            
            fuel_volumes = sorted(data_by_radius[groove_rad].keys())
            mean_values = []
            std_values = []
            
            for fuel in fuel_volumes:
                omega_values = data_by_radius[groove_rad][fuel]
                mean_values.append(np.mean(omega_values))
                std_values.append(np.std(omega_values) if len(omega_values) > 1 else 0)
            
            # PURE SCATTER PLOT with error bars
            ax.errorbar(fuel_volumes, mean_values, yerr=std_values,
                       fmt='s',  # Square markers only, no lines
                       markersize=8,
                       capsize=3, capthick=1, alpha=0.8,
                       color='darkgreen', ecolor='gray')
            
            ax.set_xlabel('Fuel (μL)', fontsize=9)
            ax.set_ylabel('ω (rad/s)', fontsize=9)
            ax.set_title(f'Radius: {groove_rad:.1f}mm', fontsize=10, weight='bold')
            ax.grid(True, linestyle=':', alpha=0.5, linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.suptitle('All Groove Radii: Angular Velocity vs Fuel Volume (Pure Scatter Plots)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('scatter_grid_all_groove_radii_vs_fuel.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  Saved: scatter_grid_all_groove_radii_vs_fuel.png")

print("\n" + "="*60)
print(f"Summary: Created {fuel_plot_count} fuel volume SCATTER plots and {radius_plot_count} groove radius SCATTER plots")
print("Plus 2 grid summary SCATTER plots")
print("All plots are PURE SCATTER PLOTS with no connecting lines")
print("All plots use groove RADIUS (not diameter) as requested")
print("All angular velocity data has been converted from deg/s to rad/s")
print(f"Conversion factor used: π/180 = {np.pi/180:.6f}")
print("All averages calculated using only the FIRST 2 SECONDS of data")
print("="*60)