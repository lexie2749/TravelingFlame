import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Set up matplotlib to display English characters
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Read the Excel file
try:
    df_raw = pd.read_excel('实验数据2(1).xlsx', header=None)
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
    # Create translation dictionary
    translations = {
        '微升': 'μL',
        '凹槽': 'groove',
        '第': 'Trial ',
        '次实验': '',
        '实验': '',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '125': '125',
        '150': '150',
        '175': '175',
        '200': '200',
        'mm': 'mm',
        '3mm': '3mm',
        '4mm': '4mm',
        '5mm': '5mm',
    }
    
    # Process the title
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
        # Fallback: simple replacement
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

# Iterate through the rows to identify experiment blocks
for index, row in df_raw.iterrows():
    # Convert all elements in the row to string for checking
    row_values = row.tolist()
    
    # Check if any cell in this row contains Chinese characters (this is likely the title row)
    if any(contains_chinese(cell) for cell in row_values):
        # Found a new experiment title
        if current_experiment_info is not None and len(current_data) > 0:
            # Save the previous experiment's data
            try:
                temp_df = pd.DataFrame(current_data, columns=['t', 'omega'])
                # Clean the dataframe
                temp_df = temp_df.dropna()
                temp_df['t'] = pd.to_numeric(temp_df['t'], errors='coerce')
                temp_df['omega'] = pd.to_numeric(temp_df['omega'], errors='coerce')
                temp_df = temp_df.dropna()
                
                # Take absolute values to make all data positive
                temp_df['t'] = temp_df['t'].abs()
                temp_df['omega'] = temp_df['omega'].abs()
                
                if not temp_df.empty:
                    # Translate the title to English
                    english_title = translate_title(current_experiment_info)
                    experiment_dataframes.append({
                        'info': english_title, 
                        'original_info': current_experiment_info,
                        'df': temp_df
                    })
                    print(f"  Saved {len(temp_df)} data points for: {english_title}")
            except Exception as e:
                print(f"  Error saving data for {current_experiment_info}: {e}")
        
        # Extract the Chinese text as experiment info
        chinese_cells = [str(cell) for cell in row_values if contains_chinese(cell)]
        current_experiment_info = " ".join(chinese_cells)
        current_data = []
        data_started = False
        print(f"\nFound new experiment: {current_experiment_info}")
        continue
    
    # Check if this row contains 't' and 'ω' or 'omega' headers
    row_str = [str(x).lower() if pd.notna(x) else '' for x in row_values]
    if 't' in row_str or 'ω' in [str(x) if pd.notna(x) else '' for x in row_values] or 'omega' in row_str:
        data_started = True
        print(f"  Found data headers at row {index}")
        continue
    
    # If we've started collecting data and this row has numeric values
    if data_started and current_experiment_info is not None:
        # Try to extract t and omega values
        # Assuming t is in the first numeric column and omega in the second
        numeric_values = []
        for val in row_values:
            if pd.notna(val):
                try:
                    numeric_val = float(val)
                    numeric_values.append(numeric_val)
                except (ValueError, TypeError):
                    continue
        
        # If we have at least 2 numeric values, use them as t and omega
        if len(numeric_values) >= 2:
            current_data.append([numeric_values[0], numeric_values[1]])

# Don't forget to save the last experiment
if current_experiment_info is not None and len(current_data) > 0:
    try:
        temp_df = pd.DataFrame(current_data, columns=['t', 'omega'])
        # Clean the dataframe
        temp_df = temp_df.dropna()
        temp_df['t'] = pd.to_numeric(temp_df['t'], errors='coerce')
        temp_df['omega'] = pd.to_numeric(temp_df['omega'], errors='coerce')
        temp_df = temp_df.dropna()
        
        # Take absolute values to make all data positive
        temp_df['t'] = temp_df['t'].abs()
        temp_df['omega'] = temp_df['omega'].abs()
        
        if not temp_df.empty:
            # Translate the title to English
            english_title = translate_title(current_experiment_info)
            experiment_dataframes.append({
                'info': english_title,
                'original_info': current_experiment_info,
                'df': temp_df
            })
            print(f"  Saved {len(temp_df)} data points for: {english_title}")
    except Exception as e:
        print(f"  Error saving last experiment data: {e}")

print(f"\n{'='*60}")
print(f"Total experiments detected: {len(experiment_dataframes)}")
print(f"{'='*60}\n")

# Create a subplot figure for all experiments
if len(experiment_dataframes) > 0:
    # Calculate grid dimensions
    n_experiments = len(experiment_dataframes)
    n_cols = min(3, n_experiments)  # Maximum 3 columns
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    # Create subplots
    if n_experiments == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]  # Make it a list for consistency
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each experiment
    for i, exp in enumerate(experiment_dataframes):
        title = exp['info']
        df_exp = exp['df']
        
        if not df_exp.empty:
            # Plot on subplot
            ax = axes[i]
            ax.scatter(df_exp['t'], df_exp['omega'], s=10, alpha=0.6)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel('t (Time)', fontsize=9)
            ax.set_ylabel('ω (Angular Velocity)', fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            print(f"Plotted experiment {i+1}: {title}")
            print(f"  Data points: {len(df_exp)}")
            print(f"  t range: [{df_exp['t'].min():.2f}, {df_exp['t'].max():.2f}]")
            print(f"  ω range: [{df_exp['omega'].min():.2f}, {df_exp['omega'].max():.2f}]")
    
    # Hide any unused subplots
    if n_experiments > 1:
        for j in range(n_experiments, len(axes)):
            axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('all_experiments_combined.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nCombined plot saved as 'all_experiments_combined.png'")

# Also save individual plots
print("\nGenerating individual plots...")
for i, exp in enumerate(experiment_dataframes):
    title = exp['info']
    df_exp = exp['df']
    
    if not df_exp.empty:
        plt.figure(figsize=(10, 6))
        plt.scatter(df_exp['t'], df_exp['omega'], s=15, alpha=0.7)
        
        # Set plot title and labels in English
        plt.title(title, fontsize=14)
        plt.xlabel('t (Time)', fontsize=12)
        plt.ylabel('ω (Angular Velocity)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        safe_title = safe_title[:50]  # Limit filename length
        filename = f"experiment_{i+1}_{safe_title}.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plot saved: {filename}")

# --- Analysis for specific volumes ---
print("\n" + "="*60)
print("Analysis for 125μL, 150μL, and 175μL experiments")
print("="*60)

# Filter experiments for specified volumes
relevant_experiments = []
for exp in experiment_dataframes:
    # Check the English title for volumes
    if any(vol in exp['info'] for vol in ['125μL', '150μL', '175μL']):
        relevant_experiments.append(exp)
        print(f"Found relevant experiment: {exp['info']}")

if relevant_experiments:
    # Create a new figure for the normalized data with wider aspect ratio
    fig, ax = plt.subplots(figsize=(18, 7))
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(relevant_experiments)))
    
    for idx, exp in enumerate(relevant_experiments):
        title = exp['info']
        df_exp = exp['df'].copy()  # Make a copy to avoid modifying original
        
        if not df_exp.empty:
            # Normalize time for each experiment
            t_min = df_exp['t'].min()
            t_max = df_exp['t'].max()
            if t_max > t_min:
                df_exp['t_normalized'] = (df_exp['t'] - t_min) / (t_max - t_min)
            else:
                df_exp['t_normalized'] = 0
            
            # Plot the normalized data
            ax.scatter(df_exp['t_normalized'], df_exp['omega'], 
                      s=20, alpha=0.7, color=colors[idx], label=title)
            print(f"  Normalized and plotted: {title}")
    
    # Set plot title and labels in English
    ax.set_title('Normalized Time vs. Angular Velocity Comparison', fontsize=14)
    ax.set_xlabel('Normalized Time (0-1)', fontsize=12)
    ax.set_ylabel('ω (Angular Velocity)', fontsize=12)
    
    # Place legend outside the plot area on the left
    ax.legend(title="Experiment", bbox_to_anchor=(-0.15, 1), loc='upper right', fontsize=9)
    
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Adjust layout to accommodate legend
    plt.subplots_adjust(left=0.2)
    
    filename = "normalized_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nNormalized comparison plot saved as '{filename}'")
else:
    print("No experiments found with 125μL, 150μL, or 175μL volumes")

# Create a summary plot showing all experiments with positive values
print("\n" + "="*60)
print("Creating summary plot with all positive values")
print("="*60)

if len(experiment_dataframes) > 0:
    # Create figure with wider aspect ratio
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Use different colors for each experiment
    colors = plt.cm.tab20(np.linspace(0, 1, len(experiment_dataframes)))
    
    for idx, exp in enumerate(experiment_dataframes):
        df_exp = exp['df']
        if not df_exp.empty:
            ax.scatter(df_exp['t'], df_exp['omega'], 
                      s=15, alpha=0.6, color=colors[idx], 
                      label=exp['info'])
    
    ax.set_title('All Experiments - Positive Values Only', fontsize=16)
    ax.set_xlabel('t (Time)', fontsize=14)
    ax.set_ylabel('|ω| (Absolute Angular Velocity)', fontsize=14)
    
    # Place legend outside the plot area on the left
    ax.legend(bbox_to_anchor=(-0.2, 1), loc='upper right', fontsize=8, 
             ncol=1, frameon=True, fancybox=True, shadow=True)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Adjust layout to accommodate legend on the left
    plt.subplots_adjust(left=0.25)
    
    plt.savefig('all_experiments_positive_values.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Summary plot with positive values saved as 'all_experiments_positive_values.png'")

# Create a new analysis plot: Groove Diameter vs Fuel Volume colored by Average Angular Velocity
print("\n" + "="*60)
print("Creating Groove Diameter vs Fuel Volume plot colored by Average Angular Velocity")
print("="*60)

# Extract parameters from experiment titles and calculate average omega
analysis_data = []
for exp in experiment_dataframes:
    original_title = exp.get('original_info', exp['info'])
    df_exp = exp['df']
    
    if not df_exp.empty:
        # Extract groove diameter (mm)
        groove_match = re.search(r'(\d+)mm', original_title)
        groove_diameter = float(groove_match.group(1)) if groove_match else None
        
        # Extract fuel volume (μL)
        volume_match = re.search(r'(\d+)(?:微升|μL|ul)', original_title, re.IGNORECASE)
        fuel_volume = float(volume_match.group(1)) if volume_match else None
        
        # Calculate average angular velocity
        avg_omega = df_exp['omega'].mean()
        
        if groove_diameter is not None and fuel_volume is not None:
            analysis_data.append({
                'groove_diameter': groove_diameter,
                'fuel_volume': fuel_volume,
                'avg_omega': avg_omega,
                'title': exp['info']
            })
            print(f"  {exp['info']}: Groove={groove_diameter}mm, Fuel={fuel_volume}μL, Avg ω={avg_omega:.2f}")

if analysis_data:
    # Convert to DataFrame for easier manipulation
    analysis_df = pd.DataFrame(analysis_data)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with color based on average omega
    scatter = ax.scatter(analysis_df['groove_diameter'], 
                        analysis_df['fuel_volume'],
                        c=analysis_df['avg_omega'],
                        s=200,  # Large points for visibility
                        cmap='viridis',  # Color map: light to dark for slow to fast
                        edgecolors='black',
                        linewidths=1.5,
                        alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Average Angular Velocity (ω)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add labels for each point
    for idx, row in analysis_df.iterrows():
        ax.annotate(f'{row["avg_omega"]:.1f}', 
                   (row['groove_diameter'], row['fuel_volume']),
                   textcoords="offset points",
                   xytext=(0, -20),
                   ha='center',
                   fontsize=8,
                   color='black',
                   weight='bold')
    
    # Set labels and title
    ax.set_xlabel('Groove Diameter (mm)', fontsize=14)
    ax.set_ylabel('Fuel Volume (μL)', fontsize=14)
    ax.set_title('Flame Propagation Speed Analysis\n(Darker color = Faster angular velocity)', fontsize=16)
    
    # Set grid
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    # Set axis limits with some padding
    x_margin = 0.5
    y_margin = 10
    ax.set_xlim(analysis_df['groove_diameter'].min() - x_margin, 
                analysis_df['groove_diameter'].max() + x_margin)
    ax.set_ylim(analysis_df['fuel_volume'].min() - y_margin, 
                analysis_df['fuel_volume'].max() + y_margin)
    
    plt.tight_layout()
    plt.savefig('groove_fuel_speed_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nGroove-Fuel-Speed analysis plot saved as 'groove_fuel_speed_analysis.png'")
    
    # Create a pivot table for better understanding
    print("\n" + "="*60)
    print("Summary Statistics:")
    print("="*60)
    
    # Group by groove diameter and fuel volume
    summary = analysis_df.groupby(['groove_diameter', 'fuel_volume'])['avg_omega'].agg(['mean', 'count'])
    summary = summary.round(2)
    print(summary)
    
    # Additional heatmap visualization
    print("\nCreating heatmap visualization...")
    
    # Prepare data for heatmap
    unique_grooves = sorted(analysis_df['groove_diameter'].unique())
    unique_fuels = sorted(analysis_df['fuel_volume'].unique())
    
    # Create a matrix for the heatmap
    heatmap_data = np.zeros((len(unique_fuels), len(unique_grooves)))
    heatmap_data[:] = np.nan
    
    for i, fuel in enumerate(unique_fuels):
        for j, groove in enumerate(unique_grooves):
            matching = analysis_df[(analysis_df['fuel_volume'] == fuel) & 
                                  (analysis_df['groove_diameter'] == groove)]
            if not matching.empty:
                heatmap_data[i, j] = matching['avg_omega'].mean()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create custom colormap with NaN values shown in gray
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('lightgray')
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', interpolation='nearest')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(unique_grooves)))
    ax.set_yticks(np.arange(len(unique_fuels)))
    ax.set_xticklabels([f'{g}mm' for g in unique_grooves])
    ax.set_yticklabels([f'{f}μL' for f in unique_fuels])
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Average Angular Velocity (ω)', fontsize=12)
    
    # Add text annotations
    for i in range(len(unique_fuels)):
        for j in range(len(unique_grooves)):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f'{heatmap_data[i, j]:.1f}',
                             ha="center", va="center", color="white", fontsize=10, weight='bold')
    
    ax.set_xlabel('Groove Diameter', fontsize=14)
    ax.set_ylabel('Fuel Volume', fontsize=14)
    ax.set_title('Angular Velocity Heatmap\n(Darker = Faster)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('groove_fuel_speed_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Heatmap saved as 'groove_fuel_speed_heatmap.png'")
    
else:
    print("Could not extract groove diameter and fuel volume from experiment titles")

print("\n" + "="*60)
print("All plots generated successfully!")
print("All data has been converted to positive values (absolute values)")
print("All titles have been translated to English")
print("="*60)