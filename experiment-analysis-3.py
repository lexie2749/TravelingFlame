import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Set up matplotlib to display Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']  # Use SimHei for Chinese
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

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
                
                if not temp_df.empty:
                    experiment_dataframes.append({'info': current_experiment_info, 'df': temp_df})
                    print(f"  Saved {len(temp_df)} data points for: {current_experiment_info}")
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
            current_data.append([numeric_values[0], numeric_values[1]])  # Fixed the typo here!

# Don't forget to save the last experiment
if current_experiment_info is not None and len(current_data) > 0:
    try:
        temp_df = pd.DataFrame(current_data, columns=['t', 'omega'])
        # Clean the dataframe
        temp_df = temp_df.dropna()
        temp_df['t'] = pd.to_numeric(temp_df['t'], errors='coerce')
        temp_df['omega'] = pd.to_numeric(temp_df['omega'], errors='coerce')
        temp_df = temp_df.dropna()
        
        if not temp_df.empty:
            experiment_dataframes.append({'info': current_experiment_info, 'df': temp_df})
            print(f"  Saved {len(temp_df)} data points for: {current_experiment_info}")
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
            ax.set_xlabel('t (时间)', fontsize=9)
            ax.set_ylabel('ω (角速度)', fontsize=9)
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
        
        # Set plot title and labels
        plt.title(title, fontsize=14)
        plt.xlabel('t (时间)', fontsize=12)
        plt.ylabel('ω (角速度)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # Sanitize title for filename
        safe_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
        safe_title = safe_title[:50]  # Limit filename length
        filename = f"experiment_{i+1}_{safe_title}.png"
        
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Individual plot saved: {filename}")

# --- Analysis for specific groove sizes ---
print("\n" + "="*60)
print("Analysis for 125μL, 150μL, and 175μL experiments")
print("="*60)

# Filter experiments for specified volumes
relevant_experiments = []
for exp in experiment_dataframes:
    info_lower = exp['info'].lower()
    # Check for different possible formats
    if any(x in info_lower for x in ['125微升', '125μl', '150微升', '150μl', '175微升', '175μl']):
        relevant_experiments.append(exp)
        print(f"Found relevant experiment: {exp['info']}")

if relevant_experiments:
    # Create a new figure for the normalized data
    plt.figure(figsize=(12, 7))
    
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
            plt.scatter(df_exp['t_normalized'], df_exp['omega'], 
                       s=20, alpha=0.7, color=colors[idx], label=title)
            print(f"  Normalized and plotted: {title}")
    
    # Set plot title and labels for the combined plot
    plt.title('Normalized Time vs. Angular Velocity Comparison', fontsize=14)
    plt.xlabel('Normalized Time (0-1)', fontsize=12)
    plt.ylabel('ω (角速度)', fontsize=12)
    plt.legend(title="实验", loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    
    filename = "normalized_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nNormalized comparison plot saved as '{filename}'")
else:
    print("No experiments found with 125μL, 150μL, or 175μL volumes")

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)