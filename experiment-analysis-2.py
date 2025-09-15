import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np

# Set up matplotlib to display Chinese characters
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Use SimHei for Chinese
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display

# Read the Excel file
df_raw = pd.read_excel('实验数据2(1).xlsx', header=None)

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

# Iterate through the rows to identify experiment blocks
for index, row in df_raw.iterrows():
    # Convert all elements in the row to string for checking
    row_values = row.tolist()
    
    # Check if any cell in this row contains Chinese characters (this is likely the title row)
    if any(contains_chinese(cell) for cell in row_values):
        # Found a new experiment title
        if current_experiment_info is not None and len(current_data) > 0:
            # Save the previous experiment's data
            temp_df = pd.DataFrame(current_data, columns=['t', 'omega'])
            # Clean the dataframe
            temp_df = temp_df.dropna()
            temp_df = temp_df[temp_df['t'].apply(lambda x: isinstance(x, (int, float)))]
            temp_df = temp_df[temp_df['omega'].apply(lambda x: isinstance(x, (int, float)))]
            temp_df['t'] = pd.to_numeric(temp_df['t'], errors='coerce')
            temp_df['omega'] = pd.to_numeric(temp_df['omega'], errors='coerce')
            temp_df = temp_df.dropna()
            
            if not temp_df.empty:
                experiment_dataframes.append({'info': current_experiment_info, 'df': temp_df})
        
        # Extract the Chinese text as experiment info
        chinese_cells = [str(cell) for cell in row_values if contains_chinese(cell)]
        current_experiment_info = " ".join(chinese_cells)
        current_data = []
        data_started = False
        print(f"Found new experiment: {current_experiment_info}")
        continue
    
    # Check if this row contains 't' and 'ω' or 'omega' headers
    row_str = [str(x).lower() for x in row_values]
    if 't' in row_str or ('ω' in [str(x) for x in row_values]) or 'omega' in row_str:
        data_started = True
        # Find the indices of t and omega columns
        t_index = None
        omega_index = None
        
        for i, val in enumerate(row_values):
            val_str = str(val).lower()
            if val_str == 't':
                t_index = i
            elif str(val) == 'ω' or val_str == 'omega' or val_str == 'w':
                omega_index = i
        
        # If we can't find both columns, try adjacent columns
        if t_index is not None and omega_index is None:
            omega_index = t_index + 1
        elif omega_index is not None and t_index is None:
            t_index = omega_index - 1
        
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
    temp_df = pd.DataFrame(current_data, columns=['t', 'omega'])
    # Clean the dataframe
    temp_df = temp_df.dropna()
    temp_df = temp_df[temp_df['t'].apply(lambda x: isinstance(x, (int, float)))]
    temp_df = temp_df[temp_df['omega'].apply(lambda x: isinstance(x, (int, float)))]
    temp_df['t'] = pd.to_numeric(temp_df['t'], errors='coerce')
    temp_df['omega'] = pd.to_numeric(temp_df['omega'], errors='coerce')
    temp_df = temp_df.dropna()
    
    if not temp_df.empty:
        experiment_dataframes.append({'info': current_experiment_info, 'df': temp_df})

print(f"\nTotal experiments detected: {len(experiment_dataframes)}")

# Create a subplot figure for all experiments
if len(experiment_dataframes) > 0:
    # Calculate grid dimensions
    n_experiments = len(experiment_dataframes)
    n_cols = min(3, n_experiments)  # Maximum 3 columns
    n_rows = (n_experiments + n_cols - 1) // n_cols
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    
    # Flatten axes array for easier iteration
    if n_experiments == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    # Plot each experiment
    for i, exp in enumerate(experiment_dataframes):
        title = exp['info']
        df_exp = exp['df']
        
        if not df_exp.empty:
            # Plot on subplot
            ax = axes[i] if n_experiments > 1 else axes[0]
            ax.plot(df_exp['t'], df_exp['omega'], marker='o', linestyle='-', markersize=3)
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
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('all_experiments_combined.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nCombined plot saved as 'all_experiments_combined.png'")

# Also save individual plots
for i, exp in enumerate(experiment_dataframes):
    title = exp['info']
    df_exp = exp['df']
    
    if not df_exp.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(df_exp['t'], df_exp['omega'], marker='o', linestyle='-', markersize=4)
        
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

print("\nAll plots generated successfully!")