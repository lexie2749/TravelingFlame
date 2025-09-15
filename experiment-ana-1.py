import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the XLSX file, specifying the sheet name.
df_full = pd.read_excel('实验数据1.xlsx', sheet_name='Sheet3', header=None, engine='openpyxl')

results = []
current_groove = None
current_fuel = None

for i in range(len(df_full)):
    row = df_full.iloc[i]
    if isinstance(row[0], str) and '凹槽' in row[0]:
        current_groove = row[0]
        current_fuel = row[1]
    elif isinstance(row[0], str) and row[0].strip() == 't':
        # Found the header for a data block
        time_col = i + 1
        
        # Determine column indices for t, theta, and omega
        t_idx = None
        theta_idx = None
        omega_idx = None
        
        # Look for the column headers 't', 'θ', 'ω'
        for j, col_val in enumerate(row):
            if isinstance(col_val, str):
                if col_val.strip() == 't':
                    t_idx = j
                elif col_val.strip() == 'θ':
                    theta_idx = j
                elif col_val.strip() == 'ω':
                    omega_idx = j
        
        if t_idx is not None and theta_idx is not None and omega_idx is not None:
            data_start = i + 1
            data_end = data_start
            while data_end < len(df_full) and pd.notna(df_full.iloc[data_end, t_idx]):
                data_end += 1
            
            # Extract data block
            data_block = df_full.iloc[data_start:data_end, [t_idx, theta_idx, omega_idx]].copy()
            data_block.columns = ['t', 'theta', 'omega']
            
            # Convert to numeric, coercing errors
            data_block['t'] = pd.to_numeric(data_block['t'], errors='coerce')
            data_block['theta'] = pd.to_numeric(data_block['theta'], errors='coerce')
            data_block['omega'] = pd.to_numeric(data_block['omega'], errors='coerce')
            
            # Drop rows with NaN values for the calculation
            data_block = data_block.dropna(subset=['t', 'theta'])
            
            if not data_block.empty:
                # Calculate angular velocity
                delta_t = data_block['t'].iloc[-1] - data_block['t'].iloc[0]
                delta_theta = data_block['theta'].iloc[-1] - data_block['theta'].iloc[0]
                
                # Avoid division by zero
                if delta_t != 0:
                    angular_velocity = delta_theta / delta_t
                else:
                    angular_velocity = np.nan
                
                # Calculate error bar from omega, dropping NaN values
                omega_values = data_block['omega'].dropna()
                error_bar = omega_values.std() if not omega_values.empty else np.nan
                
                # Store the results
                results.append({
                    'groove_mm': int(''.join(filter(str.isdigit, current_groove))),
                    'fuel_ul': int(''.join(filter(str.isdigit, current_fuel))),
                    'angular_velocity_deg_per_s': angular_velocity,
                    'omega_std_dev_deg_per_s': error_bar
                })

# Create the final DataFrame
results_df = pd.DataFrame(results)

# Convert angular velocity and omega std dev to rad/s
results_df['angular_velocity_rad_per_s'] = results_df['angular_velocity_deg_per_s'] * (np.pi / 180)
results_df['omega_std_dev_rad_per_s'] = results_df['omega_std_dev_deg_per_s'] * (np.pi / 180)

# Take the absolute value of angular velocity
results_df['angular_velocity_rad_per_s'] = np.abs(results_df['angular_velocity_rad_per_s'])

# Filter for the grooves from 3mm to 10mm
filtered_df = results_df[(results_df['groove_mm'] >= 3) & (results_df['groove_mm'] <= 10)]

# Sort the data
filtered_df = filtered_df.sort_values(by=['groove_mm', 'fuel_ul'])

# Generate plots for each groove size
groove_sizes = sorted(filtered_df['groove_mm'].unique())

for groove in groove_sizes:
    groove_df = filtered_df[filtered_df['groove_mm'] == groove]

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        groove_df['fuel_ul'],
        groove_df['angular_velocity_rad_per_s'],
        yerr=groove_df['omega_std_dev_rad_per_s'],
        fmt='o-',
        capsize=5
    )
    plt.title(f'Relationship between Fuel and Angular Velocity for {groove}mm Groove\n{groove}mm凹槽燃料与角速度的关系', fontsize=14)
    plt.xlabel('Fuel Volume (microliters)\n燃料体积(微升)', fontsize=12)
    plt.ylabel('Angular Velocity (rad/s)\n角速度 (rad/s)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'absolute_angular_velocity_groove_{groove}mm.png')
    plt.close()

print(f"Generated {len(groove_sizes)} plots with absolute angular velocity.")