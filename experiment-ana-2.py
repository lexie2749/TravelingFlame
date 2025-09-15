import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Load the data with no headers
df_full = pd.read_excel('实验数据1.xlsx', sheet_name='Sheet3', header=None)

# Initialize lists to store the extracted data
calculated_data = []
current_groove_diameter = None
current_fuel_amount = None
current_data_block = []

# Iterate through the DataFrame rows to parse the data
for index, row in df_full.iterrows():
    # Check for the start of a new data block
    if pd.notna(row[0]) and 'mm凹槽' in str(row[0]):
        # A new section has been found. Process the previous one if it exists.
        if current_data_block:
            # Create a temporary DataFrame for the previous block
            block_df = pd.DataFrame(current_data_block, columns=['t', 'theta'])
            
            # Convert columns to numeric types, coercing errors to NaN
            block_df['t'] = pd.to_numeric(block_df['t'], errors='coerce')
            block_df['theta'] = pd.to_numeric(block_df['theta'], errors='coerce')

            # Drop rows with NaN values after conversion
            block_df.dropna(subset=['t', 'theta'], inplace=True)
            
            if not block_df.empty and block_df.shape[0] > 1:
                # Calculate delta_t and delta_theta
                delta_t = block_df['t'].iloc[-1] - block_df['t'].iloc[0]
                delta_theta = block_df['theta'].iloc[-1] - block_df['theta'].iloc[0]

                # Check for zero division
                if delta_t != 0:
                    omega_deg_s = delta_theta / delta_t
                    
                    # Store the results
                    calculated_data.append({
                        'groove_diameter': current_groove_diameter,
                        'fuel_amount': current_fuel_amount,
                        'omega_deg_s': omega_deg_s
                    })
        
        # Reset for the new section
        try:
            groove_diameter_str = str(row[0])
            fuel_amount_str = str(row[1])

            # Use regex to extract numeric values
            current_groove_diameter = float(re.findall(r'(\d+\.?\d*)', groove_diameter_str)[0])
            current_fuel_amount = float(re.findall(r'(\d+\.?\d*)', fuel_amount_str)[0])

        except (IndexError, ValueError):
            # Skip rows that don't match the expected pattern
            current_groove_diameter = None
            current_fuel_amount = None
            continue

        current_data_block = []
    # Check if the row contains numerical data for t and theta, but only after a block header is found
    elif pd.to_numeric(row[0], errors='coerce') is not None and pd.to_numeric(row[1], errors='coerce') is not None:
        if current_groove_diameter is not None and current_fuel_amount is not None:
            current_data_block.append([row[0], row[1]])

# Process the last data block
if current_data_block:
    block_df = pd.DataFrame(current_data_block, columns=['t', 'theta'])
    block_df['t'] = pd.to_numeric(block_df['t'], errors='coerce')
    block_df['theta'] = pd.to_numeric(block_df['theta'], errors='coerce')
    block_df.dropna(subset=['t', 'theta'], inplace=True)
    if not block_df.empty and block_df.shape[0] > 1:
        delta_t = block_df['t'].iloc[-1] - block_df['t'].iloc[0]
        delta_theta = block_df['theta'].iloc[-1] - block_df['theta'].iloc[0]
        if delta_t != 0:
            omega_deg_s = delta_theta / delta_t
            calculated_data.append({
                'groove_diameter': current_groove_diameter,
                'fuel_amount': current_fuel_amount,
                'omega_deg_s': omega_deg_s
            })


# Create a DataFrame from the calculated data
if not calculated_data:
    raise ValueError("No valid data blocks found for calculation.")

df_final = pd.DataFrame(calculated_data)

# Convert omega from degrees/s to rad/s and take the absolute value
df_final['angular_velocity_rad_s'] = np.abs(df_final['omega_deg_s'] * (np.pi / 180))

# Convert groove diameter from mm to m
df_final['groove_diameter_m'] = df_final['groove_diameter'] / 1000

# Calculate groove area
df_final['groove_area_m2'] = np.pi * (0.15**2 - (0.15 - df_final['groove_diameter_m'])**2)

# Convert fuel amount from microliters to cubic meters (1 microliter = 1e-9 m^3)
df_final['fuel_amount_m3'] = df_final['fuel_amount'] * 1e-9

# Calculate solution height in meters
df_final['solution_height_m'] = df_final['fuel_amount_m3'] / df_final['groove_area_m2']

# Scale solution height to 10^-3 m
df_final['solution_height_scaled'] = df_final['solution_height_m'] * 1000

# Group by groove diameter and create a plot for each
groove_diameters = df_final['groove_diameter'].unique()

# Define a consistent order for plotting
plot_order = sorted(groove_diameters)

# Check if there are any valid diameters to plot
if not plot_order:
    raise ValueError("No valid groove diameters found in the data.")

# Create a scatter plot for each groove diameter
for diameter in plot_order:
    plt.figure(figsize=(10, 6))
    data_to_plot = df_final[df_final['groove_diameter'] == diameter]
    plt.scatter(data_to_plot['solution_height_scaled'], data_to_plot['angular_velocity_rad_s'])
    plt.title(f'Angular Velocity vs. Solution Height for {diameter}mm Groove')
    plt.xlabel('Solution Height ($10^{-3} m$)')
    plt.ylabel('Angular Velocity ($rad/s$)')
    plt.grid(False)
    plt.savefig(f'angular_velocity_vs_solution_height_{diameter}mm.png')
    plt.close()

# Print confirmation of the generated plots
for diameter in plot_order:
    print(f'Plot for {diameter}mm groove is saved as angular_velocity_vs_solution_height_{diameter}mm.png')

# Save the final DataFrame to a CSV file for the user
df_final.to_csv('calculated_delta_method_data.csv', index=False)
print('The processed data is saved to a CSV file named calculated_delta_method_data.csv.')