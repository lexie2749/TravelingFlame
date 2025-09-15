import pandas as pd

# File to analyze
file_name = "实验数据1.xlsx"

# Read the data, skipping the first two rows and without a header.
# Manually assign the column names.
df_sheet3 = pd.read_excel(file_name, sheet_name='Sheet3', skiprows=2, header=None, names=['t', 'x', 'y', 'θ', 'ω'])

# The columns might be objects. Convert them to numeric types.
# Use errors='coerce' to turn non-numeric values into NaN, which can be dropped.
df_sheet3['t'] = pd.to_numeric(df_sheet3['t'], errors='coerce')
df_sheet3['θ'] = pd.to_numeric(df_sheet3['θ'], errors='coerce')
df_sheet3['ω'] = pd.to_numeric(df_sheet3['ω'], errors='coerce')

# Drop any rows with NaN values resulting from the conversion
df_sheet3.dropna(subset=['t', 'θ', 'ω'], inplace=True)

# Perform the calculations on the cleaned DataFrame
if not df_sheet3.empty:
    t_first = df_sheet3['t'].iloc[0]
    t_last = df_sheet3['t'].iloc[-1]
    theta_first = df_sheet3['θ'].iloc[0]
    theta_last = df_sheet3['θ'].iloc[-1]

    # Calculate average angular velocity
    result = (theta_last - theta_first) / (t_last - t_first)

    # Calculate the standard deviation of the 'ω' column
    std_omega = df_sheet3['ω'].std()

    print(f"\nAverage angular velocity for {file_name}: {result}")
    print(f"Standard deviation of angular velocity (ω) for {file_name}: {std_omega}")
else:
    print(f"Could not perform calculation on {file_name} as the DataFrame is empty after cleaning.")