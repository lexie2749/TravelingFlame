import io
import pandas as pd
import math

# The new data provided by the user
data_string = """

t	x	y	θ	ω
9.333333E-1	-2.423817E2	8.409160E1	1.608664E2	
1.000000E0	-1.879695E2	-1.607634E2	2.205392E2	8.651196E2
1.066667E0	2.720611E1	-2.498015E2	2.762156E2	8.373782E2
1.133333E0	2.250687E2	-1.187176E2	3.321896E2	6.031343E2
1.200000E0	2.522748E2	-1.483969E1	3.566335E2	3.495773E2
1.266667E0	2.324885E2	7.914504E1	3.787999E2	3.572884E2
1.333333E0	1.739567E2	1.695913E2	4.042720E2	4.665422E2
1.400000E0	3.790768E1	2.394889E2	4.410055E2	5.441691E2
1.466667E0	-1.088244E2	2.151756E2	4.768279E2	5.947685E2
1.533333E0	-2.349618E2	8.409160E1	5.203080E2	6.093334E2
1.600000E0	-2.349618E2	-7.667176E1	5.580723E2	

"""

# Split the data string into lines, and strip leading/trailing whitespace
lines = data_string.strip().split('\n')

# Get the column names from the first line
columns = lines[0].strip().split()

# Process the data lines
data_rows = []
for line in lines[1:]:
    # Split each line by whitespace and convert to float
    row = [float(val) for val in line.strip().split()]
    data_rows.append(row)

# Create the DataFrame
df = pd.DataFrame(data_rows, columns=columns)

# The column names are 't' and 'θ'
t_first = df['t'].iloc[0]
t_last = df['t'].iloc[-1]
theta_first = df['θ'].iloc[0]
theta_last = df['θ'].iloc[-1]

# Perform the calculation for average angular velocity
result = (theta_last - theta_first) / (t_last - t_first) * math.pi / 180  # Convert degrees to radians

# Calculate the standard deviation of the 'ω' column for the error bar size
std_omega = df['ω'].std() / 10 * math.pi / 180  # Convert degrees to radians

print(f"The average angular velocity is: {result}")
print(f"The standard deviation of the angular velocity (ω) is: {std_omega}")