import pandas as pd
import matplotlib.pyplot as plt

# Define the file path
# file_path = 'complete/record-host.txt'
file_path = 'complete_2.0/replay-HardwareMonitoring.txt'

# Define column names based on the header in the text file
column_names = ['Tag', 'Time', 'GPU usage', 'CPU usage', 'Framerate', 'Frametime', 'Framerate Avg']

# Read the text file into a DataFrame
df = pd.read_csv(file_path, delimiter=',', skipinitialspace=True, names=column_names, skiprows=1, parse_dates=['Time'])
df['Time'] = (df['Time'] - df['Time'].min()).dt.total_seconds()
# Plotting
plt.figure(figsize=(10, 6))

plt.plot(df['Time'], df['GPU usage'], label='GPU usage')
plt.plot(df['Time'], df['CPU usage'], label='CPU usage')
# plt.plot(df['Time'], df['Framerate'], label='Framerate')
# plt.plot(df['Time'], df['Frametime'], label='Frametime')

plt.title('Performance Metrics Over Time')
plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.grid(True)

plt.show()
