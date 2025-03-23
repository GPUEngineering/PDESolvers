import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
file_path = '~/Downloads/simulation_results_serial.csv'
df = pd.read_csv(file_path)

# Plot the data
plt.figure(figsize=(10, 6))

for simulation in df['Simulation'].unique():
    simulation_data = df[df['Simulation'] == simulation]
    plt.plot(simulation_data['Time Step'], simulation_data['Stock Price'], color='gray', alpha=0.2)

# Add labels and title
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.title('Geometric Brownian Motion Simulations')

# Show the plot
plt.show()
