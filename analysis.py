import matplotlib.pyplot as plt

# Data from the provided performance metrics
algorithms = ['Linear', 'Polynomial', 'Tree', 'Forest', 'ANN']
execution_time_1min = [0.5127394199371338, 0.023656845092773438, 2.503251552581787, 13.294403076171875, 11.016113519668579]
mse_1min = [0.824351602771976, 0.17165778494323544, 6.895321355876022e-06, 4.842415974691045e-06, 0.0039715413608220436]

execution_time_5min = [0.4859471321105957, 0.06461596488952637, 16.008439779281616, 83.57461714744568, 73.4111099243164]
mse_5min = [0.9821924944323539, 0.04996606174539488, 0.0002988672509621931, 0.0001938067749676556, 0.0037467385411934255]

# Calculating performance ratio (performance/cost) - Lower MSE and lower execution time are better
# Since MSE close to 0 is better, we use its inverse for the ratio
performance_ratio_1min = [(1/mse) / time for mse, time in zip(mse_1min, execution_time_1min)]
performance_ratio_5min = [(1/mse) / time for mse, time in zip(mse_5min, execution_time_5min)]

# Plotting the performance ratio
plt.figure(figsize=(14, 7))

# Plotting the performance ratio in separate figures

# Plot for 1 minute interval data
plt.figure(figsize=(7, 5))
plt.bar(algorithms, performance_ratio_1min, color='blue', alpha=0.7)
plt.title('Performance/Cost Ratio for 1 Min Interval Data')
plt.ylabel('Performance/Cost Ratio')
plt.xlabel('Algorithms')
plt.xticks(rotation=45)
plt.show()

# Plot for 5 minutes interval data
plt.figure(figsize=(7, 5))
plt.bar(algorithms, performance_ratio_5min, color='green', alpha=0.7)
plt.title('Performance/Cost Ratio for 5 Min Interval Data')
plt.ylabel('Performance/Cost Ratio')
plt.xlabel('Algorithms')
plt.xticks(rotation=45)
plt.show()
