# Create plot axes for the first line plot
plt.axes ([0.05, .05, 0.425, 0.9])

# Plot in blue the % of degrees awarded to women in the Physical Sciences
plt.plot(year,physical_sciences, color='blue')

# Create plot axes for the second line plot
plt.axes([0.525, .05, 0.425, 0.9])

# Plot in red the % of degrees awarded to women in Computer Science
plt.plot (year,computer_science, color='red')

# Display the plot
plt.show()
