import os
import matplotlib.pyplot as plt

# Data
x = [75, 100, 123, 150, 175, 200, 250, 300, 350, 400, 450]

sweetness = [74.12, 77.65, 78.24, 76.76, 77.06, 75.88, 76.18, 75.00, 76.47, 73.24, 74.71]
saltiness = [74.41, 80.00, 79.41, 79.41, 77.65, 75.59, 79.12, 76.47, 75.88, 76.47, 77.35]
sourness  = [65.29, 71.18, 70.00, 66.18, 66.47, 66.47, 67.35, 64.41, 66.47, 62.94, 61.18]
bitterness= [61.76, 68.82, 60.29, 62.06, 61.47, 60.29, 58.24, 54.71, 53.82, 54.71, 54.12]
umaminess = [70.29, 68.82, 73.24, 66.76, 70.88, 72.35, 71.18, 69.12, 65.88, 70.00, 71.47]
fattiness = [70.88, 74.12, 75.29, 73.82, 74.12, 71.18, 72.94, 74.12, 74.41, 75.00, 74.12]

average   = [69.46, 73.43, 72.75, 70.83, 71.27, 70.29, 70.83, 68.97, 68.82, 68.73, 68.82]


series = {
    "Sweetness": sweetness,
    "Saltiness": saltiness,
    "Sourness": sourness,
    "Bitterness": bitterness,
    "Umaminess": umaminess,
    "Fattiness": fattiness
}

plt.figure(figsize=(9, 6))

# Plot the six taste curves
for label, y in series.items():
    plt.plot(x, y, marker='o', linewidth=2, label=label)

# Plot the average in bold (thicker line and larger markers)
plt.plot(x, average, marker='o', linewidth=4, markersize=7, label="Average")

# Styling
# plt.title("Accuracy vs. Number of Training Datapoints", fontsize=18)
plt.xlabel("Number of examples", fontsize=18)
plt.ylabel("Accuracy (%)", fontsize=18)
plt.grid(True, linestyle='--', alpha=0.4)
plt.ylim(50, 85)
plt.xlim(min(x)-10, max(x)+10)
plt.legend(title="Taste Dimension", loc="lower left", fontsize=14, title_fontsize=14)
plt.tight_layout()


# Save the figure for download
output_path = "~/research/conceptual-spaces/results/taste_accuracy_vs_datapoints.png"
output_path = os.path.expanduser(output_path)

plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()
