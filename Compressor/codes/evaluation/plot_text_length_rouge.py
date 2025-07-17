import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the paths and methods
base_path = "codes/evaluation/regenerate_results"
methods = {
    "ICAE": "ICAE_512to8_15epoch",
    "lora_param": "lora_param_512to8_15_epoch"
}

# Initialize data storage
text_lengths = [64, 192, 288, 384, 512]
compression_ratios = [length/8 for length in text_lengths]  # Calculate compression ratios
metrics = {
    "rouge-l-f": {"ICAE": [], "lora_param": []}
}

# Read data from CSV files
for method_name, folder_pattern in methods.items():
    folder_path = os.path.join(base_path, folder_pattern)
    
    # Read each text length file
    for length in text_lengths:
        csv_path = os.path.join(folder_path, f"{length}/averages.csv")
        
        if os.path.exists(csv_path):
            # Read CSV with the correct format
            df = pd.read_csv(csv_path, header=None, names=['metric', 'value'])
            # Convert to dictionary for easier access
            metrics_dict = dict(zip(df['metric'], df['value']))
            
            metrics["rouge-l-f"][method_name].append(metrics_dict.get('rouge-l-f', np.nan))
        else:
            print(f"Warning: {csv_path} not found")
            metrics["rouge-l-f"][method_name].append(np.nan)

# Create plot
plt.figure(figsize=(8, 6))

# Plot settings
colors = {'ICAE': '#00008B', 'lora_param': 'purple'}
markers = {'ICAE': 'o', 'lora_param': 's'}

# Plot ROUGE-L-F scores
for method_name in methods.keys():
    label_name = 'Ours' if method_name == 'lora_param' else method_name
    plt.plot(text_lengths, metrics["rouge-l-f"][method_name],
            marker=markers[method_name],
            color=colors[method_name],
            label=label_name,
            linewidth=1)
    for x, y in zip(text_lengths, metrics["rouge-l-f"][method_name]):
        plt.annotate(f'{y:.3f}', 
                    (x, y),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=10)
# Set labels and title
plt.xlabel('Text Length', fontsize=12)
plt.ylabel('ROUGE-L-F Score', fontsize=12)
# plt.title('ROUGE-L-F Score vs Text Length (512→2)', fontsize=14)

plt.legend(fontsize=12)

# Set x-axis ticks and compression ratio labels
ax = plt.gca()
ax.set_xticks(text_lengths)
plt.yticks(fontsize=12)

# Add compression ratio numbers at the top
ymin, ymax = ax.get_ylim()
# Extend y-axis limits by 15% at the top
ax.set_ylim(ymin, ymax * 1.03)

# Add compression ratio numbers at the top
top = ax.get_ylim()[1]  # Get new y-axis maximum value
for x, ratio in zip(text_lengths, compression_ratios):
    ax.text(x, top + 0.01 * top, f'{ratio:.0f}x',  # Place text slightly below the top
            ha='center', va='bottom', fontsize=10, color='black')

# Add explanation text
ax.text(0.5, 1.05, 'Compression Ratios', transform=ax.transAxes,
        ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.savefig('text_length_rouge_comparison_8.png', dpi=300, bbox_inches='tight')
plt.close()

# Print the data for verification
print("\nROUGE-L-F Scores by Text Length:")
for method_name in methods.keys():
    print(f"\n{method_name}:")
    for length, score in zip(text_lengths, metrics["rouge-l-f"][method_name]):
        print(f"Length {length}: {score:.4f}") 