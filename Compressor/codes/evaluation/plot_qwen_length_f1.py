import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the paths and methods
base_path = "codes/evaluation/regenerate_results"
methods = {
    "ICAE": "qwen_ICAE_512to1_3epoch",
    "lora_param": "qwen_param_512to1_3_epoch"
}

# Initialize data storage
text_lengths = [64, 128, 256, 384, 512]
compression_ratios = [length for length in text_lengths]  # Calculate compression ratios
metrics = {
    "f1": {"ICAE": [], "lora_param": []}
}

# Read data from CSV files
for method_name, folder_pattern in methods.items():
    folder_path = os.path.join(base_path, folder_pattern)
    
    # Read each text length file
    for length in text_lengths:
        csv_path = os.path.join(folder_path, f"{length}/average_results_cn.csv")
        
        if os.path.exists(csv_path):
            # Read CSV with the correct format
            df = pd.read_csv(csv_path, header=None, names=['metric', 'value'])
            # Convert to dictionary for easier access
            metrics_dict = dict(zip(df['metric'], df['value']))
            
            metrics["f1"][method_name].append(metrics_dict.get('f1', np.nan))
        else:
            print(f"Warning: {csv_path} not found")
            metrics["f1"][method_name].append(np.nan)

# Create plot
plt.figure(figsize=(8, 6))

# Plot settings
colors = {'ICAE': '#00008B', 'lora_param': 'purple'}
markers = {'ICAE': 'o', 'lora_param': 's'}

# Plot F1 scores
for method_name in methods.keys():
    label_name = 'Ours' if method_name == 'lora_param' else method_name
    plt.plot(text_lengths, metrics["f1"][method_name],
            marker=markers[method_name],
            color=colors[method_name],
            label=label_name,
            linewidth=1)

# Set labels and title
plt.xlabel('Text Length', fontsize=12)
plt.ylabel('F1 Score', fontsize=12)

plt.legend(fontsize=12)

# Set x-axis ticks and compression ratio labels
ax = plt.gca()
ax.set_xticks(text_lengths)
plt.yticks(fontsize=12)

# Add compression ratio numbers at the top
top = ax.get_ylim()[1]  # Get y-axis maximum value
for x, ratio in zip(text_lengths, compression_ratios):
    ax.text(x, top + 0.01 * top, f'{ratio:.0f}x',  # Add text slightly above y-axis maximum
            ha='center', va='bottom', fontsize=10, color='black')

# Add explanation text
ax.text(0.5, 1.05, 'Compression Ratios', transform=ax.transAxes,
        ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.savefig('text_qwen_f1_comparison_1.png', dpi=300, bbox_inches='tight')
plt.close()

# Print the data for verification
print("\nF1 Scores by Text Length:")
for method_name in methods.keys():
    print(f"\n{method_name}:")
    for length, score in zip(text_lengths, metrics["f1"][method_name]):
        print(f"Length {length}: {score:.4f}") 