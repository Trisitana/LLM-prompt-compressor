import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Define the paths and compression ratios
base_path = "codes/evaluation/"
compression_ratios = [1, 2, 4, 8]

methods = {
    "lora_param": "512QA_qwen/{}",
    "ICAE": "ICAEQA_qwen/{}", 
    "SC": "select_qwen/{}"
}

# methods = {
#     "lora_param": "512QA_weight/{}",
#     "ICAE": "ICAEQA/{}", 
#     "SC": "select_result/{}"
# }

# Initialize data storage
metrics = {
    "bleu": {"ICAE": [], "lora_param": [], "SC": []},
    "f1": {"ICAE": [], "lora_param": [], "SC": []},
    "exact_match": {"ICAE": [], "lora_param": [], "SC": []}
}

# Read data from CSV files
for ratio in compression_ratios:
    for method_name, folder_pattern in methods.items():
        folder_path = os.path.join(base_path, folder_pattern.format(ratio))
        # csv_path = os.path.join(folder_path, "overall_averages.csv"
        # csv_path = os.path.join(folder_path, "average_results.csv")

        csv_path = os.path.join(folder_path, "average_results_cn.csv")
        
        if os.path.exists(csv_path):
            # Read CSV with the correct format
            df = pd.read_csv(csv_path, header=None, names=['metric', 'value'])
            # Convert to dictionary for easier access
            metrics_dict = dict(zip(df['metric'], df['value']))
            
            metrics["bleu"][method_name].append(metrics_dict.get('bleu', np.nan))
            metrics["f1"][method_name].append(metrics_dict.get('f1', np.nan))
            metrics["exact_match"][method_name].append(metrics_dict.get('exact_match', np.nan))
        else:
            print(f"Warning: {csv_path} not found")
            metrics["bleu"][method_name].append(np.nan)
            metrics["f1"][method_name].append(np.nan)
            metrics["exact_match"][method_name].append(np.nan)

# Print metrics data
print("\nMetrics Data:")
for metric_name, metric_data in metrics.items():
    print(f"\n{metric_name.upper()}:")
    for method_name, values in metric_data.items():
        print(f"{method_name}: {values}")


# Create plots
# Plot settings
colors = {'ICAE': '#00008B', 'lora_param': 'purple', 'SC': 'green'}
markers = {'ICAE': 'o', 'lora_param': 's', 'SC': '^'}

# Plot each metric separately
for metric_name, metric_data in metrics.items():
    plt.figure(figsize=(6, 4))
    
    # Increase font sizes
    plt.rcParams.update({'font.size': 14})
    
    for method_name in methods.keys():
        label_name = 'Ours' if method_name == 'lora_param' else method_name
        plt.plot(compression_ratios, metric_data[method_name],
                marker=markers[method_name],
                color=colors[method_name],
                label=label_name,
                linewidth=1)
    
    plt.xlabel('Compressed tokens', fontsize=16)
    # Change display name for exact_match to EM
    ylabel = 'EM' if metric_name == 'exact_match' else metric_name.upper()
    plt.ylabel(ylabel, fontsize=16)
    # Change title display name for exact_match to EM
    title_metric = 'EM' if metric_name == 'exact_match' else metric_name.upper()
    plt.title(f'{title_metric} Score for QA Setting', fontsize=16)
    plt.legend(fontsize=10)
    
    # Set x-axis ticks to match compression ratios
    plt.xticks(compression_ratios, fontsize=14)
    plt.yticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'Figures/512QA_qwen/{metric_name}_comparison_qwen.png', dpi=300, bbox_inches='tight')
    plt.close()