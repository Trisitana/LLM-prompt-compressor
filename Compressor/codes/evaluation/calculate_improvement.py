import pandas as pd
import numpy as np
import os

# Define the paths and compression ratios
base_path = "codes/evaluation"
compression_ratios = [1, 2, 4, 8]

methods = {
    "ICAE": "ICAEQA_qwen/{}",
    "lora_param": "512QA_qwen/{}"
}

# Initialize data storage
metrics = {
    "bleu": {"ICAE": [], "lora_param": []},
    "f1": {"ICAE": [], "lora_param": []},
    "exact_match": {"ICAE": [], "lora_param": []}
}

# Read data from CSV files
for ratio in compression_ratios:
    for method_name, folder_pattern in methods.items():
        folder_path = os.path.join(base_path, folder_pattern.format(ratio))
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

# Calculate and print improvements
print("\nPerformance Improvements (Our Method vs ICAE):")
print("=" * 60)

for metric_name in metrics.keys():
    print(f"\n{metric_name.upper()} Improvements:")
    print("-" * 50)
    
    for i, ratio in enumerate(compression_ratios):
        icae_score = metrics[metric_name]["ICAE"][i]
        our_score = metrics[metric_name]["lora_param"][i]
        
        if not np.isnan(icae_score) and not np.isnan(our_score):
            improvement = ((our_score - icae_score) / icae_score) * 100
            print(f"\nCompression Ratio 1/{ratio}x:")
            print(f"  ICAE Score: {icae_score:.4f}")
            print(f"  Our Score:  {our_score:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
        else:
            print(f"\nCompression Ratio 1/{ratio}x: Data not available") 