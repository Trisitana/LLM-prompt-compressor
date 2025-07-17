import jieba
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from collections import Counter
import os

def calculate_metrics_averages(csv_file, average_output):
    data = pd.read_csv(csv_file)
    averages = data.mean()
    averages.to_csv(average_output, header=True, index=True)
    return averages

# initialize ROUGE
rouge = Rouge()
cc = SmoothingFunction()

def compute_exact_match(reference, candidate):
    return 1 if reference.strip() == candidate.strip() else 0

def compute_f1_score(reference_tokens, candidate_tokens):
    common = Counter(reference_tokens) & Counter(candidate_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(candidate_tokens)
    recall = num_same / len(reference_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def compute_metrics(reference, candidate):
    scores = {}

    # Compute Rouge scores
    # For Chinese text, we need to tokenize first before calculating ROUGE scores
    reference_tokens = ' '.join(list(jieba.cut(reference, cut_all=False)))
    candidate_tokens = ' '.join(list(jieba.cut(candidate, cut_all=False)))
    rouge_scores = rouge.get_scores(candidate_tokens, reference_tokens)[0]
    for key in ['rouge-1', 'rouge-2', 'rouge-l']:
        scores[f'{key}-p'] = rouge_scores[key]['p']
        scores[f'{key}-r'] = rouge_scores[key]['r']
        scores[f'{key}-f'] = rouge_scores[key]['f']

    # Compute BLEU score using jieba for Chinese tokenization
    reference_tokens = list(jieba.cut(reference, cut_all=False))
    candidate_tokens = list(jieba.cut(candidate, cut_all=False))
    scores['bleu'] = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=cc.method1)

    # Compute exact match
    scores['exact_match'] = compute_exact_match(reference, candidate)
    
    # Compute F1 score using jieba tokens
    scores['f1'] = compute_f1_score(reference_tokens, candidate_tokens)

    return scores

def main(ref_file, cand_file, output_file):
    with open(ref_file, 'r', encoding='utf-8') as ref_f, open(cand_file, 'r', encoding='utf-8') as cand_f:
        references = ref_f.readlines()
        candidates = cand_f.readlines()

    results = []
    i = 1
    for reference, candidate in zip(references, candidates):
        print(f"Processing line {i}")
        i += 1
        if candidate.strip() == "":
            print("Candidate is empty")
            continue
        result = compute_metrics(reference.strip(), candidate.strip())
        results.append(result)

    # Write results to CSV file
    with open(output_file, 'w', encoding='utf-8') as f:
        headers = [
            "line_number",
            "rouge-1-p", "rouge-1-r", "rouge-1-f",
            "rouge-2-p", "rouge-2-r", "rouge-2-f",
            "rouge-l-p", "rouge-l-r", "rouge-l-f",
            "bleu",
            "exact_match",
            "f1"
        ]
        f.write(",".join(headers) + "\n")
        for idx, result in enumerate(results):
            line_values = [
                f"{idx + 1}",
                f"{result['rouge-1-p']:.4f}", f"{result['rouge-1-r']:.4f}", f"{result['rouge-1-f']:.4f}",
                f"{result['rouge-2-p']:.4f}", f"{result['rouge-2-r']:.4f}", f"{result['rouge-2-f']:.4f}",
                f"{result['rouge-l-p']:.4f}", f"{result['rouge-l-r']:.4f}", f"{result['rouge-l-f']:.4f}",
                f"{result['bleu']:.4f}",
                f"{result['exact_match']}",
                f"{result['f1']:.4f}"
            ]
            f.write(",".join(line_values) + "\n")

if __name__ == "__main__":

    num_mem = [1, 2, 4, 8]

    for num_e in num_mem:

        for root in [f"codes/evaluation/regenerate_results/qwen_param_512to{num_e}_3_epoch", f"codes/evaluation/regenerate_results/qwen_ICAE_512to{num_e}_3epoch"]:
                nums = ['64', '128', '256', '384', '512']
                all_averages = []

                # Calculate metrics for each num
                for num in nums:
                    print(f"\nProcessing num={num}")
                    ref_file = os.path.join(root, num, 'target.txt')
                    cand_file = os.path.join(root, num, 'output.txt')
                    output_file = os.path.join(root, num, 'evaluate_metrics_cn.csv')
                    main(ref_file, cand_file, output_file)

                    csv_file = output_file
                    average_output = os.path.join(root, num, 'average_results_cn.csv')

                    averages = calculate_metrics_averages(csv_file, average_output)
                    print(f"Average metrics for num={num}:")
                    print(averages)
                    
                    all_averages.append(averages)

                # Calculate overall average across all nums
                overall_average = pd.DataFrame(all_averages).mean()
                print("\nOverall average metrics across all nums:")
                print(overall_average)
                
                overall_output = os.path.join(root, 'overall_average_results_cn.csv')
                overall_average.to_csv(overall_output, header=True)