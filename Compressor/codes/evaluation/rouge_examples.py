from rouge import Rouge
import pandas as pd
import jieba

def print_rouge_scores(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)[0]
    
    print(f"\nReference: {reference}")
    print(f"Candidate: {candidate}")
    print("\nROUGE Scores:")
    for key in ['rouge-1', 'rouge-2', 'rouge-l']:
        print(f"{key}:")
        print(f"  Precision: {scores[key]['p']:.4f}")
        print(f"  Recall: {scores[key]['r']:.4f}")
        print(f"  F1: {scores[key]['f']:.4f}")
    print("-" * 50)

def print_jieba_tokenization(text):
    # 精确模式
    tokens = list(jieba.cut(text, cut_all=False))
    print(f"\nText: {text}")
    print(f"Jieba tokens (精确模式): {tokens}")
    
    # 全模式
    tokens_all = list(jieba.cut(text, cut_all=True))
    print(f"Jieba tokens (全模式): {tokens_all}")
    
    # 搜索引擎模式
    tokens_search = list(jieba.cut_for_search(text))
    print(f"Jieba tokens (搜索引擎模式): {tokens_search}")
    print("-" * 50)

def main():
    # Example 1: Exact match
    print("Example 1: Exact match")
    reference1 = "The quick brown fox jumps over the lazy dog"
    candidate1 = "The quick brown fox jumps over the lazy dog"
    print_rouge_scores(reference1, candidate1)

    # Example 2: Partial match with different words
    print("Example 2: Partial match with different words")
    reference2 = "The quick brown fox jumps over the lazy dog"
    candidate2 = "The fast brown fox leaps over the sleepy dog"
    print_rouge_scores(reference2, candidate2)

    # Example 3: Different length but similar content
    print("Example 3: Different length but similar content")
    reference3 = "The quick brown fox jumps over the lazy dog"
    candidate3 = "A quick brown fox jumps"
    print_rouge_scores(reference3, candidate3)

    # Example 4: Completely different
    print("Example 4: Completely different")
    reference4 = "The quick brown fox jumps over the lazy dog"
    candidate4 = "The sun is shining brightly in the sky"
    print_rouge_scores(reference4, candidate4)

    # Example 5: Chinese text
    print("Example 5: Chinese text")
    reference5 = "今天天气很好，阳光明媚"
    candidate5 = "今天天气不错，阳光灿烂"
    print_rouge_scores(reference5, candidate5)

    # 中文分词例子
    print("Example 1: 基本中文分词")
    text1 = "今天天气很好，阳光明媚"
    print_jieba_tokenization(text1)

    print("Example 2: 包含专有名词")
    text2 = "我在北京大学读书"
    print_jieba_tokenization(text2)

    print("Example 3: 包含数字和标点")
    text3 = "2024年3月15日，天气晴朗，温度25℃"
    print_jieba_tokenization(text3)

    print("Example 4: 长句子")
    text4 = "人工智能是计算机科学的一个分支。它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器"
    print_jieba_tokenization(text4)

if __name__ == "__main__":
    main() 