import json
def read_jsonl_file(file_path):
    """
    Load lines of texts.

    Args:
        file_path (str): Path for lines of texts.

    Returns:
        (List[str]): List of texts.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line.strip())
            data.append(json_data)
    return data

data = read_jsonl_file('../finetuning/datasets/test.jsonl')

print(len(data))

for data in data:
    print(data['context'])