import json
from transformers import AutoTokenizer

# Load the Gemma tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Read and parse input file
with open("data/entities2.json", "r") as f:
    data_dict = json.load(f)

max_len = 50
new_dict = {}
all_words = []
for key, words in data_dict.items():
    new_dict[key] = []
    for word in words:
        # Tokenize the word
        tokens = tokenizer.encode(word)
        if len(tokens) == 2 and word not in all_words:
            new_dict[key].append(word)
            all_words.append(word)
            if len(new_dict[key]) >= max_len:
                break

with open("data/entities3.json", "w") as f:
    json.dump(new_dict, f)
