'''from datasets import load_dataset
import json
#downloading dataset and loading it
dataset = load_dataset("code_search_net","python", split="train", trust_remote_code=True)

#creating a json file
dataset = [dict(sample) for sample in dataset]
with open('dataset.json', 'w') as f:
    json.dump(dataset, f,indent=4)'''

from encoder import BPE
import json
with open("data.json", 'r') as f:
    data = json.load(f)
max_code_length = max(len(entry["code"]) for entry in data)

print(f"The maximum length of the code tag is: {max_code_length}")
combined_text = " ".join([entry['code'] for entry in data])
# print(combined_text)
bpe = BPE(combined_text)
bpe.fit(1000)
# print(len(bpe.vocab))
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
vocab_list = list(bpe.vocab)
le.fit(vocab_list)
encoded_labels = le.transform(vocab_list)
print("encoded labels:",encoded_labels)
known_data = ["def", "arr", "len", "n", "unknown_word"]
encoded_data = le.transform([word for word in known_data if word in vocab_list])
print(encoded_data) #checking with some unknown data
encoded_value = 773 #trial to check encoding
try:
    word = le.inverse_transform([encoded_value])[0]
    print(f"Encoded value {encoded_value} corresponds to the word '{word}' in the vocabulary.")
except ValueError:
    print(f"Encoded value {encoded_value} is not in the vocabulary.")

#save my vocab
vocab_dict = {int(label): word for word, label in zip(vocab_list, encoded_labels)}
with open("vocab_dict.json", 'w') as f:
    json.dump(vocab_dict, f, indent=4)
encoded_data_str = " ".join(map(str, encoded_labels))
with open("EncodedCode.txt", 'w') as f:
    f.write(encoded_data_str)