import sentencepiece as spm
import json
from sklearn.preprocessing import LabelEncoder

# Load the dataset
with open('natural_lang/data.json', 'r') as f:
    data = json.load(f)

# Collect all the code snippets as training data
codes = [entry['code'] for entry in data]
print("Code Snippets:", codes)

# Write the code snippets to a text file (required by SentencePiece)
with open('code_corpus.txt', 'w') as f:
    for code in codes:
        f.write(code + "\n")

# Train the BPE tokenizer (with vocab size 50, can be adjusted)
spm.SentencePieceTrainer.train(input='code_corpus.txt', model_prefix='bpe_model', vocab_size=135)

# Load the trained BPE model
sp = spm.SentencePieceProcessor(model_file='bpe_model.model')

# Load the vocab from the SentencePiece model
with open('bpe_model.vocab', 'r', encoding='utf-8') as f:
    vocab = [line.split('\t')[0] for line in f.readlines()]

print("Vocab Tokens:", vocab)

# Initialize and fit LabelEncoder with the vocabulary
label_encoder = LabelEncoder()
label_encoder.fit(vocab)

# Encode the entire dataset
# Handle unseen tokens by checking if they are in the label encoder's classes
import json
import numpy as np

# Convert any numpy types (e.g. int32) to native Python types
encoded_db = []

for entry in data:
    code_snippet = entry['code']
    # Tokenize the code snippet using SentencePiece
    encoded_code = sp.encode(code_snippet, out_type=str)
    
    # Initialize an empty list to store label encoded tokens
    label_encoded_code = []
    
    for token in encoded_code:
        if token in label_encoder.classes_:
            # Convert np.int32 to Python int
            label_encoded_code.append(int(label_encoder.transform([token])[0]))
        else:
            label_encoded_code.append(int(label_encoder.transform(['<unk>'])[0]))

    # Add the encoded code to the new dataset
    encoded_db.append({
        'original_code': code_snippet,
        'encoded_tokens': encoded_code,
        'label_encoded_tokens': label_encoded_code  # Ensure Python native types
    })

# Now you can serialize the data to JSON
with open('encoded_dataset.json', 'w') as f:
    json.dump(encoded_db, f, indent=4)
