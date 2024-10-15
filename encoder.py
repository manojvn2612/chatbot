"""from collections import defaultdict, Counter

class BPE:
    def __init__(self, text):
        
        self.words = list(text)  
        self.vocab = set(self.words)  
    
    def get_pair(self):
        # Count frequency of adjacent pairs
        pair_freq = defaultdict(int)
        for i in range(len(self.words) - 1):
            pair = (self.words[i], self.words[i + 1])
            pair_freq[pair] += 1
        # Return the most frequent pair
        return max(pair_freq, key=pair_freq.get) if pair_freq else None
    
    def merge_pair(self, pair):
        # Merge the most frequent pair in the entire sequence
        new_words = []
        i = 0
        while i < len(self.words) - 1:
            if (self.words[i], self.words[i + 1]) == pair:
                new_words.append("".join(pair))  # Merge the pair into one token
                i += 2  # Skip the next token as it has been merged
            else:
                new_words.append(self.words[i])
                i += 1
        if i < len(self.words):
            new_words.append(self.words[-1])
        self.words = new_words
        self.vocab.add("".join(pair))  # Add the merged token to the vocabulary
    
    def fit(self, iterations):
        # Perform the merge operation for the specified number of iterations
        for _ in range(iterations):
            pair = self.get_pair()
            if not pair:
                break  
            self.merge_pair(pair)
        print("Final words:", self.words)
        print("\nVocabulary:", sorted(self.vocab))

if __name__ == "__main__":
    text = '''def add(a,b):\n    return a+b'''
    bpe = BPE(text)
    bpe.fit(100)
    print(len(bpe.vocab))"""

"""
from collections import defaultdict

class BPE:
    def __init__(self, text):
        # Tokenize the text into characters (or words, depending on use case)
        self.words = list(text)
        self.vocab = set(self.words)  # Initialize the vocabulary with all unique tokens
    
    def get_pair(self):
        # Count frequency of adjacent pairs
        pair_freq = defaultdict(int)
        for i in range(len(self.words) - 1):
            pair = (self.words[i], self.words[i + 1])
            pair_freq[pair] += 1
        # Return the most frequent pair
        return max(pair_freq, key=pair_freq.get) if pair_freq else None
    
    def merge_pair(self, pair):
        # Merge the most frequent pair in the entire sequence
        new_words = []
        i = 0
        while i < len(self.words) - 1:
            if (self.words[i], self.words[i + 1]) == pair:
                new_words.append("".join(pair))  # Merge the pair into one token
                i += 2  # Skip the next token as it has been merged
            else:
                new_words.append(self.words[i])
                i += 1
        if i < len(self.words):
            new_words.append(self.words[-1])
        self.words = new_words
        self.vocab.add("".join(pair))  # Add the merged token to the vocabulary
    
    def fit(self, iterations):
        # Perform the merge operation for the specified number of iterations
        for _ in range(iterations):
            pair = self.get_pair()
            if not pair:
                break  # Stop if no pairs are found
            self.merge_pair(pair)
        print("Final words:", self.words)
        print("\nVocabulary:", sorted(self.vocab))

if __name__ == "__main__":
    text = '''def add(a,b):\n    return a+b'''
    bpe = BPE(text)
    bpe.fit(100)
    print(len(bpe.vocab))
    """
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import json

class BPE:
    def __init__(self, text):
        self.words = text.split()  # Split text into words
        self.vocab = set(self.words)

    def get_pair(self):
        pair_freq = defaultdict(int)
        for i in range(len(self.words) - 1):
            pair = (self.words[i], self.words[i + 1])
            pair_freq[pair] += 1
        return max(pair_freq, key=pair_freq.get) if pair_freq else None
    
    def merge_pair(self, pair):
        new_words = []
        i = 0
        while i < len(self.words) - 1:
            if (self.words[i], self.words[i + 1]) == pair:
                new_words.append("".join(pair))
                i += 2
            else:
                new_words.append(self.words[i])
                i += 1
        if i < len(self.words):
            new_words.append(self.words[-1])
        self.words = new_words
        self.vocab.add("".join(pair))

    def fit(self, iterations):
        for _ in range(iterations):
            pair = self.get_pair()
            if not pair:
                break
            self.merge_pair(pair)
        print("Final words:", self.words)
        print("\nVocabulary:", sorted(self.vocab))

class Bigram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Bigram, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)  # Use embedding_dim

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # logits shape (B, T, C)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Example of using the classes:
if __name__ == "__main__":
    text = '''def add(a,b):\n    return a+b'''
    bpe = BPE(text)
    bpe.fit(100)

    # Initialize and train Bigram model
    # Assume you have encoded data ready for the Bigram model
    vocab_size = 848
    embedding_dim = 50
    model = Bigram(vocab_size,embedding_dim)
    input = torch.tensor(X_train, dtype=torch.long).unsqueeze(0)
    output = torch.tensor(X_test, dtype=torch.long).unsqueeze(0)
    logits, loss = model(input, output)
    print("Logits shape:", logits)
    print("Loss:", loss)

    # Start generation with the first token of X_train
    first_word_token = X_train[0]
    generated_sequence = model.generate(idx=torch.tensor([[first_word_token]], dtype=torch.long), max_new_tokens=100)
    print("Generated sequence:", generated_sequence)

    # Load vocabulary dictionary
    with open("vocab_dict.json", 'r') as f:
        vocab_dict = json.load(f)

    # Inverse the vocab_dict to decode
    inv_vocab_dict = {int(k): v for k, v in vocab_dict.items()}
    # Decode the generated sequence, handling unknown tokens
    decoded_sequence = []
    for token in generated_sequence[0].tolist():
        print(token)
        if token in inv_vocab_dict:
            decoded_sequence.append(inv_vocab_dict[token])
        else:
            decoded_sequence.append("[UNK]")  # Handle unseen token
    print("Decoded sequence:", decoded_sequence)
    # Encode the generated sequence back (optional, for verification)
    encoded_sequence = [inv_vocab_dict.get(word, -1) for word in decoded_sequence]  # -1 if word not in vocab_dict
    print("Encoded sequence:", encoded_sequence)

    # Continue with your training and generation logic...
