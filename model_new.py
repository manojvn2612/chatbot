#importing modules

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.optim as optim
import re
import json

torch.manual_seed(2612) #using this manual seed to get same results and to avoid problems from different people running the code

# Embed class to convert numbers to embedding vectors
class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)

# Positional Encoding class to get insights for word in a sentence(code)
class Positional(nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(Positional, self).__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / embedding_dim)))
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

# Layer Normalization class to be passed to NN from self Attention
class LayerNormalization(nn.Module):
    def __init__(self, parameters, epsilon: float = 1e-6):
        super(LayerNormalization, self).__init__()
        self.epsilon = epsilon
        self.alpha = nn.Parameter(torch.ones(parameters))
        self.beta = nn.Parameter(torch.zeros(parameters))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.epsilon) + self.beta

# #crux of the model and provides which word should it emphasize on
class Multihead(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return torch.matmul(F.softmax(scores, dim=-1), v)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, self.nhead, 3 * self.d_k).permute(2, 0, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = self.attention(q, k, v)
        output = scores.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, d_model)
        return self.out(output)

# Custom tokenizer to better handle Python code splitting it around word
def custom_tokenizer(code):
    tokens = re.findall(r'\w+|[^\w\s]', code, re.UNICODE)
    return tokens

# Padding sequences dynamically to make them of same length
def pad_sequences(sequences, pad_value=0):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

# Create a causal mask for attention
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# model class
class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_seq_len):
        super(Model, self).__init__()
        self.embedding = Embedder(vocab_size, embedding_dim)
        self.positional_encoding = Positional(embedding_dim, max_seq_len)
        self.multihead_attention = Multihead(embedding_dim)
        self.layer_norm = LayerNormalization(embedding_dim)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)  # Linear layer to map to vocab size

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.multihead_attention(x)
        x = self.layer_norm(x)
        x = self.fc_out(x)  # Linear layer applied here
        return x

#trial loop
if __name__ == '__main__':

    # Load the data from the JSON file
    with open(r"D:\Manoj\Projects\Python\natural_lang\data.json", "r") as f:
        data = json.load(f)
    
    # Preprocess the data (tokenization)
    vocab = set()  # Vocab set
    tokenized_data = []

    # Add special tokens to the vocab
    special_tokens = ['<PAD>', '<EOS>', '<UNK>']
    vocab.update(special_tokens)

    for entry in data:
        tokens = custom_tokenizer(entry["code"])
        tokens.append('<EOS>')  # Append the EOS token
        tokenized_data.append(tokens)
        vocab.update(tokens)

    # Create vocab to index mapping
    vocab = list(vocab)
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Convert tokenized data to numerical data
    input_data = []
    for tokens in tokenized_data:
        indexed_tokens = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
        input_data.append(indexed_tokens)

    # Padding sequences to the same length
    pad_idx = word_to_idx['<PAD>']
    input_data = pad_sequences(input_data, pad_value=pad_idx)
    input_data = torch.tensor(input_data)

    print("Input data:", input_data)

    # Model hyperparameters
    embedding_dim = 512
    max_seq_len = input_data.size(1)
    num_epochs = 10
    batch_size = 2
    learning_rate = 0.001
    model = Model(vocab_size, embedding_dim, max_seq_len)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for i in range(0, len(input_data), batch_size):
            inputs = input_data[i:i+batch_size]
            targets = input_data[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1, vocab_size)
            targets = targets.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(input_data)}")

    # Evaluation 
    model.eval()
    test_data = input_data[:2]  # Using first two samples for testing
    with torch.no_grad():
        test_outputs = model(test_data)
        predicted_indices = torch.argmax(test_outputs, dim=-1)
    predicted_tokens = []
    for seq in predicted_indices:
        tokens = []
        for idx in seq:
            token = idx_to_word[idx.item()]
            if token == '<EOS>':
                break
            if token != '<PAD>':
                tokens.append(token)
        predicted_tokens.append(tokens)

    original_tokens = []
    for seq in test_data:
        tokens = []
        for idx in seq:
            token = idx_to_word[idx.item()]
            if token == '<EOS>':
                break
            if token != '<PAD>':
                tokens.append(token)
        original_tokens.append(tokens)

    for i in range(len(test_data)):
        print(f"\n--- Sample {i+1} ---")
        print("Original Tokens:")
        print(" ".join(original_tokens[i]))
        print("Predicted Tokens:")
        print(" ".join(predicted_tokens[i]))

    # save model
    torch.save(model.state_dict(), 'code_gen.pth')

    model = Model(vocab_size, embedding_dim, max_seq_len)

    # Load the saved model
    model.load_state_dict(torch.load('code_gen.pth'))
    model.eval()
