import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.optim as optim
import json
import re

torch.manual_seed(2612)

# Define the Embedder class
class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)

# Define the Positional Encoding class
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

# Custom tokenizer to better handle Python code
def custom_tokenizer(code):
    tokens = re.findall(r'\w+|[^\w\s]', code, re.UNICODE)
    return tokens

# Define the Layer Normalization class
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

# Define the Multihead Attention class
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

# Padding sequences dynamically
def pad_sequences(sequences, pad_value=0):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_len - len(seq)) for seq in sequences]

# Create a causal mask for attention
def create_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == "__main__":
    epochs = 10  # Increase number of epochs
    learning_rate = 0.00001

    # Load the data (use your dataset path)
    with open(r'D:\Manoj\Projects\Python\natural_lang\data[1].json', 'r') as f:
        data = json.load(f)

    # Tokenize and map the data
    encoded_tokens = set()
    for entry in data:
        code = entry.get('code', '')
        for token in custom_tokenizer(code):
            encoded_tokens.add(token)
    encoded_tokens = sorted(encoded_tokens)

    # Create mappings
    token_map = {token: idx for idx, token in enumerate(encoded_tokens)}
    index_to_token = {idx: token for token, idx in token_map.items()}

    # Dynamic vocab size
    vocab_size = len(token_map)

    input_sequences = []
    target_sequences = []
    for entry in data:
        code_tokens = custom_tokenizer(entry.get('code', ''))
        input_tokens = [token_map[token] for token in code_tokens[:-1] if token in token_map]
        target_tokens = [token_map[token] for token in code_tokens[1:] if token in token_map]
        input_sequences.append(input_tokens)
        target_sequences.append(target_tokens)

    # Padding dynamically
    input_sequences = pad_sequences(input_sequences)
    target_sequences = pad_sequences(target_sequences)

    input_sequences = torch.LongTensor(input_sequences)
    target_sequences = torch.LongTensor(target_sequences)

    # Define model components
    embedder = Embedder(vocab_size, embedding_dim=1024)
    positional_encoding = Positional(1024, max_seq_len=input_sequences.size(1))
    multihead = Multihead(1024, nhead=8)
    linear_layer = nn.Linear(1024, vocab_size)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(list(embedder.parameters()) + list(multihead.parameters()) + list(linear_layer.parameters()), lr=learning_rate)

    # Causal mask creation
    mask = create_causal_mask(input_sequences.size(1))

    # Training loop with validation
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        emb_output = embedder(input_sequences)
        embed_output = positional_encoding(emb_output)
        multihead_output = multihead(embed_output, mask=mask)

        logits = linear_layer(multihead_output)
        output = logits.view(-1, vocab_size)
        loss = criterion(output, target_sequences.view(-1))

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    # Decoding with clamping
    predicted_tokens = torch.argmax(logits, dim=-1)
    predicted_tokens = torch.clamp(predicted_tokens, min=0, max=vocab_size-1)

    # Decoding into readable code
    for pred_seq in predicted_tokens:
        decoded_tokens = [index_to_token[idx.item()] for idx in pred_seq if idx.item() != 0]
        decoded_code = ''.join(decoded_tokens)
        print("Decoded Code:\n", decoded_code)
    # Example of testing the model with specific tokens
    test_code = "def bubble_sort(arr):"  # Example code to test
    test_tokens = custom_tokenizer(test_code)
    test_input = [token_map[token] for token in test_tokens if token in token_map]

    # Pad test input
    test_input = pad_sequences([test_input])
    test_input_tensor = torch.LongTensor(test_input)

    # Forward pass
    with torch.no_grad():
        test_emb_output = embedder(test_input_tensor)
        test_pos_output = positional_encoding(test_emb_output)
        test_multihead_output = multihead(test_pos_output, mask=create_causal_mask(test_input_tensor.size(1)))
        test_logits = linear_layer(test_multihead_output)

    # Get predicted tokens
    predicted_tokens = torch.argmax(test_logits, dim=-1)

    # Decode the output
    # Improved decoding with additional checks
    decoded_tokens = []
    for pred_seq in predicted_tokens:
        valid_tokens = [idx.item() for idx in pred_seq if idx.item() not in [0]]  # Filter out padding
        decoded_code = ''.join(index_to_token[idx] for idx in valid_tokens)
        decoded_tokens.append(decoded_code)

    # Print decoded code snippets
    for code in decoded_tokens:
        print("Decoded Code:\n", code)
    # Print parameters of each component
    print("Embedder parameters:")
    for param in embedder.parameters():
        print(param)

    print("\nPositional Encoding parameters:")
    for param in positional_encoding.parameters():
        print(param)

    print("\nMultihead Attention parameters:")
    for param in multihead.parameters():
        print(param)

    print("\nLinear layer parameters:")
    for param in linear_layer.parameters():
        print(param)