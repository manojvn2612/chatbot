import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import torch.optim as optim
import json

torch.manual_seed(2612)

class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # Initialize the embedding layer

    def forward(self, x):
        return self.embeddings(x)

# class Positional(nn.Module):
#     def __init__(self, embedding_dim, max_seq_len):
#         super(Positional, self).__init__()
#         self.embedding_dim = embedding_dim
#         pe = torch.zeros(max_seq_len, embedding_dim)
#         for pos in range(max_seq_len):
#             for i in range(0, embedding_dim, 2):
#                 pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
#                 pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/embedding_dim)))
#         pe = pe.unsqueeze(0)  # Add a batch dimension
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x * math.sqrt(self.embedding_dim)  # Scale the input
#         seq_len = x.size(1)
#         x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
#         return x

class Positional(nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(Positional, self).__init__()
        self.embedding_dim = embedding_dim
        # Adjust max_seq_len to the length of input tokens
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embedding_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / embedding_dim)))
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Scale the input
        x = x * math.sqrt(self.embedding_dim)  
        # Make sure that the positional encoding matches the input size
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x

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

if __name__ == "__main__":
    epochs = 100
    learning_rate = 0.00001

    #parameters to adjust my model parameters
    vocab_size = 1000
    embedding_dim = 1024
    sequence_length = 150
    batch_size = 2

    '''# Example input and target sequences
    input_tokens = torch.LongTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    target_tokens = torch.LongTensor([[2, 3, 4, 0], [6, 7, 8, 0]])

    # Check if token values are out of range trial

    assert input_tokens.max().item() < vocab_size, "Input tokens contain values out of vocabulary range!"
    assert target_tokens.max().item() < vocab_size, "Target tokens contain values out of vocabulary range!"
'''

    with open('D:\Manoj\Projects\Python\encoded_dataset.json', 'r') as f:
        data = json.load(f)
    for entry in data:
        input_tokens = torch.LongTensor(entry['label_encoded_tokens'][:-1])  # Input tokens (all except last token)
        target_tokens = torch.LongTensor(entry['label_encoded_tokens'][1:])  # Target tokens (all except first token)

        input_tokens = input_tokens.unsqueeze(0)
        target_tokens = target_tokens.unsqueeze(0)

        # Check if token values are out of range
        assert input_tokens.max().item() < vocab_size, "Input tokens contain values out of vocabulary range!"
        assert target_tokens.max().item() < vocab_size,"Target tokens contain values out of vocabulary range!"

    # Define the model components
    embedder = Embedder(vocab_size, embedding_dim)  # Embedder: vocab_size = 10, embedding_dim = 8
    positional_encoding = Positional(embedding_dim, max_seq_len=sequence_length)
    multihead = Multihead(embedding_dim)  # Multihead Attention: d_model = 8
    linear_layer = nn.Linear(embedding_dim, vocab_size)
    
    criterion = nn.CrossEntropyLoss()  # Loss function (cross-entropy)
    optimizer = optim.Adam(list(embedder.parameters()) + list(multihead.parameters()) + list(linear_layer.parameters()), lr=learning_rate)  # Optimizer

    # Create an attention mask (causal mask for future tokens)
    size_tokens = input_tokens.size(1)
    mask = torch.tril(torch.ones(size_tokens, size_tokens))  # Lower-triangular causal mask
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    input_tokens = torch.LongTensor(entry['label_encoded_tokens'][:-1])  # Input tokens (all except last token)
    target_tokens = torch.LongTensor(entry['label_encoded_tokens'][1:])  # Target tokens (all except first token)

    # Add batch dimension by unsqueezing

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        optimizer.zero_grad()
        
        emb_output = embedder(input_tokens)  # Get embedding for input tokens
        embed_output = positional_encoding(emb_output)  # Apply positional encoding
        multihead_output = multihead(embed_output, mask=mask)  # Apply multihead attention
        
        # Pass through the linear layer
        logits = linear_layer(multihead_output)
        
        # Reshape output to be suitable for the loss function (batch * seq_len, vocab_size)
        output = logits.view(-1, vocab_size)
        
        # Compute loss (output and target should be flattened for CrossEntropyLoss)
        loss = criterion(output, target_tokens.view(-1))
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Get the predicted next word
        predicted_tokens = torch.argmax(logits, dim=-1)  # Get predicted tokens
        
        # Print the loss for each epoch
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        
        # Print the predicted tokens for each batch
        print("Predicted tokens:")
        print(predicted_tokens)
