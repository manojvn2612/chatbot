import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(2612)

#Embedding layer (Class)
class Embedder(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # Initialize the embedding layer
    
    def forward(self, x):
        return self.embeddings(x)

#giving Positional encoding to the model
class Positional(nn.Module):
    def __init__(self, embedding_dim, max_seq_len):
        super(Positional, self).__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_seq_len, embedding_dim)
        for pos in range(max_seq_len):
            for i in range(0, embedding_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/embedding_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i)/embedding_dim)))
        pe = pe.unsqueeze(0)  # Add a batch dimension
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_dim)  # Scale the input
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]  # Use the positional encoding
        return x

#layer normalization before passing it through the layer of feed forward network
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

#crux of transformers attention self (multi head) attention mechanism which get the meaning of the input parameters
class Multihead(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == float('-inf'), float('-inf'))
        return torch.matmul(F.softmax(scores, dim=-1), v)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, self.nhead, 3 * self.d_k).permute(2, 0, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = self.attention(q, k, v, mask)
        output = scores.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_len, d_model)
        return self.dropout(self.out(output))

#main class which will be used to train the model
class TransformerModel(nn.Module):#decoder model
    def __init__(self, vocab_size, d_model, max_seq_len, nhead):
        super(TransformerModel, self).__init__()
        self.embedder = Embedder(vocab_size, d_model)
        self.positional = Positional(d_model, max_seq_len)
        self.multihead = Multihead(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)  # Final layer to map to vocab size

    def forward(self, x, mask=None):
        x = self.embedder(x)
        x = self.positional(x)
        x = self.multihead(x, mask)
        x = self.norm1(x)  # Apply layer normalization after multi-head attention
        x = self.ff(x)
        x = self.norm2(x)  # Apply layer normalization after feed-forward
        return self.fc_out(x)

#dataset pasisng
class MyDataset(Dataset):
    def __init__(self, input_tokens, target_tokens):
        self.input_tokens = input_tokens
        self.target_tokens = target_tokens

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return self.input_tokens[idx], self.target_tokens[idx]

if __name__ == "__main__":
    #all the parameters
    vocab_size = 10
    embedding_dim = 512
    max_seq_len = 4
    nhead = 8
    epochs = 5
    lr = 0.001
    model = TransformerModel(vocab_size, embedding_dim, max_seq_len, nhead)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    input_tokens_data = torch.LongTensor([[1, 2, 3, 4], [1, 2, 3, 5]])  # Example input data
    target_tokens_data = torch.LongTensor([[2, 3, 4, 0], [2, 3, 5, 0]])  # Example target data

    # Create dataset and dataloader
    dataset = MyDataset(input_tokens_data, target_tokens_data)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    seq_len = input_tokens_data.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_input, batch_target in dataloader:
            optimizer.zero_grad()

            # Create a causal mask for each batch
            seq_len = batch_input.size(1)
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

            # Forward pass
            outputs = model(batch_input, mask)

            # Reshape for the loss calculation
            outputs = outputs.view(-1, vocab_size)
            batch_target = batch_target.view(-1)

            # Compute loss
            loss = criterion(outputs, batch_target)
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

            # Convert logits to predicted token indices
            _, predicted_indices = torch.max(outputs, dim=-1)
            correct_predictions += (predicted_indices == batch_target).sum().item()
            total_predictions += batch_target.numel()

        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions * 100

        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
