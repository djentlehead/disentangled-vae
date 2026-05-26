import torch.nn as nn

class StyleClassifier(nn.Module):

    def __init__(self, vocab_size: int, style_vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, style_vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        last_hidden = hidden[-1, :, :]
        logits = self.fc(last_hidden)
        return logits
    