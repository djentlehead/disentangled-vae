import torch
import torch.nn as nn

class CVAE(nn.Module):
    def __init__(self, vocab_size: int, style_vocab_size: int, embedding_dim: int, hidden_dim: int, latent_dim: int, style_embedding_dim: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.style_embedding = nn.Embedding(style_vocab_size, style_embedding_dim)
        
        self.encoder_lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        

        self.decoder_lstm = nn.LSTM(latent_dim + style_embedding_dim, hidden_dim, batch_first=True)
        self.decoder_fc = nn.Linear(hidden_dim, vocab_size)

    def encode(self, x):

        embedded = self.token_embedding(x)
        _, (hidden, _) = self.encoder_lstm(embedded)
        hidden = hidden.squeeze(0)
        

        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, style, seq_len):
        style_vec = self.style_embedding(style)
        
        z_styled = torch.cat([z, style_vec], dim=1)

        repeated_z = z_styled.unsqueeze(1).repeat(1, seq_len, 1)
        
        output, _ = self.decoder_lstm(repeated_z)
        logits = self.decoder_fc(output)
        return logits

    def forward(self, x, style):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        seq_len = x.size(1) 
        logits = self.decode(z, style, seq_len)
        return logits, mu, log_var
    