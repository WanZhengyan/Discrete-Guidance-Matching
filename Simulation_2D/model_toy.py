import torch
from torch import nn, Tensor

# Pretrained model for toy datasets
class ToyMLP(nn.Module):
    def __init__(
        self, vocab_size: int = 32, hidden_dim=256, length=2, time_dim=1):
        super().__init__()
        self.length = length
        self.time_dim = time_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.time_embedding = nn.Linear(1, 1)
        self.token_embedding = torch.nn.Embedding(self.vocab_size, hidden_dim)

        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * length + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.vocab_size * length)
        )
    def forward(self, x, t):
        x = self.token_embedding(x)
        B, N, d = x.shape # (B: batch size, N: sequence length, d: embedding dimension)
        x = x.reshape(B, N * d)
        if self.time_dim == 1: # if time_dim == 0 then no time embedding
            t = self.time_embedding(t.unsqueeze(-1)) # shape: (B, time_dim)
            h = torch.cat([x, t], dim=1)
        else:
            h = x
        h = self.block(h)
        h = h.reshape(B, N, self.vocab_size)
        return h # output shape: (B, N, vocab_size); each element is a logit for the vocabulary
    
# Guidance model for toy datasets
class RateGuidance(nn.Module): # Output \E(p(y=1|x_1,t)|X_t=z), where z is different with x_t only one token
    def __init__(
        self, vocab_size: int = 32, hidden_dim: int = 256, length: int = 2, time_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.length = length
        self.time_dim = time_dim

        self.time_embedding = nn.Linear(1, 1)
        self.token_embedding = torch.nn.Embedding(self.vocab_size, hidden_dim)


        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * (self.length) + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_t, t): # (x_t, t) -> (batch_size, length, vocav_size)
        x_t = self.token_embedding(x_t)
        B, N, d = x_t.shape # (B: batch size, N: sequence length, d: embedding dimension)
        x_t = x_t.reshape(B, N * d)

        if self.time_dim == 1: # if time_dim == 0 then no time embedding
            t = self.time_embedding(t.unsqueeze(-1)) # shape: (B, time_dim)
            h = torch.cat([x_t, t], dim=1)
        else:
            h = x_t
        h = self.block(h)
        return h
    
class PosteriorGuidance(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 256, length: int = 2, time_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.length = length
        self.time_dim = time_dim

        self.time_embedding = nn.Linear(1, 1)
        self.token_embedding = torch.nn.Embedding(self.vocab_size, hidden_dim)

        self.block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * (self.length) + self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.vocab_size * length)
        )

    def forward(self, x_t, t): # (x_t, t) -> (batch_size, length, vocav_size)
        x_t = self.token_embedding(x_t)
        B, N, d = x_t.shape # (B: batch size, N: sequence length, d: embedding dimension)
        x_t = x_t.reshape(B, N * d)

        if self.time_dim == 1: # if time_dim == 0 then no time embedding
            t = self.time_embedding(t.unsqueeze(-1)) # shape: (B, time_dim)
            h = torch.cat([x_t, t], dim=1)
        else:
            h = x_t
        h = self.block(h)
        h = h.reshape(B, N, self.vocab_size)
        return h # output shape: (B, N, vocab_size); each element is a logit for the vocabulary
    

# density ratio model as an oracle
class DensityRatio(nn.Module):
    def __init__(
        self, vocab_size: int = 128, hidden_dim: int = 128, length: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.length = length

        # projecting each token to a (hidden_dim)-dimensional space (hidden_dim * self.length dimensions in total)
        self.embedding = torch.nn.Embedding(self.vocab_size, hidden_dim)

        self.main = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim * self.length, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor: # (batch, length)
        # x shape: (batch, length)
        x = self.embedding(x)  # (batch, length, hidden_dim)
        x = x.view(x.size(0), -1)  # (batch, length * hidden_dim)
        x = self.main(x)
        return x