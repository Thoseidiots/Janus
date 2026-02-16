import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

VOCAB_SIZE = 256 + 3  # bytes + pad/bos/eos
CONTEXT_LENGTH = 512
EMBED_DIM = 256

class ByteTokenizer:
    def encode(self, text: str) -> List[int]:
        return list(text.encode('utf-8'))
    
    def decode(self, tokens: List[int]) -> str:
        return bytes([t for t in tokens if t < 256]).decode('utf-8', errors='replace')

class ByteLLM(nn.Module):
    """Transformer operating directly on UTF-8 bytes"""
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_emb = nn.Embedding(CONTEXT_LENGTH, EMBED_DIM)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, 
            nhead=4, 
            dim_feedforward=EMBED_DIM*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)
        
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.shape
        tok_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)
        x = self.norm(x)
        logits = self.head(x)
        
        if targets is None:
            return logits, None
            
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1))
        return logits, loss
        
    @torch.no_grad()
    def generate(self, prompt: str, max_new: int = 100, temperature: float = 0.8):
        tokenizer = ByteTokenizer()
        input_ids = [257] + tokenizer.encode(prompt)  # BOS + prompt
        idx = torch.tensor([input_ids], dtype=torch.long)
        
        for _ in range(max_new):
            idx_cond = idx[:, -CONTEXT_LENGTH:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)
            if next_byte.item() == 258:  # EOS
                break
            idx = torch.cat((idx, next_byte), dim=1)
            
        generated = idx[0].tolist()[len(input_ids):]
        return tokenizer.decode(generated)
