import torch
import torch.nn as nn
import torch.nn.functional as F

class RelativePositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(RelativePositionEmbedding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.relative_position_embedding = nn.Embedding(2 * max_len - 1, d_model)

    def forward(self, seq_len, device):
        range_vec = torch.arange(seq_len, device=device)
        distance_mat = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        distance_mat_clipped = distance_mat.clamp(-self.max_len + 1, self.max_len - 1) + self.max_len - 1
        return self.relative_position_embedding(distance_mat_clipped)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, rel_pos_emb, mask=None):
        batch_size, seq_len, _ = q.size()
        
        q = self.query(q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.key(k).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        rel_pos_emb = rel_pos_emb.view(seq_len, seq_len, self.n_heads, self.d_k).permute(2, 0, 1, 3)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        scores += torch.einsum('bhld,lrd->bhlr', q, rel_pos_emb)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class TransformerXLBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerXLBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mem, rel_pos_emb, mask=None):
        x = torch.cat([mem, x], dim=1)
        attn_output = self.attention(x, x, x, rel_pos_emb, mask)
        x = self.layer_norm1(attn_output + x[:, -attn_output.size(1):])
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(ff_output + x)
        return x, mem[:, -attn_output.size(1):]


class TransformerXL(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, dropout=0.1, mem_len=128, max_len=512):
        super(TransformerXL, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.mem_len = mem_len
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = RelativePositionEmbedding(d_model, max_len)
        
        self.blocks = nn.ModuleList([
            TransformerXLBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.linear = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mems=None):
        if mems is None:
            mems = [torch.zeros(x.size(0), 0, self.d_model, device=x.device) for _ in range(self.n_layers)]
        
        emb = self.embedding(x)
        rel_pos_emb = self.positional_embedding(x.size(1), x.device)
        
        new_mems = []
        for i, block in enumerate(self.blocks):
            mem = mems[i]
            emb, new_mem = block(emb, mem, rel_pos_emb)
            new_mems.append(new_mem)
        
        output = self.linear(emb)
        return output, new_mems

