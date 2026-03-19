"""
DeepSentinel — Deep Learning Model
Three branches: BERT, Bi-LSTM, GNN — fused via attention-weighted concat.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torch_geometric.nn import GCNConv, global_mean_pool

NUM_CLASSES = 5   # misinformation, mental_health, cyberbullying, political_unrest, normal

# ─── Branch 1: BERT Encoder ───────────────────────────────────────────────────

class BERTBranch(nn.Module):
    """Fine-tuneable BERT encoder. Outputs [batch, 768]."""

    def __init__(self, model_name="bert-base-uncased", freeze_layers=8):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)

        # Freeze lower layers to speed up training
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # [CLS] token pooled output → shape [batch, 768]
        return self.dropout(out.pooler_output)


# ─── Branch 2: Bi-LSTM Temporal Encoder ──────────────────────────────────────

class BiLSTMBranch(nn.Module):
    """
    Bi-LSTM over word embeddings.
    Captures temporal escalation patterns across token sequence.
    Outputs [batch, 256].
    """

    def __init__(self, vocab_size=30522, embed_dim=128, hidden=128, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.3)
        # Attention over time steps
        self.attn = nn.Linear(hidden * 2, 1)

    def forward(self, input_ids, attention_mask):
        x = self.embedding(input_ids)                          # [B, T, E]
        x, _ = self.lstm(x)                                   # [B, T, 2H]

        # Masked attention pooling (ignore padding positions)
        scores = self.attn(x).squeeze(-1)                      # [B, T]
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)       # [B, T, 1]
        pooled = (x * weights).sum(dim=1)                      # [B, 2H]
        return self.dropout(pooled)


# ─── Branch 3: Graph Neural Network (Hashtag Graph) ──────────────────────────

class GNNBranch(nn.Module):
    """
    2-layer GCN over the hashtag co-occurrence graph.
    Outputs [batch, 64] — one pooled graph embedding per sample.
    """

    def __init__(self, in_features=16, hidden=64, out=64):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, out)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)                      # [B, out]


# ─── Fusion Layer ─────────────────────────────────────────────────────────────

class AttentionFusion(nn.Module):
    """
    Learns soft weights over the three branch outputs
    before concatenation → shared representation.
    """

    def __init__(self, bert_dim=768, lstm_dim=256, gnn_dim=64):
        super().__init__()
        total = bert_dim + lstm_dim + gnn_dim
        self.gate = nn.Linear(total, 3)

    def forward(self, bert_out, lstm_out, gnn_out):
        concat = torch.cat([bert_out, lstm_out, gnn_out], dim=-1)   # [B, total]
        weights = F.softmax(self.gate(concat), dim=-1)               # [B, 3]

        # Weighted sum (broadcast over feature dims)
        fused = (
            weights[:, 0:1] * bert_out +
            weights[:, 1:2] * lstm_out +
            weights[:, 2:3] * gnn_out
        )
        return fused, weights


# ─── Full DeepSentinel Model ──────────────────────────────────────────────────

class DeepSentinel(nn.Module):
    """
    End-to-end crisis detection model.
    Input:  BERT token ids + Bi-LSTM token ids + PyG graph batch
    Output: logits [batch, NUM_CLASSES]
    """

    def __init__(self):
        super().__init__()
        self.bert_branch = BERTBranch()
        self.lstm_branch = BiLSTMBranch()
        self.gnn_branch  = GNNBranch()
        self.fusion      = AttentionFusion()

        # Classification head
        # bert(768) + lstm(256) + gnn(64) → 1088 fused dim after gating
        # Use the fused (max-dim) representation from BERT as the dominant signal
        self.classifier  = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, bert_ids, bert_mask, lstm_ids, lstm_mask,
                gnn_x=None, gnn_edge_index=None, gnn_batch=None):

        bert_out = self.bert_branch(bert_ids, bert_mask)         # [B, 768]
        lstm_out = self.lstm_branch(lstm_ids, lstm_mask)         # [B, 256]

        # GNN is optional (not every post has hashtags)
        if gnn_x is not None and gnn_edge_index is not None:
            gnn_out = self.gnn_branch(gnn_x, gnn_edge_index, gnn_batch)  # [B, 64]
        else:
            gnn_out = torch.zeros(bert_out.size(0), 64, device=bert_out.device)

        fused, branch_weights = self.fusion(bert_out, lstm_out, gnn_out)
        logits = self.classifier(bert_out + fused[:, :768])      # residual

        return logits, branch_weights


# ─── Loss & Metrics ───────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss to handle class imbalance in crisis datasets."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== DeepSentinel Model Test ===")
    model = DeepSentinel()
    total_params = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters:    {total_params:,}")
    print(f"Trainable parameters:{trainable:,}")

    # Dummy forward pass
    B, T = 4, 128
    bert_ids  = torch.randint(0, 30522, (B, T))
    bert_mask = torch.ones(B, T, dtype=torch.long)
    lstm_ids  = torch.randint(0, 30522, (B, T))
    lstm_mask = torch.ones(B, T, dtype=torch.long)

    logits, weights = model(bert_ids, bert_mask, lstm_ids, lstm_mask)
    print(f"Output logits shape: {logits.shape}")          # [4, 5]
    print(f"Branch weights:      {weights[0].detach()}")
    print("Forward pass OK!")
