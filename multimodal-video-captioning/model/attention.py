import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Engineers a localized spatio-temporal attention layer to align dynamic 
    visual feature mappings with tokenized linguistic embeddings.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, text_embeddings: torch.Tensor, video_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_embeddings (Tensor): [Batch, Text_Seq_Len, Embed_Dim] (Queries)
            video_features (Tensor):  [Batch, Video_Frames * Spatial, Embed_Dim] (Keys/Values)
        Returns:
            Tensor: Aligned multimodal representation [Batch, Text_Seq_Len, Embed_Dim]
        """
        # Query = Text, Key/Value = Video features
        attn_output, _ = self.cross_attention(
            query=text_embeddings, 
            key=video_features, 
            value=video_features
        )
        x = self.layer_norm(text_embeddings + attn_output)
        
        # Feed-Forward Network
        ffn_output = self.ffn(x)
        output = self.ffn_norm(x + ffn_output)
        return output