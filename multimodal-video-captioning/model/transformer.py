import torch
import torch.nn as nn
from model.attention import CrossAttentionFusion

class MultimodalTransformer(nn.Module):
    """
    End-to-end multimodal network pairing video feature extractions 
    with a decoupled Generative Transformer decoder.
    """
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_layers: int = 3):
        super(MultimodalTransformer, self).__init__()
        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)
        
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=embed_dim * 4, 
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        
        # Projection layers for global cross-modal semantic alignment
        self.video_projection = nn.Linear(embed_dim, embed_dim)
        self.text_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, video_feats: torch.Tensor, text_embeds: torch.Tensor) -> tuple:
        # 1. Align cross-modal representations via cross-attention
        fused_context = self.fusion(text_embeds, video_feats)
        
        # 2. Autoregressive sequence generation decoding step
        decoded_output = self.transformer_decoder(tgt=text_embeds, memory=fused_context)
        
        # 3. Project to a shared metric space for InfoNCE contrastive evaluation
        proj_video = self.video_projection(video_feats.mean(dim=1))
        proj_text = self.text_projection(decoded_output.mean(dim=1))
        
        return decoded_output, proj_video, proj_text