import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalContrastiveLoss(nn.Module):
    """
    Implements a joint InfoNCE contrastive loss function across text-video modalities.
    """
    def __init__(self, temperature: float = 0.07):
        super(MultimodalContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, projected_video: torch.Tensor, projected_text: torch.Tensor) -> torch.Tensor:
        # L2 Normalize vectors to compute proper cosine similarities
        video_features = F.normalize(projected_video, dim=-1)
        text_features = F.normalize(projected_text, dim=-1)

        # Pairwise dot products generating a [Batch, Batch] similarity topology
        logits = torch.matmul(video_features, text_features.T) / self.temperature

        # Labels target the true diagonal matches
        batch_size = projected_video.size(0)
        labels = torch.arange(batch_size, device=projected_video.device)

        # Bidirectional cross-entropy formulation (Video-to-Text + Text-to-Video)
        loss_v2t = F.cross_entropy(logits, labels)
        loss_t2v = F.cross_entropy(logits.T, labels)

        return (loss_v2t + loss_t2v) / 2.0