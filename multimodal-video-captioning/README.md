# Multimodal Generative Video Captioning & Temporal Action Transformer

An end-to-end multi-GPU deep learning architecture designed to extract spatiotemporal representations from unsegmented video streams and align them with linguistic generative decoders using structured multi-modal cross-attention.

## Key Technical Implementations
* **Generative Architecture:** Bridges 3D spatiotemporal video feature mappings with a decoupled, autoregressive Generative Transformer Decoder.
* **Cross-Attention Fusion:** Implements localized sequence-to-sequence cross-attention layers targeting fine-grained temporal and structural context matching.
* **Contrastive Optimization:** Drives zero-shot semantic matching via a symmetric multi-modal InfoNCE contrastive loss module.
* **Distributed Scaling:** Features full PyTorch Distributed Data Parallel (DDP) synchronization with FP16 Mixed Precision (AMP) to achieve linear multi-GPU efficiency.

## Quick Start
To launch the multi-GPU training pipeline across your system's available graphics cluster cards, run:
```bash
python train_ddp.py