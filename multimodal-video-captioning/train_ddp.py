import os
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler

from model.transformer import MultimodalTransformer
from utils.data_loader import VideoCaptioningDataset, prepare_distributed_dataloader
from utils.loss import MultimodalContrastiveLoss

def train_ddp_node(rank: int, world_size: int):
    """
    Execution engine spawned across individual isolated GPU worker threads under DDP framework.
    """
    # 1. Spawn and bind distributed process parameters
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2. Pipeline generation
    dataset = VideoCaptioningDataset(num_samples=400)
    dataloader = prepare_distributed_dataloader(dataset, batch_size=16, rank=rank, world_size=world_size)

    # 3. Model assembly & distributed wrap
    model = MultimodalTransformer().to(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # 4. Criteria, Optimizers and AMP Scalers
    contrastive_criterion = MultimodalContrastiveLoss().to(rank)
    generation_criterion = torch.nn.MSELoss()  # Proxy placeholder optimization target
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    model.train()
    for epoch in range(5):
        dataloader.sampler.set_epoch(epoch)
        for video_feats, text_embeds in dataloader:
            video_feats = video_feats.to(rank, non_blocking=True)
            text_embeds = text_embeds.to(rank, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast Forward sequence under FP16 Mixed Precision Execution
            with autocast():
                decoded_out, proj_video, proj_text = model(video_feats, text_embeds)
                
                loss_contrastive = contrastive_criterion(proj_video, proj_text)
                loss_generation = generation_criterion(decoded_out, text_embeds)
                total_loss = loss_generation + loss_contrastive

            # Scale and execute backward gradients
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if rank == 0:
            print(f"Epoch {epoch:02d} | Aggregated Rank-0 Training Loss: {total_loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"Detected {world_size} CUDA devices. DDP multi-process requires at least 2 target GPUs.")
    else:
        print(f"Launching distributed training pipeline across {world_size} local GPUs...")
        torch.multiprocessing.spawn(train_ddp_node, args=(world_size,), nprocs=world_size, join=True)