import torch.multiprocessing as mp
import os
import torch
import argparse

from src.trainer.train_chroma_lora import train_chroma

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_path", type=str, help="Path to training config JSON file"
    )
    args = parser.parse_args()

    world_size = torch.cuda.device_count()

    mp.spawn(
        train_chroma,
        args=(world_size, False, args.config_path),
        nprocs=world_size,
        join=True,
    )
