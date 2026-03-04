import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, DistilBertModel

from data.loaders import build_real_text_dataset, build_train_dataset
from training.discriminator import UntokenDiscriminator
from training.reconstructor import UntokenReconstructor
from training.scheduler import TrainConfig
from training.trainer import UntokenTrainer
from untoken.compressor import UntokenCompressor, UntokenCompressorConfig


def setup_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--phase", type=int, default=None, help="Run only this phase (1/2/3)")
    parser.add_argument("--resume", default=None, help="Checkpoint dir to resume from")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 2 batches per phase")
    args = parser.parse_args()

    rank, local_rank, world_size = setup_distributed()
    cfg = TrainConfig.from_yaml(args.config)
    seed_everything(cfg.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    vocab_size = tokenizer.vocab_size

    comp_cfg = UntokenCompressorConfig(target_ratio=cfg.target_compression_ratio)
    compressor = UntokenCompressor(comp_cfg)
    reconstructor = UntokenReconstructor(vocab_size)
    discriminator = UntokenDiscriminator(vocab_size)

    if not args.resume:
        _pretrained = DistilBertModel.from_pretrained("distilbert-base-uncased")
        compressor.encoder.load_state_dict(_pretrained.state_dict())
        del _pretrained

    if args.resume:
        from pathlib import Path
        ckpt = Path(args.resume)
        compressor = UntokenCompressor.from_pretrained(str(ckpt / "compressor"))
        rec_state = torch.load(ckpt / "reconstructor.pt", map_location="cpu")
        reconstructor.load_state_dict(rec_state["reconstructor"])
        if rank == 0:
            print(f"Resumed from {args.resume}")

    trainer = UntokenTrainer(
        compressor, reconstructor, discriminator, tokenizer, cfg,
        rank=rank, world_size=world_size,
    )

    phases = [args.phase] if args.phase else [1, 2, 3]

    if 1 in phases:
        ds1 = build_train_dataset(1, cfg.data_dir, tokenizer)
        trainer.train_phase1(ds1, quick=args.quick)

    if 2 in phases:
        ds2 = build_train_dataset(2, cfg.data_dir, tokenizer)
        real_ds = build_real_text_dataset(cfg.data_dir, tokenizer)
        trainer.train_phase2(ds2, real_ds, quick=args.quick)

    if 3 in phases:
        ds3 = build_train_dataset(3, cfg.data_dir, tokenizer)
        trainer.train_phase3(ds3, quick=args.quick)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
