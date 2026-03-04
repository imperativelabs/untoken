from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    OneCycleLR,
)
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from training.gumbel import apply_compression_mask, get_temperature, gumbel_soft_select, straight_through_select
from training.losses import (
    AdversarialGeneratorLoss,
    CompressionRatioLoss,
    DiscriminatorLoss,
    GatingRegularizationLoss,
    ReconstructionLoss,
    SemanticSimilarityLoss,
    SupervisedCompressionLoss,
)
from training.scheduler import TrainConfig


class UntokenTrainer:
    def __init__(self, compressor, reconstructor, discriminator, tokenizer, config: TrainConfig,
                 rank: int = 0, world_size: int = 1):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0
        self.device = torch.device(f"cuda:{rank}")

        self.compressor = compressor.to(self.device)
        self.reconstructor = reconstructor.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.tokenizer = tokenizer

        if world_size > 1:
            self.compressor = DDP(self.compressor, device_ids=[rank], find_unused_parameters=True)
            self.reconstructor = DDP(self.reconstructor, device_ids=[rank])
            self.discriminator = DDP(self.discriminator, device_ids=[rank])

        self._compile_models()

        self.loss_recon = ReconstructionLoss(pad_token_id=tokenizer.pad_token_id)
        self.loss_semantic = SemanticSimilarityLoss()
        self.loss_adv_g = AdversarialGeneratorLoss()
        self.loss_adv_d = DiscriminatorLoss()
        self.loss_ratio = CompressionRatioLoss(config.target_compression_ratio)
        self.loss_gating = GatingRegularizationLoss()
        self.loss_supervised = SupervisedCompressionLoss()

        self.wandb = None
        if self.is_main:
            self._init_wandb()

    def _compile_models(self):
        pass

    def _init_wandb(self):
        try:
            import wandb
            wandb.init(project="untoken", mode=os.environ.get("WANDB_MODE", "disabled"))
            self.wandb = wandb
        except Exception:
            pass

    def _log(self, metrics: dict, step: int):
        if self.is_main and self.wandb:
            self.wandb.log(metrics, step=step)

    def _make_loader(self, dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle) \
            if self.world_size > 1 else None
        nw = self.config.num_workers
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(shuffle and sampler is None),
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
            prefetch_factor=(2 if nw > 0 else None),
            multiprocessing_context=("spawn" if nw > 0 else None),
        )

    def _make_optimizers_phase1(self, phase_cfg):
        raw_comp = self.compressor.module if hasattr(self.compressor, "module") else self.compressor
        param_groups = [
            {"params": raw_comp.encoder.parameters(), "lr": phase_cfg.lr * 0.1},
            {"params": raw_comp.head.parameters(), "lr": phase_cfg.lr},
        ]
        opt_c = AdamW(param_groups, weight_decay=self.config.weight_decay, betas=self.config.betas)
        raw_rec = self.reconstructor.module if hasattr(self.reconstructor, "module") else self.reconstructor
        opt_r = AdamW(raw_rec.parameters(), lr=phase_cfg.lr, weight_decay=self.config.weight_decay)
        return opt_c, opt_r

    def _make_optimizers_phase2(self, phase_cfg):
        raw_comp = self.compressor.module if hasattr(self.compressor, "module") else self.compressor
        raw_rec = self.reconstructor.module if hasattr(self.reconstructor, "module") else self.reconstructor
        raw_disc = self.discriminator.module if hasattr(self.discriminator, "module") else self.discriminator
        opt_g = AdamW(
            list(raw_comp.parameters()) + list(raw_rec.parameters()),
            lr=phase_cfg.lr, weight_decay=self.config.weight_decay, betas=self.config.betas,
        )
        opt_d = AdamW(raw_disc.parameters(), lr=phase_cfg.disc_lr,
                      weight_decay=self.config.weight_decay, betas=self.config.betas)
        return opt_g, opt_d

    def _make_optimizer_phase3(self, phase_cfg):
        raw_comp = self.compressor.module if hasattr(self.compressor, "module") else self.compressor
        raw_rec = self.reconstructor.module if hasattr(self.reconstructor, "module") else self.reconstructor
        return AdamW(
            list(raw_comp.parameters()) + list(raw_rec.parameters()),
            lr=phase_cfg.lr, weight_decay=self.config.weight_decay, betas=self.config.betas,
        )

    def _forward_compressor(self, batch, soft_mask_fn):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        special_tokens_mask = batch["special_tokens_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        scores = self.compressor(input_ids, attention_mask, special_tokens_mask)
        keep_mask = soft_mask_fn(scores)
        keep_mask = keep_mask * attention_mask.float()
        keep_mask = keep_mask * (1 - special_tokens_mask.float())

        comp_ids, comp_pos, comp_attn = apply_compression_mask(
            input_ids, keep_mask, self.tokenizer.pad_token_id
        )
        return scores, keep_mask, comp_ids, comp_pos, comp_attn, input_ids, attention_mask, labels

    def train_phase1(self, dataset, quick: bool = False):
        cfg = self.config.phase1
        raw_comp = self.compressor.module if hasattr(self.compressor, "module") else self.compressor
        raw_comp.freeze_encoder()

        loader = self._make_loader(dataset, cfg.batch_size)
        if self.is_main:
            print(f"[phase1] loader batches={len(loader)}, dataset size={len(dataset)}")
        opt_c, opt_r = self._make_optimizers_phase1(cfg)
        total_steps = max(1, len(loader) * cfg.epochs)
        sched_c = OneCycleLR(opt_c, max_lr=cfg.lr, total_steps=total_steps, pct_start=cfg.warmup_ratio)
        sched_r = OneCycleLR(opt_r, max_lr=cfg.lr, total_steps=total_steps, pct_start=cfg.warmup_ratio)

        global_step = 0
        for epoch in range(cfg.epochs):
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            for batch in tqdm(loader, disable=not self.is_main, desc=f"Phase1 E{epoch+1}"):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    soft_fn = lambda s: gumbel_soft_select(s, cfg.gumbel_temperature)
                    scores, keep_mask, comp_ids, comp_pos, comp_attn, input_ids, attn_mask, labels = \
                        self._forward_compressor(batch, soft_fn)

                    recon_logits = self.reconstructor(comp_ids, comp_pos, input_ids, comp_attn)
                    l_r = self.loss_recon(recon_logits, input_ids)
                    l_c = self.loss_ratio(scores, attn_mask)
                    l_g = self.loss_gating(scores, attn_mask)
                    l_s = self.loss_supervised(scores, labels)

                    w = cfg.losses
                    loss = w.reconstruction * l_r + w.compression_ratio * l_c + \
                           w.gating * l_g + w.supervised * l_s

                opt_c.zero_grad(); opt_r.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.compressor.parameters()) + list(self.reconstructor.parameters()),
                    self.config.max_grad_norm
                )
                opt_c.step(); opt_r.step()
                sched_c.step(); sched_r.step()

                self._log({"phase1/loss": loss.item(), "phase1/recon": l_r.item(),
                           "phase1/ratio": l_c.item()}, global_step)
                global_step += 1
                if quick and global_step >= 2:
                    break
            if quick:
                break

        raw_comp.unfreeze_encoder()
        self._save_checkpoint("phase1")

    def train_phase2(self, train_dataset, real_dataset, quick: bool = False):
        cfg = self.config.phase2
        loader = self._make_loader(train_dataset, cfg.batch_size)
        real_loader = self._make_loader(real_dataset, cfg.batch_size)
        real_iter = iter(real_loader)

        opt_g, opt_d = self._make_optimizers_phase2(cfg)
        total_steps = len(loader) * cfg.epochs // cfg.grad_accum
        sched_g = CosineAnnealingWarmRestarts(opt_g, T_0=total_steps // cfg.cosine_restarts)

        global_step = 0
        phase_total = len(loader) * cfg.epochs
        for epoch in range(cfg.epochs):
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            for step, batch in enumerate(tqdm(loader, disable=not self.is_main, desc=f"Phase2 E{epoch+1}")):
                current = epoch * len(loader) + step
                temp = get_temperature(current, phase_total, cfg.gumbel_t_start, cfg.gumbel_t_end)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    soft_fn = lambda s: gumbel_soft_select(s, temp)
                    scores, keep_mask, comp_ids, comp_pos, comp_attn, input_ids, attn_mask, labels = \
                        self._forward_compressor(batch, soft_fn)

                    recon_logits = self.reconstructor(comp_ids, comp_pos, input_ids, comp_attn)
                    l_r = self.loss_recon(recon_logits, input_ids)
                    l_c = self.loss_ratio(scores, attn_mask)
                    l_g = self.loss_gating(scores, attn_mask)
                    fake_logits = self.discriminator(comp_ids, comp_attn)
                    l_a = self.loss_adv_g(fake_logits)

                    w = cfg.losses
                    loss_g = (w.reconstruction * l_r + w.compression_ratio * l_c +
                              w.gating * l_g + w.adversarial * l_a)

                    if (step + 1) % self.config.semantic_loss_interval == 0:
                        orig_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                        comp_texts = self.tokenizer.batch_decode(comp_ids, skip_special_tokens=True)
                        l_sem = self.loss_semantic(orig_texts, comp_texts, self.device)
                        loss_g = loss_g + w.semantic * l_sem

                if (step + 1) % cfg.grad_accum == 0:
                    opt_g.zero_grad()
                    (loss_g / cfg.grad_accum).backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.compressor.parameters()) + list(self.reconstructor.parameters()),
                        self.config.max_grad_norm
                    )
                    opt_g.step()
                    sched_g.step()
                else:
                    (loss_g / cfg.grad_accum).backward()

                try:
                    real_batch = next(real_iter)
                except StopIteration:
                    real_iter = iter(real_loader)
                    real_batch = next(real_iter)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    real_ids = real_batch["input_ids"].to(self.device)
                    real_attn = real_batch["attention_mask"].to(self.device)
                    real_logits = self.discriminator(real_ids, real_attn)
                    loss_d = self.loss_adv_d(real_logits, fake_logits.detach())

                opt_d.zero_grad()
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.config.max_grad_norm)
                opt_d.step()

                self._log({"phase2/loss_g": loss_g.item(), "phase2/loss_d": loss_d.item(),
                           "phase2/temperature": temp}, global_step)
                global_step += 1
                if quick and global_step >= 2:
                    break
            if quick:
                break

        self._save_checkpoint("phase2")

    def train_phase3(self, dataset, quick: bool = False):
        cfg = self.config.phase3
        loader = self._make_loader(dataset, cfg.batch_size)
        opt = self._make_optimizer_phase3(cfg)
        total_steps = len(loader) * cfg.epochs
        sched = LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=total_steps)

        global_step = 0
        for epoch in range(cfg.epochs):
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)
            for batch in tqdm(loader, disable=not self.is_main, desc=f"Phase3 E{epoch+1}"):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    scores, keep_mask, comp_ids, comp_pos, comp_attn, input_ids, attn_mask, labels = \
                        self._forward_compressor(batch, straight_through_select)

                    recon_logits = self.reconstructor(comp_ids, comp_pos, input_ids, comp_attn)
                    l_r = self.loss_recon(recon_logits, input_ids)
                    l_c = self.loss_ratio(scores, attn_mask)

                    w = cfg.losses
                    loss = w.reconstruction * l_r + w.compression_ratio * l_c

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.compressor.parameters()) + list(self.reconstructor.parameters()),
                    self.config.max_grad_norm
                )
                opt.step()
                sched.step()

                self._log({"phase3/loss": loss.item(), "phase3/recon": l_r.item()}, global_step)
                global_step += 1
                if quick and global_step >= 2:
                    break
            if quick:
                break

        self._save_checkpoint("phase3")

    def _save_checkpoint(self, tag: str):
        if not self.is_main:
            return
        ckpt_dir = Path(self.config.checkpoint_dir) / tag
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        raw_comp = self.compressor.module if hasattr(self.compressor, "module") else self.compressor
        raw_comp.save_pretrained(str(ckpt_dir / "compressor"))
        torch.save({
            "reconstructor": (self.reconstructor.module if hasattr(self.reconstructor, "module")
                              else self.reconstructor).state_dict(),
        }, ckpt_dir / "reconstructor.pt")
        print(f"[rank0] Saved checkpoint: {ckpt_dir}")
