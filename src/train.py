# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import hashlib
import logging
import os
import random
import subprocess
import sys
import yaml

import numpy as np

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.pretrained import init_from_pretrained, sha256_file
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.agri_dataset import make_agrifield40k

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def git_commit():
    """Best-effort current commit hash for provenance in checkpoints."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def hardware_report(device):
    """Describe the device actually used, so runs are attributable."""
    if device.type != 'cuda':
        return {'device_type': device.type}
    index = torch.cuda.current_device()
    return {
        'device_type': 'cuda',
        # index is the program-visible ordinal; physical is what nvidia-smi shows
        'visible_device_index': index,
        'name': torch.cuda.get_device_name(index),
        'total_memory_mib': float(torch.cuda.get_device_properties(index).total_memory / 1024 ** 2),
        # str(): torch.__version__ is a str subclass that yaml.safe_dump rejects
        'torch_version': str(torch.__version__),
        'cuda_version': str(torch.version.cuda),
        'world_visible_devices': int(torch.cuda.device_count()),
    }



def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    # -- Stage 2: single-GPU engineering knobs (do not change the JEPA objective)
    use_sdpa = bool(args['meta'].get('use_sdpa', False))
    activation_checkpointing = bool(args['meta'].get('activation_checkpointing', False))
    pretrained_checkpoint = args['meta'].get('pretrained_checkpoint')
    pretrained_sha256 = args['meta'].get('pretrained_sha256')
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    dataset_name = args['data'].get('dataset', 'imagenet1k')
    manifest_path = args['data'].get('manifest')
    data_split = args['data'].get('split', 'train')
    data_seed = int(args['data'].get('seed', _GLOBAL_SEED))
    max_samples = args['data'].get('max_samples')
    random.seed(data_seed)
    np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(data_seed)
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    stop_after_epoch = args['optimization'].get('stop_after_epoch')
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']
    # -- gradient accumulation: micro-steps per optimizer update
    grad_accum = int(args['optimization'].get('grad_accum', 1))
    if grad_accum < 1:
        raise ValueError(f'optimization.grad_accum must be >= 1, got {grad_accum}')

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    os.makedirs(folder, exist_ok=True)
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        use_sdpa=use_sdpa,
        activation_checkpointing=activation_checkpointing)
    target_encoder = copy.deepcopy(encoder)

    pretrained_report = None
    if (not load_model) and pretrained_checkpoint:
        # ---------------------------------------------------------------- #
        #  OPERATION A: Pretrained Initialization
        #  Transfer the official I-JEPA weights into a NEW agricultural run.
        #  Optimizer / schedulers / scaler / epoch counters are NOT loaded.
        # ---------------------------------------------------------------- #
        if pretrained_sha256:
            actual = sha256_file(pretrained_checkpoint)
            if actual != pretrained_sha256:
                raise ValueError(
                    f'Pretrained checkpoint SHA-256 mismatch: expected '
                    f'{pretrained_sha256}, got {actual}')
            logger.info(f'Pretrained checkpoint SHA-256 verified: {actual}')
        pretrained_report = init_from_pretrained(
            pretrained_checkpoint,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            device=device)
        logger.info(
            'Operation A: loaded %d encoder / %d predictor / %d target_encoder tensors',
            pretrained_report['encoder_keys_loaded'],
            pretrained_report['predictor_keys_loaded'],
            pretrained_report.get('target_encoder_keys_loaded'))

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders/samplers
    if dataset_name == 'agrifield40k':
        if not manifest_path:
            raise ValueError('data.manifest is required for dataset=agrifield40k')
        _, unsupervised_loader, unsupervised_sampler = make_agrifield40k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            manifest_path=manifest_path,
            root_path=root_path,
            split=data_split,
            drop_last=True,
            seed=data_seed,
            max_samples=max_samples)
    elif dataset_name == 'imagenet1k':
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    ipe = len(unsupervised_loader)
    # -- number of optimizer updates per epoch under gradient accumulation
    optimizer_steps_per_epoch = (ipe + grad_accum - 1) // grad_accum
    manifest_sha256 = None
    if dataset_name == 'agrifield40k':
        with open(manifest_path, 'rb') as manifest_file:
            manifest_sha256 = hashlib.file_digest(manifest_file, 'sha256').hexdigest()

    # -- init optimizer and scheduler
    #    LR/WD schedules advance once per optimizer update, so they are sized in
    #    optimizer steps, not micro-steps.
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=optimizer_steps_per_epoch,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule (one EMA update per optimizer update)
    total_optimizer_steps = optimizer_steps_per_epoch * num_epochs * ipe_scale
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/total_optimizer_steps
                          for i in range(int(total_optimizer_steps)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
        if 'rng_state' not in checkpoint:
            raise ValueError(f'Checkpoint has no RNG state: {load_path}')
        if dataset_name == 'agrifield40k':
            if checkpoint.get('manifest_sha256') != manifest_sha256:
                raise ValueError(f'Checkpoint manifest does not match {manifest_path}')
        rng_state = checkpoint['rng_state']
        random.setstate(rng_state['python'])
        np.random.set_state(rng_state['numpy'])
        torch.set_rng_state(rng_state['torch'])
        if torch.cuda.is_available() and rng_state['cuda'] is not None:
            torch.cuda.set_rng_state_all(rng_state['cuda'])
        if 'loader_generator_state' in checkpoint:
            unsupervised_loader.generator.set_state(checkpoint['loader_generator_state'])
        mask_collator._itr_counter.value = checkpoint['mask_collator_step']
        # -- advance LR/WD/momentum to the resumed optimizer-step position
        for _ in range(start_epoch * optimizer_steps_per_epoch):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)

    optimizer_steps = checkpoint.get('optimizer_steps', 0) if load_model else 0
    optimizer_steps_at_start = optimizer_steps
    micro_steps = checkpoint.get('micro_steps', 0) if load_model else 0
    micro_steps_at_start = micro_steps
    ema_formula_verified = False
    first_loss = None
    last_loss = None
    if stop_after_epoch is not None and not (0 < stop_after_epoch <= num_epochs):
        raise ValueError('optimization.stop_after_epoch must be within the configured epoch range')
    epoch_limit = min(num_epochs, stop_after_epoch or num_epochs)
    if load_model and start_epoch >= epoch_limit:
        raise ValueError(f'No epochs remain: checkpoint epoch={start_epoch}, limit={epoch_limit}')
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def save_checkpoint(epoch):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'optimizer_steps': optimizer_steps,
            'micro_steps': micro_steps,
            'grad_accum': grad_accum,
            'config': args,
            'hardware': hardware_report(device),
            'git_commit': git_commit(),
            'pretrained_init': None if pretrained_report is None else {
                'source': pretrained_report['path'],
                'sha256': pretrained_report['sha256'],
                'load_mode': pretrained_report['load_mode'],
                'encoder_keys_loaded': pretrained_report['encoder_keys_loaded'],
                'predictor_keys_loaded': pretrained_report['predictor_keys_loaded'],
                'target_encoder_keys_loaded': pretrained_report.get('target_encoder_keys_loaded'),
                'predictor_inherited': pretrained_report['predictor_inherited'],
                'source_epoch': pretrained_report['epoch_in_source_checkpoint'],
            },
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            'loader_generator_state': unsupervised_loader.generator.get_state(),
            'mask_collator_step': mask_collator._itr_counter.value,
        }
        if manifest_sha256 is not None:
            save_dict['manifest_sha256'] = manifest_sha256
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                torch.save(save_dict, save_path.format(epoch=f'{epoch + 1}'))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, epoch_limit):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()
        step_times_ms = []

        # -- gradient-accumulation state for this epoch.  Read LR/WD from a group
        #    that is NOT weight-decay-excluded, so the reported WD is real.
        _last_lr = optimizer.param_groups[0]['lr']
        _last_wd = next(g['weight_decay'] for g in optimizer.param_groups
                        if not g.get('WD_exclude', False))

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            # -- deterministic accumulation window: derived only from (itr, ipe)
            #    so a resumed run reconstructs exactly the same windows.
            window_start = (itr // grad_accum) * grad_accum
            window_size = min(grad_accum, ipe - window_start)
            do_update = (itr - window_start + 1) == window_size

            def load_imgs():
                # -- unsupervised imgs
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                nonlocal ema_formula_verified, _last_lr, _last_wd
                if do_update:
                    # -- LR/WD advance once per optimizer update, not per micro-step
                    _last_lr = scheduler.step()
                    _last_wd = wd_scheduler.step()
                _new_lr, _new_wd = _last_lr, _last_wd
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.amp.autocast(
                    device_type='cuda', dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss at epoch={epoch + 1}, itr={itr}: {loss}')

                #  Step 2. Backward & step
                #    BF16 keeps the upstream GradScaler path (scale on backward,
                #    unscale before step).  With gradient accumulation the scaling
                #    factor cancels, so a single scaler window spans the
                #    accumulation window and the unscale happens at the update.
                if use_bfloat16:
                    scaler.scale(loss / grad_accum).backward()
                else:
                    (loss / grad_accum).backward()

                if not ema_formula_verified:
                    def first_gradient_probe(module, label):
                        for name, param in module.named_parameters():
                            if param.grad is not None and torch.count_nonzero(param.grad).item():
                                return name, param
                        raise RuntimeError(f'No nonzero {label} gradient was produced')

                    encoder_probe_name, encoder_probe = first_gradient_probe(encoder.module, 'encoder')
                    _, predictor_probe = first_gradient_probe(predictor.module, 'predictor')
                    target_probe = dict(target_encoder.module.named_parameters())[encoder_probe_name]
                    encoder_before = encoder_probe.detach().clone()
                    predictor_before = predictor_probe.detach().clone()
                    target_before = target_probe.detach().clone()

                if do_update:
                    if use_bfloat16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    grad_stats = grad_logger(encoder.named_parameters())
                    optimizer.zero_grad()

                    # Step 3. momentum update of target encoder
                    with torch.no_grad():
                        m = next(momentum_scheduler)
                        if not ema_formula_verified:
                            expected_target = target_before.mul(m).add_(
                                (1. - m) * encoder_probe.detach())
                        for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                            param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
                        if not ema_formula_verified:
                            if torch.equal(encoder_before, encoder_probe) or torch.equal(predictor_before, predictor_probe):
                                raise RuntimeError('Optimizer did not update encoder and predictor parameters')
                            if not torch.allclose(target_probe, expected_target, rtol=1e-6, atol=1e-7):
                                raise RuntimeError('Target encoder EMA does not match the configured update formula')
                            ema_formula_verified = True
                    return (loss.detach().item(), _new_lr, _new_wd, grad_stats)
                else:
                    grad_stats = None
                    return (loss.detach().item(), _new_lr, _new_wd, grad_stats)

            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)
            step_times_ms.append(etime)
            micro_steps += 1
            first_loss = loss if first_loss is None else first_loss
            last_loss = loss

            if do_update:
                optimizer_steps += 1

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            if not np.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch + 1}, itr={itr}: {loss}')

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)

    if not ema_formula_verified:
        raise RuntimeError('No optimizer update was verified in this run')
    if rank == 0:
        verification = {
            'dataset': dataset_name,
            'split': data_split,
            'model_name': model_name,
            'patch_size': patch_size,
            'pred_depth': pred_depth,
            'pred_emb_dim': pred_emb_dim,
            'use_sdpa': use_sdpa,
            'activation_checkpointing': activation_checkpointing,
            'grad_accum': grad_accum,
            'optimizer_steps_this_run': optimizer_steps - optimizer_steps_at_start,
            'optimizer_steps_total': optimizer_steps,
            'micro_steps_this_run': micro_steps - micro_steps_at_start,
            'micro_steps_total': micro_steps,
            'optimizer_steps_per_epoch': optimizer_steps_per_epoch,
            'loss_first': first_loss,
            'loss_last': last_loss,
            'loss_finite': True,
            'encoder_updated': True,
            'predictor_updated': True,
            'ema_formula_verified': ema_formula_verified,
            'checkpoint': latest_path,
            'checkpoint_epoch': epoch_limit,
            'batch_size': batch_size,
            'crop_size': crop_size,
            'peak_memory_mib': torch.cuda.max_memory_allocated(device) / 1024.**2 if device.type == 'cuda' else None,
            'step_time_ms_mean': float(np.mean(step_times_ms)) if step_times_ms else None,
            'step_time_ms_last': step_times_ms[-1] if step_times_ms else None,
            'manifest_sha256': manifest_sha256,
            'git_commit': git_commit(),
            'hardware': hardware_report(device),
            'pretrained_init': None if pretrained_report is None else {
                'source': pretrained_report['path'],
                'sha256': pretrained_report['sha256'],
                'encoder_keys_loaded': pretrained_report['encoder_keys_loaded'],
                'predictor_keys_loaded': pretrained_report['predictor_keys_loaded'],
                'target_encoder_keys_loaded': pretrained_report.get('target_encoder_keys_loaded'),
                'predictor_inherited': pretrained_report['predictor_inherited'],
            },
        }
        with open(os.path.join(folder, 'training_verification.yaml'), 'w') as verification_file:
            yaml.safe_dump(verification, verification_file)


if __name__ == "__main__":
    main()
