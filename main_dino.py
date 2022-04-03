# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import copy
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision.transforms import Compose
from torch.utils.tensorboard import SummaryWriter

import utils
from dino_cpc.loss import DINOLossCPC
from dino_cpc.transforms import MinimalSizeResize

from dino_cpc.transforms import DataAugmentationDINOCPC
from dino_cpc.utils import handle_flips, PatchMatcher
from models import swin_transformer as swins
from models.vision_transformer import DINOHead

torchvision_archs = sorted(name for name in torchvision_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(torchvision_models.__dict__[name]))


def get_args_parser():
    # TODO: remove all the transformers from this code, keep only swin (the code is broken from the others)
    parser = argparse.ArgumentParser('DINO', add_help=False)

    parser.add_argument("--exp_name", required=True, help="The experiment name")
    parser.add_argument("--use_wandb", default=False, type=utils.bool_flag, help="If True will log results to wandb")

    # Model parameters
    parser.add_argument('--arch', default='swin_small', type=str,
                        choices=['swin_small', 'swin_tiny'],
                        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=4, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=(65536 // 16, 65536 // 8, 65536 // 4, 65536 // 2), type=int, nargs=4, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well, one for each level""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
                        help="Whether to use batch normalizations in projection head (Default: False)")
    parser.add_argument("--add_global_dino_loss", default=False, type=utils.bool_flag,
                        help="Whether to add a global dino loss for each level (SWIN only)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=55, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=300, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=5, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--max_pairs', type=int, nargs=4, default=(512, 512, 512, 512),
                        help="Maximum number of pairs to take at each layer")
    parser.add_argument('--multi_level_matching', type=utils.bool_flag, default=True, help="If True will use multi level patching")

    # Multi-crop parameters
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--global_crop_size', type=int, default=224, help="The size of the global crop")
    parser.add_argument('--local_crop_size', type=int, default=96, help="The size of the local crop")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
                        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser


def train_dino(args):
    utils.init_distributed_mode(args)

    if args.use_wandb and utils.is_main_process():
        import wandb
        wandb.init(name=args.exp_name, project="DINO_SWIN_IMPROVMENTS", sync_tensorboard=True)
        wandb.config.update(vars(args))

    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building student and teacher networks ... ============
    if args.arch in swins.__dict__.keys():
        student = swins.__dict__[args.arch](
            img_size=args.global_crop_size,
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = swins.__dict__[args.arch](patch_size=args.patch_size,
                                            img_size=args.global_crop_size)
        student.head = None  # Multi cropper makes it identity so no need to assign it
        teacher.head = None
        num_features = student.num_features
    else:
        raise RuntimeError(f"Unknown architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student_head = torch.nn.ModuleDict()
    teacher_head = torch.nn.ModuleDict()
    num_patches_map = {}
    with torch.no_grad():
        for curr_res in (args.global_crop_size, args.local_crop_size):
            # Run dummy input in order to understand the dimensions of the embeddings and the crops coordinates
            out_agg = student(torch.rand(1, 3, curr_res, curr_res))[0]
            num_patches_map[curr_res] = [l.size(2) for l in out_agg]
            res_config = []
            patch_size = student.patch_embed.patch_size
            assert patch_size[0] == patch_size[1], "Patches must be symmetric"
            for i, out in enumerate(out_agg):  # out layout: [batch, dim, num_patches_w, num_patches_h]
                dim = out.size(-3)
                res_config.append(dim)

            for i, curr_dim in enumerate(res_config):
                #  Although the teacher does not need all the dino heads as the student because it see only the global scales
                #  We still add the heads inorder to make the networks identical
                teacher_head[f"{curr_res}_{i}"] = DINOHead(in_dim=curr_dim,
                                                           out_dim=args.out_dim[i],
                                                           use_bn=args.use_bn_in_head,
                                                           norm_last_layer=True)
                student_head[f"{curr_res}_{i}"] = DINOHead(in_dim=curr_dim,
                                                           out_dim=args.out_dim[i],
                                                           use_bn=args.use_bn_in_head,
                                                           norm_last_layer=args.norm_last_layer)

    if args.multi_level_matching:
        # We "double" the network depth, we take all the layers twice except the last layer
        args.max_pairs = tuple(list(args.max_pairs) + list(args.max_pairs[:-1]))
        new_patch_map = {}
        for res, patch_size in num_patches_map.items():
            new_val = list(patch_size) + [p / 2 for p in patch_size[:-1]]  # We dont take the last one (small enough)
            for p in new_val: assert p % 1 == 0
            new_val = [int(p) for p in new_val]
            new_patch_map[res] = new_val
        num_patches_map = new_patch_map

    student = utils.MultiCropWrapper(student, student_head, args.global_crop_size, args.local_crop_size)
    teacher = utils.MultiCropWrapper(teacher, teacher_head, args.global_crop_size, args.local_crop_size)

    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing data ... ============
    # assert args.global_crop_size // args.local_crop_size == args.global_crop_size / args.local_crop_size, "The global crop should be a multiplication of the local crop size so we can compare the patches"
    maximal_patch_size = args.patch_size * (2 ** (student.module.backbone.num_layers - 1))
    transform = Compose([
        MinimalSizeResize(minimal_size=maximal_patch_size * 10,
                          patch_size=maximal_patch_size),
        # TODO: make it random resize between ranges, merge it into DataAugmentationDINOCPC ?
        DataAugmentationDINOCPC(local_crops_number=args.local_crops_number,
                                global_crop_size=args.global_crop_size,
                                local_crop_size=args.local_crop_size,
                                maximal_patch_size=maximal_patch_size)
    ])
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ preparing loss ... ============
    dino_loss = DINOLossCPC(args=args, num_features=num_features)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    writer = SummaryWriter(log_dir=args.output_dir) if utils.is_main_process() else None

    start_time = time.time()
    print("Starting DINO training !")
    it = 0
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        with torch.autograd.set_detect_anomaly(True):
            train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
                                          data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
                                          epoch, fp16_scaler, args, num_patches_map, writer, it, args.exp_name, args.multi_level_matching)
        it = train_stats['it']
        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.use_wandb:
        wandb.finish()


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, patches_map, writer, it, experiment_name, multi_lvl_patch: bool):
    metric_logger = utils.MetricLogger(delimiter="  ", writer=writer, it=it, experiment_name=experiment_name)
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for _, ((images, crops_bbox, crops_flipped, orig_img_size), _) in enumerate(
            metric_logger.log_every(data_loader, 10, header)):

        # update weight decay and learning rate according to their schedule
        it = metric_logger.it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move data to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        crops_bbox = crops_bbox.cuda()
        crops_flipped = crops_flipped.cuda()

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # Find out what we need to calculate
            global_bbox = crops_bbox[:, :2, :].reshape(-1, 4)  # First 2 are the global boxes
            local_bbox = crops_bbox[:, 2:, :].reshape(-1, 4)  # The rest are local

            # Reshape inputs so they will have batch dim
            batch_size = args.batch_size_per_gpu
            global_bbox = global_bbox.view(batch_size, -1, 4)
            local_bbox = local_bbox.view(batch_size, -1, 4)

            relevant_matches_local_global = []
            relevant_matches_global_global = []

            max_pairs = args.max_pairs

            # Now we check which local boxes matches the global boxes and compute the loss between them as well
            for lvl_idx, num_pairs_to_grab in enumerate(max_pairs):
                matches_local_global = PatchMatcher.find_matches(crop_a=local_bbox,
                                                                 crop_size_a=args.local_crop_size,
                                                                 num_patches_a=patches_map[args.local_crop_size][
                                                                     lvl_idx],
                                                                 crop_b=global_bbox,
                                                                 crop_size_b=args.global_crop_size,
                                                                 num_patches_b=patches_map[args.global_crop_size][
                                                                     lvl_idx])

                matches_global_global = PatchMatcher.find_matches(crop_a=global_bbox,
                                                                  crop_size_a=args.global_crop_size,
                                                                  num_patches_a=patches_map[args.global_crop_size][
                                                                      lvl_idx],
                                                                  crop_b=global_bbox,
                                                                  crop_size_b=args.global_crop_size,
                                                                  num_patches_b=patches_map[args.global_crop_size][
                                                                      lvl_idx])
                # Remove all global matches which matches the same view, because its not interesting
                matches_global_global = matches_global_global[
                    matches_global_global[:, 0] != matches_global_global[:, 2]]

                # # START - DEBUG CODE
                # if lvl_idx == 3:
                #     global_images = torch.cat([im.unsqueeze(1) for im in images[:2]], dim=1).view(-1, 224, 224, 3)
                #     num_patches = patches_map[args.global_crop_size][lvl_idx]
                #     crop_size = 224
                #     patch_size_pixels = crop_size // num_patches
                #     crop_a = global_images[0].cpu().numpy().astype(np.float32) / 255
                #     crop_b = global_images[1].cpu().numpy().astype(np.float32) / 255
                #     relevant_matches = [c for c in matches_global_global.numpy() if c[0] == 0 and c[2] == 1]
                #     for curr_match in relevant_matches:
                #         patch_idx_a_yx = (curr_match[1] // num_patches, curr_match[1] % num_patches)
                #         patch_idx_b_yx = (curr_match[3] // num_patches, curr_match[3] % num_patches)
                #
                #         crop_a[patch_idx_a_yx[0] * patch_size_pixels: (patch_idx_a_yx[0] + 1) * (patch_size_pixels), patch_idx_a_yx[1] * patch_size_pixels: (patch_idx_a_yx[1] + 1)* patch_size_pixels, :] = (crop_a[patch_idx_a_yx[0] * patch_size_pixels: (patch_idx_a_yx[0] + 1) * (patch_size_pixels), patch_idx_a_yx[1] * patch_size_pixels: (patch_idx_a_yx[1] + 1)* patch_size_pixels, :] + (0, 0, 1)) / 2
                #         crop_b[patch_idx_b_yx[0] * patch_size_pixels: (patch_idx_b_yx[0] + 1) * patch_size_pixels,
                #         patch_idx_b_yx[1] * patch_size_pixels: (patch_idx_b_yx[1] + 1) * patch_size_pixels, :] = (crop_b[patch_idx_b_yx[0] * patch_size_pixels: (patch_idx_b_yx[0] + 1) * patch_size_pixels,
                #         patch_idx_b_yx[1] * patch_size_pixels: (patch_idx_b_yx[1] + 1) * patch_size_pixels, :] + (0, 0, 1)) / 2
                #         break
                #     attached_crops = np.concatenate([crop_a, crop_b], axis=1)
                #     loli =3
                # # END - DEBUG CODE

                # Take subset
                for curr_matches, rel_matches in ((matches_local_global, relevant_matches_local_global),
                                                  (matches_global_global, relevant_matches_global_global)):
                    perm = torch.randperm(curr_matches.size(0))
                    idx = perm[:max_pairs[lvl_idx]]
                    rel_matches.append(curr_matches[idx])

            teacher_relevant_matches_global = [torch.cat([r1[:, 2:], r2[:, 2:]], dim=0) for r1, r2 in
                                               zip(relevant_matches_local_global, relevant_matches_global_global)]
            student_relevant_matches_local = [r[:, :2] for r in relevant_matches_local_global]
            student_relevant_matches_global = [r[:, :2] for r in relevant_matches_global_global]

            # Make the order of the images the same as the bbox
            student_images = []
            teacher_images = []
            for curr_image_id in range(args.batch_size_per_gpu):
                for curr_crop_index in range(len(images)):
                    curr_image = images[curr_crop_index][curr_image_id].unsqueeze(0)
                    student_images.append(curr_image)
                    if curr_crop_index < 2:  # only global
                        teacher_images.append(curr_image)

            teacher_output, teacher_raw_output = teacher(teacher_images, crops_flipped,
                                     global_matches=teacher_relevant_matches_global,
                                                         multi_lvl_match=multi_lvl_patch)  # only the 2 global views pass through the teacher
            student_output, student_raw_output = student(student_images, crops_flipped, local_matches=student_relevant_matches_local,
                                     global_matches=student_relevant_matches_global,
                                                         multi_lvl_match=multi_lvl_patch)

            global_match_student_pred = student_output[args.global_crop_size]
            local_match_student_pred = student_output[args.local_crop_size]
            local_match_teacher_pred = [o[:local_match_student_pred[i].size(0), :] for i, o in
                                        enumerate(teacher_output[args.global_crop_size])]
            global_match_teacher_pred = [o[local_match_student_pred[i].size(0):, :] for i, o in
                                         enumerate(teacher_output[args.global_crop_size])]

            loss, sep_loss, global_loss = dino_loss(global_match_student_pred,
                                                    local_match_student_pred,
                                                    local_match_teacher_pred,
                                                    global_match_teacher_pred,
                                                    teacher_raw_output,
                                                    student_raw_output,
                                                    epoch,
                                                    args)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(**{f"loss_lvl_{i}": lvl_loss for i, lvl_loss in enumerate(sep_loss)})
        if len(global_loss):
            metric_logger.update(**{f"global_loss_lvl_{i}": lvl_loss for i, lvl_loss in enumerate(global_loss)})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        metric_logger.finish_step()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['it'] = metric_logger.it

    return stats


def save_args(args: argparse.Namespace):
    args_dict = vars(args)
    with open(os.path.join(args.output_dir, "input_args.json"), "w") as out_file:
        json.dump(args_dict, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir,
                                   f"{args.exp_name}_{timestampStr}")  # This code breaks the ability to revive workers
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    save_args(args)

    train_dino(args)
