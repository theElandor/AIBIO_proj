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
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
from dataset import Rxrx1
from collate import *
from hubconf import dino_vitb16
from bio_utils import load_dino_weights, get_samples_per_domain, get_batch_domains

import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import models as torchvision_models
import pandas as pd
import wandb
import utils
import vision_transformer as vits
from vision_transformer import DINOHead
from dist_utils import *


def init_wandb(dino_config):
    #assert wandb.Api().api_key, "the api key has not been set!\n"
    # print(f"wandb key: {wandb.api.api_key}")
    wandb.login(verify=True)
    wandb.init(
        project="AIBIO_proj",
        name="dino_train",
        config=dino_config
    )

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=30, type=int,
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
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=20, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    
    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.9, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.3, 0.5),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    # ================== custom options ==============
    parser.add_argument("--load_pretrained", default=None, type=str, help="Specify weights paths to load them.")    
    parser.add_argument("--metadata_path", default=None, type=str, help="Path of the metadata to use.")
    parser.add_argument("--cell_type", default=None, type=str, help="Cell type to use.")
    parser.add_argument("--acc_steps", default=1, type=int, help="Gradient accumulation steps to perform.") # B * steps = effective B    
    parser.add_argument("--easy_task", default=False, type=utils.bool_flag, help="If True, uses easy augmentations.")
    parser.add_argument("--sample_diff_cell_type", default=False, type=utils.bool_flag, help="If True, cross domain learning is applied also on cell_type.")
    parser.add_argument("--channels",default=6, type=int, help="Number of channels to use during training.")
    # ================== new loss options ==============
    parser.add_argument("--custom_loss", default=False, type=utils.bool_flag, help="Whether to use CDCL loss function.")
    parser.add_argument("--multi_center_training", default=False, type=utils.bool_flag, help="Whether to use CDCL loss function.")
    parser.add_argument("--barlow_loss", default=False, type=utils.bool_flag, help="Whether to use Barlow loss.")
    parser.add_argument("--barlow_loss_weight", default=0.2, type=float, help="Barlow loss weight")
    return parser


def train_dino(args):
    #utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    init_wandb(dict(vars(args)))
    cudnn.benchmark = True

    # ============ preparing data ... ============

    df = pd.read_csv(args.metadata_path)
    dataset = Rxrx1(args.data_path,
                    dataframe=df,
                    subset=args.cell_type,
                    split='train',
                    sample_diff_cell_type=args.sample_diff_cell_type,
                    channels=args.channels
                )
    # ============ collate function selection ============
    if args.easy_task:
        collate = tuple_channelnorm_collate_easy
    else:
        assert args.channels in [3, 6], "Only 3 and 6 channels are supported."
        collate = tuple_channelnorm_collate_6c if args.channels == 6 else tuple_channelnorm_collate_3c

    data_loader = torch.utils.data.DataLoader(
        dataset,
        #sampler=sampler,
        shuffle=True,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        collate_fn = collate,
        pin_memory=False,
        prefetch_factor=4,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            in_channels = args.channels,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            in_channels = args.channels,)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")
    # ============ custom weights loading ============
    if args.load_pretrained is not None:
        load_dino_weights(student, args.load_pretrained, checkpoint_key="student")
        load_dino_weights(teacher, args.load_pretrained, checkpoint_key="teacher")
    #=================================================
    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        print("Syncronizing batch norms....")
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    # student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    #teacher_without_ddp.load_state_dict(student.module.state_dict())
    teacher_without_ddp.load_state_dict(student.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")
    # ============ preparing loss ... ============
    examples_in_each_domain, mapping= get_samples_per_domain(args.metadata_path)
    if args.multi_center_training:
        print("Printing examples in each domain for multi centering...")
        print(examples_in_each_domain)
    if args.custom_loss:
        number_of_domains = examples_in_each_domain.shape[0]
        dino_loss = DINOLossMultiCenter(
            args.out_dim,
            args.multi_center_training, # multi-center training
            args.local_crops_number + 2,
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
            num_domains=number_of_domains,
            examples_in_each_domain=examples_in_each_domain,
            device_id="0",
            only_cross_domain=False, # still need to understand what is this
            dino_loss=True,
            barlow_loss=args.barlow_loss,
            barlow_loss_weight=args.barlow_loss_weight,
        ).cuda()
    else:
        dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups, lr=args.lr)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============

    # custom scheduling, linear warmup for 10 epochs then cosine annealing,
    # with 8 (by default, can be changed) accumulation steps to increase the effective batch size.
    # Basically these optimizers work AS IF the batch size was args.batch_size_per_gpu * args.acc_steps
    # This means that in the training loop, we need to call the steps only every args.acc_steps
    total_iterations = (args.epochs * len(data_loader)) // args.acc_steps
    warmup_iterations = (args.warmup_epochs * len(data_loader)) // args.acc_steps
    def warmup_lambda(iteration):
        if iteration < warmup_iterations:
            return float(iteration) / float(max(1, warmup_iterations))
        else:
            return 1.0
    lr_schedule_warmup = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    lr_schedule_cosine = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iterations - warmup_iterations, eta_min=args.min_lr)

    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader) // args.acc_steps,
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader) // args.acc_steps,)
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

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        print(f"Started epoch {epoch}", flush=True)
#        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule_warmup, lr_schedule_cosine, warmup_iterations, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, mapping, args)

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
        #utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth')
        utils.save_single_gpu(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
           # utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
           utils.save_single_gpu(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        #if utils.is_main_process():
        with (Path(args.output_dir) / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule_warmup,lr_schedule_cosine,warmup_iterations,wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, mapping, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for it, (images, _,metadata) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # global training iteration, not considering gradient accumulation
        it = len(data_loader) * epoch + it
        # update weight decay with original schedule
        optimizer.param_groups[0]["weight_decay"] = wd_schedule[it // args.acc_steps]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            # if we are using the code from the original github repo,
            # the forward pass of the loss needs the indexes of the domains
            # found in the batch to compute the centering of the teacher output
            if args.custom_loss:
                d1 = torch.tensor(get_batch_domains(metadata, mapping))
                d2 = torch.tensor(get_batch_domains(metadata, mapping))
                batch_domains = [d1,d2]
                loss, d_loss, b_loss = dino_loss(student_output, teacher_output, epoch, batch_domains)
                loss, d_loss, b_loss = loss/args.acc_steps, d_loss/args.acc_steps, b_loss/args.acc_steps

            # otherwise we just pass our own metadata and everything is done in the loss
            else:
                loss = dino_loss(student_output, teacher_output, epoch, metadata) / args.acc_steps

            wandb.log({"train_loss":loss.item()})
            if args.custom_loss:
                wandb.log({"barlow_loss":b_loss.item()})
                wandb.log({"dino_loss":d_loss.item()})

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # =============== Gradient accumulation =================
        if fp16_scaler is None:
            loss.backward()
        else:
            fp16_scaler.scale(loss).backward()
        # =============== Update student (every acc_steps) =========================
        if (it+1) % args.acc_steps == 0:
            param_norms = None
            if fp16_scaler is None:
                if args.clip_grad:
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                optimizer.step()
                optimizer.zero_grad()
            else:
                if args.clip_grad:
                    fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    param_norms = utils.clip_gradients(student, args.clip_grad)
                utils.cancel_gradients_last_layer(epoch, student,
                                                args.freeze_last_layer)
                fp16_scaler.step(optimizer)
                fp16_scaler.update()
                optimizer.zero_grad()
            # ============== Update learning rate =====================
            if (it // args.acc_steps) < warmup_iterations:
                lr_schedule_warmup.step()
                wandb.log({"lr":float(lr_schedule_warmup.get_lr()[0])})
            else:
                lr_schedule_cosine.step()
                wandb.log({"lr":float(lr_schedule_cosine.get_lr()[0])})
            # =============== Update momentum + EMA =========================
            # EMA update for the teacher
            with torch.no_grad():
                m = momentum_schedule[it // args.acc_steps]  # momentum parameter
                wandb.log({"teacher_momentum":float(m)})
                for param_q, param_k in zip(student.parameters(), teacher_without_ddp.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    """Original dino loss with an attempt of making it work with multiple centers.
    There is somehting wrong with the custom centering procedure, so it is not working properly.
    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs,student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        # ============ temperature setup ================
        self.register_buffer(f"center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch, metadata):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        targets = None
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)        
        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        #dist.all_reduce(batch_center)
        #batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        batch_center = batch_center / (len(teacher_output))
        #EMA update (same for both cases)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOLossMultiCenter(nn.Module):
    """Custom dino loss which handles multiple centers taken by the original paper."""
    def __init__(self, out_dim, multi_center_training, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, num_domains = None, examples_in_each_domain = None,
                 device_id = None, 
                 only_cross_domain = False, dino_loss = True,
                 barlow_loss = False, barlow_loss_weight = 0.2,
                 barlow_lambda_off_diag = 1e-3, 
                 barlow_loss_batch_com = False,
                 update_centering = True):
        
        super().__init__()
        self.iter = 0
        self.student_temp            = student_temp
        self.center_momentum         = center_momentum
        self.center_momentum_anti    = 1.0 - center_momentum
        self.ncrops                  = ncrops
        self.out_dim                 = out_dim
        self.dino_loss_scaling       = torch.log(torch.tensor(self.out_dim))
        self.num_domains             = num_domains
        self.examples_in_each_domain = examples_in_each_domain
        self.only_cross_domain       = only_cross_domain
        self.barlow_loss             = barlow_loss
        self.barlow_loss_batch_com   = barlow_loss_batch_com
        self.dino_loss               = dino_loss
        self.device_id               = device_id
        self.multi_center_training   = multi_center_training 
        self.barlow_lambda_off_diag  = barlow_lambda_off_diag 
        self.update_centering        = update_centering
        
        if (self.num_domains is not None) and self.multi_center_training:
            self.register_buffer("center", torch.zeros(self.num_domains, out_dim))
            self.domain_wise_centering = True
            self.examples_in_each_domain = torch.tensor([self.examples_in_each_domain],dtype=torch.float32).t().cuda()
        else:
            self.domain_wise_centering = False # Changes if num_domains is not none
            self.register_buffer("center", torch.zeros(1, out_dim))
            
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, int(warmup_teacher_temp_epochs)),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        if self.barlow_loss:
            self.barlow_loss_weight = barlow_loss_weight # defined in arguments, sets the scalar importance of the barlow-loss
            
            self.bn = nn.BatchNorm1d(self.out_dim, momentum=None, affine=False, track_running_stats=False)
            
            self.barlow_scaling_factor = 1.0*(1.0-self.barlow_lambda_off_diag) + 0.01*self.barlow_lambda_off_diag # just a scaling factor that makes sure the loss is 1 if doing poorly, expected value for on diagonal is 1.0 while off diag is 1
        
            
    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def sharpen_and_chunk_student_input(self,student_output):
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)
        return student_out

    def sharpen_and_chunk_teacher_input(self,teacher_output,domain_center,temp): # KEEP !
        teacher_centered = (teacher_output - domain_center) / temp
        teacher_out      = F.softmax(teacher_centered, dim=-1)
        teacher_out      = teacher_out.detach().chunk(2)
        teacher_centered = teacher_centered.detach().chunk(2)
        return teacher_out, teacher_centered
    
    def get_domainwise_centers(self,domain_belonging):
        domain_belonging = torch.cat(domain_belonging, 0).cuda(non_blocking=True)
        domain_center = self.get_centers(len(domain_belonging), self.center, self.num_domains, domain_belonging, self.out_dim)
        
        return domain_center, domain_belonging
    
    def calculate_barlow_loss(self,teacher_centered,student_out,i_t,i_s,batch_size,distribute_before_batch_norm=False):
        
        if distribute_before_batch_norm:
            
            if ddp_is_on():
                dist_teacher_centered   = dist_gather(teacher_centered[i_t], cat_dim=-1)
                synchronize() 
                dist_student_out = dist_gather(student_out[i_s], cat_dim=-1)
                synchronize()
            else:
                dist_teacher_centered = teacher_centered[i_t]
                dist_student_out = student_out[i_s]
                

            # Do we want centring as an ablation?
            # Should we use the centered teacher output?
            c = self.bn(dist_teacher_centered).T @ self.bn(dist_student_out)
        
            c.div_(batch_size)
            
            self.iter_loss_component += 1 
            # if self.iter % 50 == 0  and is_rank0():
                
            #     diag = torch.diagonal(c, 0).clone().detach().cpu().numpy()
            #     # wandb.log({f'barllow_diag-{self.iter_loss_component}-rank_{torch.cuda.current_device()}': diag}, 
            #     #           step=self.iter)                        

            on_diag  = torch.diagonal(c).add_(-1).pow_(2).mean()
            off_diag = self.off_diagonal(c).pow_(2).mean()
            return (on_diag*(1.0-self.barlow_lambda_off_diag) + self.barlow_lambda_off_diag * off_diag)
            
        else:

            c = self.bn(teacher_centered[i_t]).T @ self.bn(student_out[i_s]) 

            # sum the cross-correlation matrix between all gpus
            c.div_(batch_size)
            if ddp_is_on():
                dist.all_reduce(c)
                synchronize()     

            self.iter_loss_component +=1 
            # if self.iter % 50 == 0  and is_rank0():

            #     diag = torch.diagonal(c, 0).clone().detach().cpu().numpy()
            #     # wandb.log({f'barllow_diag-{self.iter_loss_component}-rank_{torch.cuda.current_device()}': diag}, 
            #     #           step=self.iter)                        

            on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
            off_diag = self.off_diagonal(c).pow_(2).mean()
            return  (on_diag*(1.0-self.barlow_lambda_off_diag) + self.barlow_lambda_off_diag * off_diag)
    
    def forward(self, student_output, teacher_output, epoch, domain_belonging=None):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        
        student_out = self.sharpen_and_chunk_student_input(student_output)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        
        if self.domain_wise_centering:
            domain_center, domain_belonging = self.get_domainwise_centers(domain_belonging)
            
            teacher_out,   teacher_centered = self.sharpen_and_chunk_teacher_input(teacher_output, domain_center, temp)
            
            
        else:
            domain_center = self.center

            teacher_out,   teacher_centered = self.sharpen_and_chunk_teacher_input(teacher_output, domain_center, temp)
        
        
        batch_size   = student_out[0].shape[0]
        batch_size   = sum(dist_gather(batch_size))
        if ddp_is_on():
            synchronize()
        barlow_loss  = 0
        dino_loss    = 0
        
        n_loss_terms = 0
        
        self.iter_loss_component = 0
        self.iter += 1 
        
        for i_t, q in enumerate(teacher_out):
            for i_s in range(len(student_out)):
                
                if i_s == i_t:
                    # we skip cases where student and teacher operate on the same view
                    continue
                elif (self.only_cross_domain) and ((i_t == 0 and i_s in [2,3,4]) or (i_t == 1 and i_s in [5,6,7])):
                    # if only doing cross domain learning, then skip views from the same image
                    continue
                
                n_loss_terms += 1
                if self.dino_loss:
                    loss = torch.sum(-q * F.log_softmax(student_out[i_s], dim=-1), dim=-1)
                    dino_loss += loss.mean()
                
                if self.barlow_loss:
                    barlow_loss += self.calculate_barlow_loss(teacher_centered,student_out,
                                                         i_t,i_s,batch_size,
                                                              distribute_before_batch_norm=self.barlow_loss_batch_com)
        if self.dino_loss:
            dino_loss  /= n_loss_terms 
            dino_loss  /= self.dino_loss_scaling
        if self.barlow_loss:
            barlow_loss /= n_loss_terms
            barlow_loss /= self.barlow_scaling_factor
        if self.barlow_loss and self.dino_loss:
            total_loss   = dino_loss*(1.0-self.barlow_loss_weight)+barlow_loss*(self.barlow_loss_weight) #Change back to this
        elif self.barlow_loss:
            total_loss   = barlow_loss
        else:
            total_loss   = dino_loss
        
        if self.update_centering:
            if self.domain_wise_centering:
                self.update_domain_wise_centers(len(domain_belonging),
                                                self.center,
                                                self.num_domains,
                                                domain_belonging,
                                                self.out_dim,
                                                teacher_output)
            else:
                self.update_center(teacher_output)
        
        # if self.iter % 10 == 0  and is_rank0():
        # wandb.log({'barlow_loss': barlow_loss, 'dino_loss': dino_loss})
        # if self.iter % 50 == 0:
        #     for cent in range(self.center.shape[0]):
        #         out = self.center[cent,:].clone().detach().cpu().numpy()
        #         wandb.log({f'centering_vector_domain-{cent}': out},
        #                     step=self.iter)
        return total_loss, dino_loss, barlow_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        batch_center = dist_average_tensor(batch_center)
        batch_center = batch_center / len(teacher_output)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    @torch.no_grad()
    def update_domain_wise_centers(self, batch_size, center_values, num_domains, domain_belong, width, teacher_output):
        ## Expand teacher out over the number of domains
        teac = teacher_output.expand((num_domains,-1,-1)).permute(1,0,2)

        ## Domain center selection tensor of equal size of center tensor. 
        ## Basically a index tensor to select the correct centering for each Mini-Batch sample
        
        eps = 1e-10
        ## Create zero matrix of shape batch_size x num_domains
        one_hot_label = torch.zeros(batch_size, num_domains).cuda(non_blocking=True)

        ## Make one hot vector of each row of the one_hot_vector_label matrix based on the domain_belonging
        one_hot_label = one_hot_label.scatter(dim=1, index=domain_belong.unsqueeze(0).T, value=1)#.type(torch.cuda.int64)

        ## Expand along the width to make the matrix point wise multiplicatable with the center tensor
        one_hot_label = one_hot_label.repeat(width,1,1).permute(1,2,0)

        ## Sum over the mini-batch dimention to get the sum of change to apply to the centering matrix. 
        ## Then sum the number of instances of such domain to know how to calculate 
        ## the mean and how important the centering should be considered
        
        sum_of_teacher_outs_for_dimention = (teac*one_hot_label).sum(0)
        num_of_teacher_outs_for_dimention = (one_hot_label).sum(0)
        
        if ddp_is_on():
            dist.all_reduce(sum_of_teacher_outs_for_dimention)
            synchronize()
            dist.all_reduce(num_of_teacher_outs_for_dimention)
            synchronize()
        
        update_centers = sum_of_teacher_outs_for_dimention /(num_of_teacher_outs_for_dimention+eps)

        weight = num_of_teacher_outs_for_dimention/(num_of_teacher_outs_for_dimention.sum(0))

        weight = weight/((self.examples_in_each_domain+eps)/self.examples_in_each_domain.sum())
                    
        update_proportion = self.center_momentum+self.center_momentum_anti*(1-weight)
        self.center = self.center.cuda() * update_proportion + update_centers * self.center_momentum_anti*weight

            
    def get_centers(self,batch_size,center_values,num_domains,domain_belong,width):

        ## Expand domain centers to third dimention covering the mini-batch size

        cent = center_values.expand((batch_size,-1,-1)).cuda(non_blocking=True)

        ## Domain center selection tensor of equal size of cent tensor. Basically a index tensor to select the correct centering for each Mini-Batch sample
        ## Create zero matrix of shape batch_size x num_domains
        one_hot_label = torch.zeros(batch_size, num_domains).cuda(non_blocking=True)

        ## Make one hot vector of each row of the one_hot_vector_label matrix based on the domain_belonging
        one_hot_label = one_hot_label.scatter(dim=1, index=domain_belong.unsqueeze(0).T, value=1)#.type(torch.cuda.int64)

        ## Expand along the width to make the matrix point wise multiplicatable with the center tensor
        
        one_hot_label = one_hot_label.repeat(width,1,1).permute(1,2,0)

        ## Sum over the num_domains dimension to get the corresponding center value. The sum should be over n-1 zero values and one non zero representing the image batch belongings index and correponding center value.
        
        centers_to_use_for_domain_aligning = (cent*one_hot_label).sum(1)
        return centers_to_use_for_domain_aligning


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
