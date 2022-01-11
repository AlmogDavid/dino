import argparse
from typing import List, Dict, Optional
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from dino_cpc.utils import PatchMatcher



class DINOLossCPCSingle(nn.Module):
    def __init__(self, out_dim, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training unstable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach()  # stop grad

        loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1).mean()
        self.update_center(teacher_output)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINOLossCPC(nn.Module):

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.dino_loss = torch.nn.ModuleList(DINOLossCPCSingle(
            out_dim,
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda() for out_dim in args.out_dim)
        self.max_pairs = args.max_pairs

    def __call__(self,
                 global_match_student_pred: List[torch.Tensor],
                 local_match_student_pred: List[torch.Tensor],
                 local_match_teacher_pred: List[torch.Tensor],
                 global_match_teacher_pred: List[torch.Tensor],
                 epoch: int,
                 args: argparse.Namespace
                 ):
        loss = 0
        sep_loss = []
        for i in range(len(global_match_teacher_pred)):
            curr_lvl_loss = 0
            for student_p, teacher_p in ((global_match_student_pred[i], global_match_teacher_pred[i]),
                                         (local_match_student_pred[i], local_match_teacher_pred[i])):
                curr_loss = self.dino_loss[i](student_output=student_p,
                                        teacher_output=teacher_p,
                                        epoch=epoch)
                curr_lvl_loss += curr_loss
            curr_lvl_loss = curr_lvl_loss / 2 # Mean between the global global and local global
            loss += curr_lvl_loss
            sep_loss.append(curr_lvl_loss.detach().cpu().item())

        loss = loss / len(global_match_teacher_pred) # Mean over all the levels
        return loss, sep_loss