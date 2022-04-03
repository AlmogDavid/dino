import argparse
from typing import List, Dict, Optional
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
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
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

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

    def __init__(self, args: argparse.Namespace, num_features: List[int]):
        super().__init__()
        all_out_dim = list(args.out_dim)
        if args.multi_level_matching:
            all_out_dim += all_out_dim[:-1]
            num_features += num_features[:-1]
        self.dino_loss = torch.nn.ModuleList(DINOLossCPCSingle(
            out_dim,
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        ).cuda() for out_dim in all_out_dim)
        self.global_dino_loss = None
        if args.add_global_dino_loss:
            self.global_dino_loss = torch.nn.ModuleList(DINOLoss(
                out_dim,
                args.local_crops_number + 2,  # 2 global
                args.warmup_teacher_temp,
                args.teacher_temp,
                args.warmup_teacher_temp_epochs,
                args.epochs,
            ).cuda() for out_dim in num_features)
        self.max_pairs = args.max_pairs

    def _prepare_to_global_loss(self, lvl_pred: torch.Tensor, num_views: int) -> torch.Tensor:
        pooled_pred = F.avg_pool2d(lvl_pred, lvl_pred.size(-1))
        pooled_pred = [t.squeeze(1) for t in torch.split(pooled_pred.reshape(-1, num_views, pooled_pred.size(1)), 1, dim=1)]
        return pooled_pred

    def __call__(self,
                 global_match_student_pred: List[torch.Tensor],
                 local_match_student_pred: List[torch.Tensor],
                 local_match_teacher_pred: List[torch.Tensor],
                 global_match_teacher_pred: List[torch.Tensor],
                 teacher_pred: torch.Tensor,
                 student_pred: torch.Tensor,
                 epoch: int,
                 args: argparse.Namespace
                 ):
        loss = 0
        sep_loss = []
        global_loss = []
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

            if self.global_dino_loss is not None:
                for i in range(len(global_match_teacher_pred)):
                    curr_teacher_pred = self._prepare_to_global_loss(teacher_pred[0][i], 2)
                    curr_student_pred = [student_pred[1][i], student_pred[0][i]]
                    num_local_views = curr_student_pred[1].size(0) // curr_teacher_pred[0].size(0)
                    curr_student_pred = self._prepare_to_global_loss(curr_student_pred[0], 2) + self._prepare_to_global_loss(curr_student_pred[1], num_local_views)

                    curr_global_loss = self.global_dino_loss[i](student_output=torch.stack(curr_student_pred),
                                                                teacher_output=torch.stack(curr_teacher_pred),
                                                                epoch=epoch)
                    loss += curr_global_loss
                    global_loss.append(curr_global_loss.detach().cpu().item())

        loss = loss / len(global_match_teacher_pred) # Mean over all the levels
        return loss, sep_loss, global_loss
