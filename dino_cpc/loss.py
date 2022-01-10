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
        # student_out = student_out.chunk(self.ncrops) # ALMOG: no need to chunks here because we give it what we want it to compare

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

    def _handle_flips(self, flips: torch.Tensor, pred: List[torch.Tensor], bbox: Optional[torch.Tensor] = None,
                      img_size: Optional[torch.Tensor] = None):
        for i in range(len(pred)):
            pred[i][flips] = torch.flip(pred[i][flips], [2])

        if bbox is not None:
            flipped_bbox = torch.stack(
                [img_size[:, 0] - bbox[:, 2], bbox[:, 1], img_size[:, 0] - bbox[:, 0], bbox[:, 3]], dim=1)
            bbox[flips] = flipped_bbox[flips]
            return pred, bbox
        else:
            return pred

    def __call__(self,
                 student_pred: List[torch.Tensor],
                 teacher_pred: List[torch.Tensor],
                 bboxes: torch.Tensor,
                 flips: torch.Tensor,
                 orig_img_size: torch.Tensor,
                 epoch: int,
                 args: argparse.Namespace
                 ):
        loss = 0

        # Handle the flips
        global_flips = flips[:, :2].reshape(-1)  # First 2 are the global
        local_flip = flips[:, 2:].reshape(-1)
        global_bbox = bboxes[:, :2, :].reshape(-1, 4)  # First 2 are the global boxes
        local_bbox = bboxes[:, 2:, :].reshape(-1, 4)  # The rest are local
        global_student_pred = student_pred[:len(teacher_pred)]
        local_student_pred = student_pred[len(teacher_pred):]
        global_img_size = orig_img_size[:, :2, :].reshape(-1, 2)
        local_img_size = orig_img_size[:, 2:, :].reshape(-1, 2)

        local_student_pred, local_bbox = self._handle_flips(flips=local_flip,
                                                            bbox=local_bbox,
                                                            pred=local_student_pred,
                                                            img_size=local_img_size)
        global_student_pred, global_bbox = self._handle_flips(flips=global_flips,
                                                              bbox=global_bbox,
                                                              pred=global_student_pred,
                                                              img_size=global_img_size)
        teacher_pred = self._handle_flips(flips=global_flips,
                                          pred=teacher_pred)
        # Now all the predictions has the right coordinate system and we can start comparing them

        # Reshape inputs so they will have batch dim
        batch_size = args.batch_size_per_gpu

        local_student_pred = [pred.view(batch_size, -1, pred.size(1), pred.size(2), pred.size(3)) for pred in
                              local_student_pred]
        global_student_pred = [pred.view(batch_size, -1, pred.size(1), pred.size(2), pred.size(3)) for pred in
                               global_student_pred]
        teacher_pred = [pred.view(batch_size, -1, pred.size(1), pred.size(2), pred.size(3)) for pred in
                        teacher_pred]
        global_bbox = global_bbox.view(batch_size, -1, 4)
        local_bbox = local_bbox.view(batch_size, -1, 4)

        # Now we check which local boxes matches the global boxes and compute the loss between them as well
        for lvl_idx, num_pairs_to_grab in enumerate(args.max_pairs):
            matches_local_global = PatchMatcher.find_matches(crop_a=local_bbox,
                                                             crop_size_a=args.local_crop_size,
                                                             num_patches_a=local_student_pred[lvl_idx].size(2),
                                                             crop_b=global_bbox,
                                                             crop_size_b=args.global_crop_size,
                                                             num_patches_b=global_student_pred[lvl_idx].size(2))

            matches_global_global = PatchMatcher.find_matches(crop_a=global_bbox,
                                                              crop_size_a=args.global_crop_size,
                                                              num_patches_a=global_student_pred[lvl_idx].size(2),
                                                              crop_b=global_bbox,
                                                              crop_size_b=args.global_crop_size,
                                                              num_patches_b=global_student_pred[lvl_idx].size(2))
            # Remove all global matches which matches the same view, because its not interesting
            matches_global_global = matches_global_global[matches_global_global[:, 0] != matches_global_global[:, 2]]

            for curr_matches, pred_student, pred_teacher in ((matches_local_global, local_student_pred[lvl_idx], teacher_pred[lvl_idx]),
                                                             (matches_global_global, global_student_pred[lvl_idx], teacher_pred[lvl_idx])):
                perm = torch.randperm(curr_matches.size(0))
                idx = perm[:args.max_pairs[lvl_idx]]
                curr_matches = curr_matches[idx]
                relevant_pred_idx_student = curr_matches[:, :2]
                relevant_pred_idx_teacher = curr_matches[:, 2:]

                pred_student = pred_student.view(pred_student.size(0) * pred_student.size(1), -1, pred_student.size(-1)) # [BOX_ID, PATCH_ID, PATCH_EMB]
                pred_teacher = pred_teacher.view(pred_teacher.size(0) * pred_teacher.size(1), -1, pred_teacher.size(-1))

                # Prepare the indexing
                relevant_pred_idx_student = relevant_pred_idx_student[:, 0] * pred_student.size(1) + \
                                            relevant_pred_idx_student[:, 1]
                relevant_pred_idx_teacher = relevant_pred_idx_teacher[:, 0] * pred_teacher.size(1) + \
                                            relevant_pred_idx_teacher[:, 1]
                pred_student = pred_student.view(-1, pred_student.size(2))[relevant_pred_idx_student]
                pred_teacher = pred_teacher.view(-1, pred_teacher.size(2))[relevant_pred_idx_teacher]

                loss += self.dino_loss[lvl_idx](student_output=pred_student,
                                        teacher_output=pred_teacher,
                                        epoch=epoch)

        return loss