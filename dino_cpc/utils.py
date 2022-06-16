from typing import List, Optional
import torch


def handle_flips(flips: torch.Tensor, pred: List[torch.Tensor]):
    for i in range(len(pred)):
        pred[i][flips] = torch.flip(pred[i][flips], [2])

    return pred


class PatchMatcher:
    """
    Helper class for matching patches between different crops
    """

    @staticmethod
    def _get_intersection_with_patches(box: torch.Tensor, num_patches: int, valid_boxes_idx: torch.Tensor) -> torch.Tensor:
        patch_size_normed = 1/num_patches
        start_patch_x = torch.floor(box[:, 0] / patch_size_normed)
        end_patch_x = torch.ceil(box[:, 2] / patch_size_normed)
        start_patch_y = torch.floor(box[:, 1] / patch_size_normed)
        end_patch_y = torch.ceil(box[:, 3] / patch_size_normed)
        all_patches = []
        for bbox_id, orig_box_id in enumerate(valid_boxes_idx):
            relevant_patches_xy = torch.cartesian_prod(torch.arange(start_patch_x[bbox_id], end_patch_x[bbox_id]), torch.arange(start_patch_y[bbox_id], end_patch_y[bbox_id]))
            relevant_patches_idx = relevant_patches_xy[:, 1] * num_patches + relevant_patches_xy[:, 0]
            box_id = torch.empty(relevant_patches_idx.size(0)).fill_(orig_box_id).to(relevant_patches_idx)
            all_patches.append(torch.stack([box_id, relevant_patches_idx], dim=1))
        all_patches = torch.cat(all_patches)
        return all_patches

    @staticmethod
    def _get_intersection_bbox(box_a: torch.Tensor,
                               box_b: torch.Tensor,
                               crop_size_a: float,
                               crop_size_b: float):
        batch_size = box_a.size(0)
        box_a = box_a.view(-1, 4)
        box_b = box_b.view(-1, 4)

        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))

        inter_boxes = torch.cat([min_xy, max_xy], dim=2)
        box_a_coord = inter_boxes - box_a[:, :2].unsqueeze(1).repeat([1, B, 2])
        box_a_coord_norm = box_a_coord / crop_size_a
        box_b_coord = inter_boxes - box_b[:, :2].unsqueeze(0).repeat([A, 1, 2])
        box_b_coord_norm = box_b_coord / crop_size_b

        non_zero_cond = (max_xy - min_xy) > 0
        non_empty_boxes = torch.logical_and(non_zero_cond[:, :, 0], non_zero_cond[:, : , 1])
        num_crops_image_a = box_a.size(0) // batch_size
        num_crops_image_b = box_b.size(0) // batch_size
        same_image_cond = torch.block_diag(*[torch.ones((num_crops_image_a, num_crops_image_b)) for _ in range(batch_size)]).to(non_empty_boxes)
        valid_boxes_cond = torch.logical_and(non_empty_boxes, same_image_cond)

        valid_boxes_idx_a, valid_boxes_idx_b = torch.nonzero(valid_boxes_cond, as_tuple=True)
        box_a_coord_norm = box_a_coord_norm[valid_boxes_cond]
        box_b_coord_norm = box_b_coord_norm[valid_boxes_cond]
        return box_a_coord_norm, box_b_coord_norm, valid_boxes_idx_a, valid_boxes_idx_b  # This might be with zero dim if there are no intersections, we assume there is always at least one intersection

    @staticmethod
    def find_matches(crop_a: torch.Tensor,
                     crop_size_a: int,
                     num_patches_a: int,
                     crop_b: torch.Tensor,
                     num_patches_b: int,
                     crop_size_b: int) -> torch.Tensor:
        # 1. Compute the intersection in the original image between all bboxes
        a_crop_coord_norm, b_crop_coord_norm, valid_boxes_idx_a, valid_boxes_idx_b = PatchMatcher._get_intersection_bbox(crop_a,
                                                                           crop_b,
                                                                           crop_size_a,
                                                                           crop_size_b)

        # 2. Now find out what are the patches indices for the relevant intersections
        a_crop_emb_patches = PatchMatcher._get_intersection_with_patches(a_crop_coord_norm, num_patches_a, valid_boxes_idx_a)
        b_crop_emb_patches = PatchMatcher._get_intersection_with_patches(b_crop_coord_norm, num_patches_b, valid_boxes_idx_b)

        assert a_crop_emb_patches.shape == b_crop_emb_patches.shape

        # Now: a_crop_emb_patches[:, 0] == b_crop_emb_patches[:, 0]
        matches = torch.cat([a_crop_emb_patches, b_crop_emb_patches], dim=1)
        return matches.type(torch.LongTensor)  # <first_box_id, first_patch_index, second_box_id, second_patch_index>
