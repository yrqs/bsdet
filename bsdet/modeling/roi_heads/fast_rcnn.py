"""Implement the CosineSimOutputLayers and  FastRCNNOutputLayers with FC layers."""

import torch
import logging
import numpy as np
from torch import nn
from torch.nn import functional as F
from fvcore.nn import smooth_l1_loss
from detectron2.utils.registry import Registry
from detectron2.layers import batched_nms, cat
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

ROI_HEADS_OUTPUT_REGISTRY = Registry("ROI_HEADS_OUTPUT")
ROI_HEADS_OUTPUT_REGISTRY.__doc__ = """
Registry for the output layers in ROI heads in a generalized R-CNN model."""

logger = logging.getLogger(__name__)

from fvcore.nn import sigmoid_focal_loss_jit
from ..meta_arch.gdl import decouple_layer, AffineLayer
from sklearn import manifold
from ..meta_arch.mlp import MLP

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
        boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputs.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputs.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image,
            scores_per_image,
            image_shape,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )
        for scores_per_image, boxes_per_image, image_shape in zip(
            scores, boxes, image_shapes
        )
    ]
    return tuple(list(x) for x in zip(*result_per_image))


def fast_rcnn_inference_single_image(
        boxes, scores, image_shape, score_thresh, nms_thresh, topk_per_image
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # Apply per-class NMS
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    return result, filter_inds[:, 0]


def py_focal_loss(pred,
                  target,
                  gamma=2.0,
                  alpha=0.25):
    # pred_sigmoid = pred.sigmoid()
    pred_sigmoid = pred
    target = target.type_as(pred)

    # pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # focal_weight = (alpha * target + (1 - alpha) *
    #                 (1 - target)) * pt.pow(gamma)
    # loss = F.binary_cross_entropy(
    #     pred, target, reduction='none') * focal_weight

    ce_loss = F.binary_cross_entropy(pred, target, reduction='none')
    p_t = pred_sigmoid * target + (1 - pred_sigmoid) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss

    return loss


def calculate_iou(bboxes1, bboxes2):
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])

    left_line = torch.cat([bboxes1[:, 0][:, None], bboxes2[:, 0][:, None]], dim=1).max(1)[0]
    top_line = torch.cat([bboxes1[:, 1][:, None], bboxes2[:, 1][:, None]], dim=1).max(1)[0]
    right_line = torch.cat([bboxes1[:, 2][:, None], bboxes2[:, 2][:, None]], dim=1).min(1)[0]
    bottom_line = torch.cat([bboxes1[:, 3][:, None], bboxes2[:, 3][:, None]], dim=1).min(1)[0]

    zero_ind = (left_line >= right_line) | (top_line >= bottom_line)

    intersect = (right_line - left_line) * (bottom_line - top_line)
    ious = intersect.float() / (area1 + area2 - intersect).float()
    ious[zero_ind] = 0.
    return ious


class FastRCNNOutputs(object):
    """
    A class that stores information about outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta,
            use_triplet=False,
            use_PN_triplet=False,
            triplet_margin=0.15,
            use_neg_triplet=False,
            triplet_neg_margin=0.1,
            neg_thresh=0.1,
            metric_type='cos_sim',
            fl_gamma=2.0,
            fl_alpha=0.25,
            head_type=None,
            cls_loss_type='focal_loss',
            neg_iou_thresh=(0.3, 0.4),
            reg_loss_weight=1.0,
            extra=None,
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta

        self.objectness_logits = torch.cat([p.objectness_logits for p in proposals])

        if 'matched_ious' in proposals[0].get_fields().keys():
            self.matched_ious = torch.cat([p.matched_ious for p in proposals])
            self.matched_labels = torch.cat([p.matched_labels for p in proposals])

        box_type = type(proposals[0].proposal_boxes)
        # cat(..., dim=0) concatenates over all images in the batch
        self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
        assert (
            not self.proposals.tensor.requires_grad
        ), "Proposals should not require gradients!"
        self.image_shapes = [x.image_size for x in proposals]

        # The following fields should exist only when training.
        if proposals[0].has("gt_boxes"):
            self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
            assert proposals[0].has("gt_classes")
            self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)

        self.use_triplet = use_triplet
        self.triplet_margin = triplet_margin
        self.extra = extra
        self.use_neg_triplet = use_neg_triplet
        self.neg_triplet_margin = triplet_neg_margin
        self.neg_thresh = neg_thresh
        self.metric_type = metric_type
        self.fl_alpha = fl_alpha
        self.fl_gamma = fl_gamma
        self.use_PN_triplet = use_PN_triplet
        self.head_type = head_type
        self.cls_loss_type = cls_loss_type
        self.neg_iou_thresh = neg_iou_thresh
        self.reg_loss_weight = reg_loss_weight

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (
            (fg_pred_classes == bg_class_ind).nonzero().numel()
        )
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        storage.put_scalar(
            "fast_rcnn/cls_accuracy", num_accurate / num_instances
        )
        if num_fg > 0:
            storage.put_scalar(
                "fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg
            )
            storage.put_scalar(
                "fast_rcnn/false_negative", num_false_negative / num_fg
            )

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self.head_type in ('FastRCNNOutputLayers', 'CosSimHeadRM', 'CosSimHeadTFA'):
            self._log_accuracy()
            return F.cross_entropy(
                self.pred_class_logits, self.gt_classes, reduction="mean"
            )

        gt_labels_target = F.one_hot(self.gt_classes, num_classes=self.pred_class_logits.size(1)).float()[:, :-1]

        # loss_cls = sigmoid_focal_loss_jit(
        #     self.pred_class_logits[:, :-1],
        #     gt_labels_target.to(self.pred_class_logits.dtype),
        #     alpha=self.fl_alpha,
        #     gamma=self.fl_gamma,
        # )

        loss_cls = py_focal_loss(
            # self.extra['scores_ori'],
            self.pred_class_logits[:, :-1],
            gt_labels_target.to(self.pred_class_logits.dtype),
            alpha=self.fl_alpha,
            gamma=self.fl_gamma,
        )

        loss_cls = loss_cls.sum()

        num_pos = ((self.gt_classes >= 0) & (self.gt_classes < self.pred_class_logits.size(1) - 1)).sum()
        if num_pos == 0:
            num_pos = self.gt_classes.size(0) * 0.2

        return loss_cls / num_pos

    def focal_loss(self):
        gt_classes = self.gt_classes.clone()

        gt_labels_target = F.one_hot(gt_classes, num_classes=self.pred_class_logits.size(1)).float()[:, :-1].to(
            self.pred_class_logits.dtype)

        loss_cls = py_focal_loss(
            self.pred_class_logits[:, :-1],
            gt_labels_target,
            alpha=self.fl_alpha,
            gamma=self.fl_gamma,
        )
        loss_cls = loss_cls.sum()

        num_pos = ((gt_classes >= 0) & (gt_classes < self.pred_class_logits.size(1) - 1)).sum()
        if num_pos == 0:
            num_pos = gt_classes.size(0) * 0.2

        return loss_cls / num_pos

    def smooth_l1_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        gt_proposal_deltas = self.box2box_transform.get_deltas(
            self.proposals.tensor, self.gt_boxes.tensor
        )
        box_dim = gt_proposal_deltas.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = torch.nonzero(
            (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        ).squeeze(1)
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(
                box_dim, device=device
            )
        loss_box_reg = smooth_l1_loss(
            self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
            gt_proposal_deltas[fg_inds],
            self.smooth_l1_beta,
            reduction="sum",
            # reduction="mean",
        )
        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def triplet_loss(self, iou_thresh=0.5):
        num_fg_classes = self.pred_class_logits.shape[1] - 1
        pos_inds = (self.gt_classes >= 0) & (self.gt_classes < num_fg_classes) & (self.matched_ious >= iou_thresh)
        if pos_inds.sum() == 0:
            return torch.zeros(1).to(self.pred_class_logits.device)
        gt_classes_pos = self.gt_classes[pos_inds]
        gt_classes_pos_fg = F.one_hot(gt_classes_pos, num_classes=num_fg_classes).bool()
        gt_classes_pos_bg = ~gt_classes_pos_fg
        if self.metric_type == 'cos_sim':
            cos_sim = self.extra['cos_sim']
            cos_sim_pos = cos_sim[pos_inds]
            cos_sim_pos_fg = cos_sim_pos[gt_classes_pos_fg]
            cos_sim_pos_bg = cos_sim_pos[gt_classes_pos_bg].reshape(cos_sim_pos_fg.shape[0], -1)
            loss_triplet = F.relu(cos_sim_pos_bg.max(dim=1)[0] - cos_sim_pos_fg + self.triplet_margin).mean()
        else:
            loss_triplet = torch.zeros(1).to(self.pred_class_logits.device)

        return loss_triplet

    def softmax_cls_loss(self):
        softmax_scores = self.extra['softmax_scores']
        pos_inds = (self.gt_classes >= 0) & (self.gt_classes < self.pred_class_logits.size(1) - 1) & (
                    self.matched_ious > 0.7)
        fg_gt_classes = self.gt_classes[pos_inds].clone()
        # fg_softmax_scores = softmax_scores[pos_inds, :-1]
        fg_softmax_scores = softmax_scores[pos_inds, :6]
        # fg_softmax_scores = softmax_scores[pos_inds, :16]
        # fg_gt_classes[fg_gt_classes >= 15] = 15
        fg_gt_classes[fg_gt_classes < 15] = 0
        fg_gt_classes[fg_gt_classes >= 15] = fg_gt_classes[fg_gt_classes >= 15] - 14
        # fg_gt_classes[fg_gt_classes > 0] = fg_gt_classes[fg_gt_classes > 0] - 14
        return F.cross_entropy(fg_softmax_scores, fg_gt_classes, reduction="mean")
        # return F.cross_entropy(softmax_scores, self.gt_classes, reduction="mean") * 0.25

    def CosSimNegHead_loss(self):
        def neg_triplet_loss(iou_thresh=0.7):
            num_classes = self.pred_class_logits.size(1) - 1
            cos_sim = self.extra['cos_sim']
            neg_cos_sim = self.extra['neg_cos_sim']
            neg_scores = self.extra['neg_scores']

            cos_sim_cat = torch.cat([cos_sim[:, :, None], neg_cos_sim[:, :, None]], dim=-1)
            with torch.no_grad():
                hn_inds = (self.matched_ious > self.neg_iou_thresh[0]) & (
                        self.matched_ious < self.neg_iou_thresh[1]) & (self.gt_classes == num_classes)
            neg_labels = self.matched_labels.clone().detach()
            neg_labels[~hn_inds] = num_classes
            num_hn = hn_inds.sum()

            if num_hn > 0:
                hn_labels = neg_labels[hn_inds]
                hn_labels_oh = F.one_hot(hn_labels, num_classes=num_classes).bool()
                cos_sim_cat_hn = cos_sim_cat[hn_inds]
                cos_sim_cat_hn = cos_sim_cat_hn[hn_labels_oh]
                loss_hn_neg = F.relu(cos_sim_cat_hn[:, 0] - cos_sim_cat_hn[:, 1] + self.neg_triplet_margin).mean()
            else:
                loss_hn_neg = torch.zeros(1).to(self.pred_class_logits.device)

            pos_inds = (self.gt_classes >= 0) & (self.gt_classes < num_classes) & (self.matched_ious >= iou_thresh)

            if pos_inds.sum() > 0:
                labels_pos = self.gt_classes[pos_inds]
                labels_pos_one_hot = F.one_hot(labels_pos.flatten(0), num_classes=num_classes).bool()
                cos_sim_cat_pos = cos_sim_cat[pos_inds][labels_pos_one_hot]
                loss_hn_pos = F.relu(cos_sim_cat_pos[:, 1] - cos_sim_cat_pos[:, 0] + self.neg_triplet_margin).mean()
                del labels_pos, labels_pos_one_hot, cos_sim_cat_pos
            else:
                loss_hn_pos = torch.zeros(1).to(cos_sim.device)

            hn_neg_inds = ((self.matched_ious >= iou_thresh) | (self.matched_ious <= 0.05)) & (neg_labels == num_classes)
            scores = torch.cat([neg_scores[hn_inds], neg_scores[hn_neg_inds]], dim=0)
            targets = F.one_hot(torch.cat([neg_labels[hn_inds], neg_labels[hn_neg_inds]], dim=0),
                                num_classes=num_classes + 1).float()[:, :-1]

            loss_neg_cls = py_focal_loss(
                scores,
                targets.to(self.pred_class_logits.dtype),
                alpha=self.fl_alpha,
                gamma=self.fl_gamma,
            )

            avg_scale = hn_inds.sum().item()
            if avg_scale == 0:
                avg_scale = scores.size(0) * 0.2
            del hn_neg_inds, scores, targets

            loss_neg_cls_hn = loss_neg_cls[:num_hn].sum()
            loss_neg_cls_hn_neg = loss_neg_cls[num_hn:].sum()

            loss_neg_cls_hn = loss_neg_cls_hn / avg_scale
            loss_neg_cls_hn_neg = loss_neg_cls_hn_neg / avg_scale

            return loss_hn_pos, loss_hn_neg, loss_neg_cls_hn, loss_neg_cls_hn_neg

        def cls_loss():
            gt_classes = self.gt_classes.clone()
            num_classes = self.pred_class_logits.size(1) - 1
            gt_labels_target = F.one_hot(gt_classes, num_classes=self.pred_class_logits.size(1)).float()[:, :-1].to(
                self.pred_class_logits.dtype)
            pos_inds = ((gt_classes >= 0) & (gt_classes < num_classes))
            neg_inds = (gt_classes == num_classes)
            scores = self.pred_class_logits[:, :-1]
            scores_pos = scores[pos_inds]
            scores_neg = scores[neg_inds]
            gt_labels_target_pos = gt_labels_target[pos_inds]
            gt_labels_target_neg = gt_labels_target[neg_inds]
            loss_cls = py_focal_loss(
                torch.cat([scores_pos, scores_neg], dim=0),
                torch.cat([gt_labels_target_pos, gt_labels_target_neg], dim=0),
                alpha=self.fl_alpha,
                gamma=self.fl_gamma,
            )
            loss_cls_pos = loss_cls[:pos_inds.sum()].sum()
            loss_cls_neg = loss_cls[-neg_inds.sum():].sum()
            num_pos = ((gt_classes >= 0) & (gt_classes < num_classes)).sum()
            if num_pos == 0:
                num_pos = gt_classes.size(0) * 0.2

            return loss_cls_pos / num_pos, loss_cls_neg / num_pos

        # The positive and negative losses are returned separately only for the convenience of observing the loss of each positive and negative sample.
        loss_cls_pos, loss_cls_neg = cls_loss()
        losses_dict = {
            "loss_cls_pos": loss_cls_pos,
            "loss_cls_neg": loss_cls_neg,
            "loss_box_reg": self.smooth_l1_loss() * self.reg_loss_weight,
        }
        if self.use_triplet:
            losses_dict["loss_triplet"] = self.triplet_loss(iou_thresh=0.5)
        if self.use_neg_triplet:
            loss_hn_pos, loss_hn_neg, loss_neg_cls_hn, loss_neg_cls_hn_neg = neg_triplet_loss(iou_thresh=0.7)
            losses_dict["loss_hn_pos"] = loss_hn_pos
            losses_dict["loss_hn_neg"] = loss_hn_neg
            losses_dict["loss_neg_cls_hn"] = loss_neg_cls_hn
            losses_dict["loss_neg_cls_hn_neg"] = loss_neg_cls_hn_neg

        return losses_dict

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """

        if self.head_type == 'CosSimNegHead':
            return self.CosSimNegHead_loss()

        # baseline: FastRCNNOutputLayers
        losses_dict = {
            "loss_cls": self.softmax_cross_entropy_loss(),
            "loss_box_reg": self.smooth_l1_loss(),
        }
        return losses_dict

    def predict_boxes(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        num_pred = len(self.proposals)
        B = self.proposals.tensor.shape[1]
        K = self.pred_proposal_deltas.shape[1] // B
        boxes = self.box2box_transform.apply_deltas(
            self.pred_proposal_deltas.reshape(num_pred * K, B),
            self.proposals.tensor.unsqueeze(1)
                .expand(num_pred, K, B)
                .reshape(-1, B),
        )
        return boxes.reshape(num_pred, K * B).split(
            self.num_preds_per_image, dim=0
        )

    def predict_probs(self):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        if self.head_type in ('FastRCNNOutputLayers', ):
            probs = F.softmax(self.pred_class_logits, dim=-1)
        else:
            probs = self.pred_class_logits
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Args:
            score_thresh (float): same as fast_rcnn_inference.
            nms_thresh (float): same as fast_rcnn_inference.
            topk_per_image (int): same as fast_rcnn_inference.
        Returns:
            list[Instances]: same as fast_rcnn_inference.
            list[Tensor]: same as fast_rcnn_inference.
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes

        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            score_thresh,
            nms_thresh,
            topk_per_image,
        )

@ROI_HEADS_OUTPUT_REGISTRY.register()
class CosSimNegHead(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
            self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(CosSimNegHead, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        self.cos_scale = cfg.MODEL.ROI_HEADS.COS_SCALE
        self.cos_power = cfg.MODEL.ROI_HEADS.COS_POWER
        self.neg_mode = cfg.MODEL.ROI_HEADS.NEG_MODE
        self.neg_weight = cfg.MODEL.ROI_HEADS.NEG_WEIGHT
        self.neg_cos_iter = cfg.MODEL.ROI_HEADS.NEG_COS_ITER
        self.neg_channels = cfg.MODEL.ROI_HEADS.NEG_CHANNELS
        self.neg_rep_type = cfg.MODEL.ROI_HEADS.NEG_REP_TYPE
        self.neg_emb_type = cfg.MODEL.ROI_HEADS.NEG_EMB_TYPE
        self.neg_input_detach = cfg.MODEL.ROI_HEADS.NEG_INPUT_DETACH
        self.hn_selection = cfg.MODEL.ROI_HEADS.HN_SELECTION
        self.base_novel_cls_loss = cfg.MODEL.ROI_HEADS.BASE_NOVEL_CLS_LOSS
        self.num_base_classes = cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES
        self.reg_fc_channels = cfg.MODEL.ROI_HEADS.REG_FC_CHANNELS
        self.add_reg_fc = cfg.MODEL.ROI_HEADS.ADD_REG_FC
        self.num_base_classes = cfg.MODEL.ROI_HEADS.NUM_BASE_CLASSES
        self.cls_fc_channels = cfg.MODEL.ROI_HEADS.CLS_FC_CHANNELS
        self.add_cls_fc = cfg.MODEL.ROI_HEADS.ADD_CLS_FC

        self.grad_scale = cfg.MODEL.ROI_HEADS.BACKWARD_SCALE

        if self.base_novel_cls_loss:
            self.bn_cls_head = nn.Linear(input_size, 2)

        self.reps = nn.Linear(input_size, num_classes + 1)
        if num_classes == 15:
            self.neg_reps = nn.Linear(input_size, 15 * self.neg_mode, bias=False)
        elif num_classes == 60:
            self.neg_reps = nn.Linear(input_size, 60 * self.neg_mode, bias=False)
        else:
            self.neg_reps = nn.Linear(input_size, num_classes * self.neg_mode, bias=False)
        nn.init.normal_(self.neg_reps.weight, std=0.01)

        if self.add_reg_fc:
            self.reg_fc = nn.Linear(input_size, self.reg_fc_channels)
            nn.init.normal_(self.reg_fc.weight, std=0.01)
            nn.init.constant_(self.reg_fc.bias, 0)

        if self.add_cls_fc:
            self.cls_fc = nn.Linear(input_size, self.cls_fc_channels)
            nn.init.normal_(self.cls_fc.weight, std=0.01)
            nn.init.constant_(self.cls_fc.bias, 0)

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(self.reg_fc_channels, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.reps.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.reps, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.current_iter = 0
        self.num_classes = num_classes

        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)

        if self.add_reg_fc:
            proposal_deltas = self.bbox_pred(self.reg_fc(x))
        else:
            proposal_deltas = self.bbox_pred(x)

        if self.add_cls_fc:
            x = self.cls_fc(x)

        reps = self.reps.weight[:-1, :]

        split_size = 5
        # When the number of categories is very large, there may be a risk of exceeding the available memory, so a split processing is performed.
        if self.num_classes > split_size:
            x_ex = x[:, None, :].expand(-1, split_size, -1)
            cos_sim_list = []
            reps_list = reps.split(split_size, dim=0)
            for temp_reps in reps_list:
                temp_reps_ex = temp_reps[None, :, :].expand_as(x_ex)
                cos_sim_list.append(F.cosine_similarity(x_ex, temp_reps_ex, dim=2))
            cos_sim = torch.cat(cos_sim_list, dim=1)
            x_ex = x[:, None, :].expand(-1, reps.size(0), -1)
        else:
            x_ex = x[:, None, :].expand(-1, reps.size(0), -1)
            reps_ex = reps[None, :, :].expand_as(x_ex)
            cos_sim = F.cosine_similarity(x_ex, reps_ex, dim=2)

        if self.neg_emb_type in ('none',):
            x_neg_ex = x_ex
        elif self.neg_emb_type in ('att',):
            with torch.no_grad():
                reps_norm_ex = F.normalize(reps, p=2, dim=1)[None, :, :].expand(x.size(0), -1, -1)
                att = (reps_norm_ex.abs().detach()).tanh()
            x_neg_ex = x_ex * att
        else:
            x_neg_ex = x_ex

        neg_reps = self.neg_reps.weight.reshape(-1, self.neg_mode, self.neg_channels)

        if self.num_classes > split_size:
            neg_cos_sim_list = []
            neg_reps_list = neg_reps.split(split_size, dim=0)
            x_neg_ex_list = x_neg_ex.split(split_size, dim=1)
            for nr, xne in zip(neg_reps_list, x_neg_ex_list):
                nr_ex = nr[None, :, :, :].expand(x.size(0), -1, -1, -1)
                neg_cos_sim_list.append(
                    F.cosine_similarity(xne[:, :, None, :].expand_as(nr_ex), nr_ex, dim=3).max(2)[0])
            neg_cos_sim = torch.cat(neg_cos_sim_list, dim=1)
        else:
            neg_reps_ex = neg_reps[None, :, :, :].expand(x.size(0), -1, -1, -1)
            neg_cos_sim = \
            F.cosine_similarity(x_neg_ex[:, :, None, :].expand_as(neg_reps_ex), neg_reps_ex, dim=3).max(2)[0]

        # fused_cos_sim = cos_sim - self.neg_weight * F.leaky_relu(neg_cos_sim)

        neg_scores = torch.exp(-(torch.sub(1, neg_cos_sim)) ** self.cos_power * self.cos_scale)
        scores_ori = torch.exp(-(torch.sub(1, cos_sim)) ** self.cos_power * self.cos_scale)
        # fused_scores = torch.exp(-(torch.sub(1, fused_cos_sim))**self.cos_power * self.cos_scale)
        fused_scores = None
        scores = scores_ori

        bg_pad = torch.ones(scores.size(0), 1).to(scores.device) * -1e5
        scores = torch.cat([scores, bg_pad], dim=1)

        extra = dict(
            cos_sim=cos_sim,
            fused_scores=fused_scores,
            neg_cos_sim=neg_cos_sim,
            neg_scores=neg_scores,
            scores_ori=scores_ori,
        )

        return scores, proposal_deltas, extra

# baseline
@ROI_HEADS_OUTPUT_REGISTRY.register()
class FastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    def __init__(
            self, cfg, input_size, num_classes, cls_agnostic_bbox_reg, box_dim=4
    ):
        """
        Args:
            cfg: config
            input_size (int): channels, or (channels, height, width)
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            box_dim (int): the dimension of bounding boxes.
                Example box dimensions: 4 for regular XYXY boxes and 5 for rotated XYWHA boxes
        """
        super(FastRCNNOutputLayers, self).__init__()

        if not isinstance(input_size, int):
            input_size = np.prod(input_size)

        # The prediction layer for num_classes foreground classes and one
        # background class
        # (hence + 1)
        self.reps = nn.Linear(input_size, num_classes + 1)
        # self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.reps.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.reps, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self._do_cls_dropout = cfg.MODEL.ROI_HEADS.CLS_DROPOUT
        self._dropout_ratio = cfg.MODEL.ROI_HEADS.DROPOUT_RATIO

        self.reg_fc_channels = cfg.MODEL.ROI_HEADS.REG_FC_CHANNELS
        self.reg_fc = nn.Linear(input_size, self.reg_fc_channels)
        nn.init.normal_(self.reg_fc.weight, std=0.01)
        nn.init.constant_(self.reg_fc.bias, 0)

    def forward(self, x):
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        proposal_deltas = self.bbox_pred(self.reg_fc(x))
        # proposal_deltas = self.bbox_pred(x)

        if self._do_cls_dropout:
            x = F.dropout(x, self._dropout_ratio, training=self.training)
        scores = self.reps(x)
        return scores, proposal_deltas, dict(cos_sim=None)
