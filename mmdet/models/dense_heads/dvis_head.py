# Copyright (c) OpenMMLab. All rights reserved.
import copy
# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule, caffe2_xavier_init
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmdet.utils import (ConfigType, InstanceList, OptInstanceList,
                         OptMultiConfig)
# from ..losses import QualityFocalLoss
from ..utils import multi_apply
from ..utils.mean_shift import clustering_features


def preprocess_panoptic_gt(gt_labels: Tensor, gt_masks: Tensor,
                           gt_semantic_seg: Tensor, num_things: int,
                           num_stuff: int) -> Tuple[Tensor, Tensor]:
    """Preprocess the ground truth for a image specifically for DVIS. @TODO
    description.

    Args:
        gt_labels (Tensor): Ground truth labels of each bbox,
            with shape (num_gts, ).
        gt_masks (BitmapMasks): Ground truth masks of each instances
            of a image, shape (num_gts, h, w).
        gt_semantic_seg (Tensor | None): Ground truth of semantic
            segmentation with the shape (1, h, w).
            [0, num_thing_class - 1] means things,
            [num_thing_class, num_class-1] means stuff,
            255 means VOID. It's None when training instance segmentation.

    Returns:
        tuple[Tensor, Tensor]: a tuple containing the following targets.

            - labels (Tensor): Ground truth class indices for a
                image, with shape (n, ), n is the sum of number
                of stuff type and number of instance in a image.
            - masks (Tensor): Ground truth mask for a image, with
                shape (n, h, w). Contains stuff and things when training
                panoptic segmentation, and things only when training
                instance segmentation.
    """

    things_masks: torch.Tensor = gt_masks.to_tensor(
        dtype=torch.bool, device=gt_labels.device).long()

    # create a thing index variant
    # things_ind_mask = things_masks.argmax(dim=0, keepdim=True)
    # print(gt_labels.shape, things_masks.shape, things_ind_mask.shape)

    # see util/panoptic_gt_processing.py for original
    return gt_labels, things_masks  # , things_ind_mask


@MODELS.register_module()
class DVISHead(BaseModule):
    r"""Head of DETR. DETR:End-to-End Object Detection with Transformers.

    More details can be found in the `paper
    <https://arxiv.org/pdf/2005.12872>`_ .

    Args:
        num_classes (int): Number of categories excluding the background.
        embed_dims (int): The dims of Transformer embedding.
        num_reg_fcs (int): Number of fully-connected layers used in `FFN`,
            which is then used for the regression head. Defaults to 2.
        sync_cls_avg_factor (bool): Whether to sync the `avg_factor` of
            all ranks. Default to `False`.
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to `CrossEntropyLoss`.
        loss_bbox (:obj:`ConfigDict` or dict): Config of the regression bbox
            loss. Defaults to `L1Loss`.
        loss_iou (:obj:`ConfigDict` or dict): Config of the regression iou
            loss. Defaults to `GIoULoss`.
        train_cfg (:obj:`ConfigDict` or dict): Training config of transformer
            head.
        test_cfg (:obj:`ConfigDict` or dict): Testing config of transformer
            head.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    _version = 2

    def __init__(
            self,
            num_classes: int,
            in_channels: List[int],
            feat_channels: int,
            out_channels: int,
            embed_dims: int = 256,
            num_reg_fcs: int = 2,
            sync_cls_avg_factor: bool = False,
            loss_cls: ConfigType = dict(
                type='CrossEntropyLoss',
                bg_cls_weight=0.1,
                use_sigmoid=False,
                loss_weight=1.0,
                class_weight=1.0),
            # num_transformer_feat_level: int = 3,
            pixel_decoder: ConfigType = ...,
            enforce_decoder_input_project: bool = False,
            num_things_classes: int = 80,
            num_stuff_classes: int = 53,
            loss_bbox: ConfigType = dict(type='L1Loss', loss_weight=5.0),
            loss_iou: ConfigType = dict(type='GIoULoss', loss_weight=2.0),
            train_cfg: ConfigType = dict(
                assigner=dict(
                    type='HungarianAssigner',
                    match_costs=[
                        dict(type='ClassificationCost', weight=1.),
                        dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                        dict(type='IoUCost', iou_mode='giou', weight=2.0)
                    ])),
            test_cfg: ConfigType = dict(max_per_img=100),
            init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        class_weight = None  # loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is DVISHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR repo, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided ' \
                                            'when train_cfg is set.'
            assigner = train_cfg['assigner']
            self.assigner = TASK_UTILS.build(assigner)
            if train_cfg.get('sampler', None) is not None:
                raise RuntimeError('DETR do not build sampler.')
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.loss_cls = MODELS.build(loss_cls)
        # self.loss_bbox = MODELS.build(loss_bbox)
        # self.loss_iou = MODELS.build(loss_iou)

        # assert pixel_decoder.encoder.layer_cfg. \
        #     self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=2 * out_channels)
        self.hyper_dim = out_channels
        self.pixel_decoder = MODELS.build(pixel_decoder_)

        # if isinstance(self.pixel_decoder, PixelDecoder) and (
        #         self.decoder_embed_dims != in_channels[-1]
        #         or enforce_decoder_input_project):
        #     self.decoder_input_proj = Conv2d(
        #         in_channels[-1], self.decoder_embed_dims, kernel_size=1)
        # else:
        self.decoder_input_proj = nn.Identity()

        # load anchor points and construct weights
        anchor_data = torch.load('discrete_points_9.pth')
        self.margin = anchor_data['margin'].cuda()
        self.anchors = anchor_data['anchors'].cuda(
        )  # shape of [anchors, feat_dim]
        self.anchor_weights = self.anchors.view(self.anchors.shape[0],
                                                self.anchors.shape[1], 1, 1)
        self.anchor_conv = Conv2d(
            self.anchor_weights.shape[1],
            self.anchor_weights.shape[0],
            kernel_size=1,
            bias=False,
            padding=0)
        self.anchor_conv.weight = nn.Parameter(
            self.anchor_weights, requires_grad=False)

        # create the learnable classification anchors
        with torch.no_grad():
            init_cls = torch.empty(self.num_classes,
                                   self.anchors.shape[1]).normal_(
                                       mean=0.0, std=1.0).cuda()
            init_cls /= torch.linalg.norm(init_cls, ord=2, dim=1, keepdim=True)
            init_cls.requires_grad = True
        self.class_anchors = nn.Parameter(init_cls)
        # self.class_weights = self.class_anchors.view(self.class_anchors.shape[0], self.class_anchors.shape[1], 1, 1)
        # self.class_conv = Conv2d(self.class_weights.shape[1], self.class_weights.shape[0], kernel_size=1, bias=False, padding=0)
        # if self.loss_cls.use_sigmoid:
        #     self.cls_out_channels = num_classes
        # else:
        #     self.cls_out_channels = num_classes + 1

        # self._init_layers()

    def init_weights(self) -> None:
        if isinstance(self.decoder_input_proj, Conv2d):
            caffe2_xavier_init(self.decoder_input_proj, bias=0)

        self.pixel_decoder.init_weights()

    # def _init_layers(self) -> None:
    #     """Initialize layers of the transformer head."""
    #     # cls branch
    #     self.fc_cls = Linear(self.embed_dims, self.cls_out_channels)
    #     # reg branch
    #     self.activate = nn.ReLU()
    #     self.reg_ffn = FFN(
    #         self.embed_dims,
    #         self.embed_dims,
    #         self.num_reg_fcs,
    #         dict(type='ReLU', inplace=True),
    #         dropout=0.0,
    #         add_residual=False)
    #     # NOTE the activations of reg_branch here is the same as
    #     # those in transformer, but they are actually different
    #     # in DAB-DETR (prelu in transformer and relu in reg_branch)
    #     self.fc_reg = Linear(self.embed_dims, 4)

    def forward(self, x: Tuple[Tensor],
                batch_data_samples: SampleList) -> Tuple[Tensor]:
        """"
        @TODO FIX
        Forward function.

        Args:
            hidden_states (Tensor): Features from transformer decoder. If
                `return_intermediate_dec` in detr.py is True output has shape
                (num_decoder_layers, bs, num_queries, dim), else has shape
                (1, bs, num_queries, dim) which only contains the last layer
                outputs.
        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - layers_cls_scores (Tensor): Outputs from the classification head,
              shape (num_decoder_layers, bs, num_queries, cls_out_channels).
              Note cls_out_channels should include background.
            - layers_bbox_preds (Tensor): Sigmoid outputs from the regression
              head with normalized coordinate format (cx, cy, w, h), has shape
              (num_decoder_layers, bs, num_queries, 4).
        """

        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        # batch_size = x[0].shape[0]
        # input_img_h, input_img_w = batch_img_metas[0]['batch_input_shape']

        # when backbone is swin, memory is output of last stage of swin.
        # when backbone is r50, memory is output of tranformer encoder.
        mask_features, _ = self.pixel_decoder(x, batch_img_metas)

        # project to expected output chan dimension
        # memory = self.decoder_input_proj(memory)

        # layers_cls_scores = self.fc_cls(hidden_states)
        # layers_bbox_preds = self.fc_reg(
        #     self.activate(self.reg_ffn(hidden_states))).sigmoid()
        # return layers_cls_scores, layers_bbox_preds

        # handle saliency via feature norms
        norms = torch.linalg.vector_norm(
            mask_features[:self.hyper_dim], ord=2, dim=1, keepdim=True)
        saliency = 1.1 * norms - 4.2

        return mask_features[:, :self.
                             hyper_dim], mask_features[:, self.
                                                       hyper_dim:], norms, saliency

    def preprocess_gt(
            self, batch_gt_instances: InstanceList,
            batch_gt_semantic_segs: List[Optional[PixelData]]) -> InstanceList:
        """Preprocess the ground truth for all images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``labels``, each is
                ground truth labels of each bbox, with shape (num_gts, )
                and ``masks``, each is ground truth masks of each instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[Optional[PixelData]]): Ground truth of
                semantic segmentation, each with the shape (1, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.

        Returns:
            list[obj:`InstanceData`]: each contains the following keys

                - labels (Tensor): Ground truth class indices\
                    for a image, with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (Tensor): Ground truth mask for a\
                    image, with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(batch_gt_instances)
        num_stuff_list = [self.num_stuff_classes] * len(batch_gt_instances)
        gt_labels_list = [
            gt_instances['labels'] for gt_instances in batch_gt_instances
        ]
        gt_masks_list = [
            gt_instances['masks'] for gt_instances in batch_gt_instances
        ]
        gt_semantic_segs = [
            None if gt_semantic_seg is None else gt_semantic_seg.sem_seg
            for gt_semantic_seg in batch_gt_semantic_segs
        ]
        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list)
        labels, masks = targets
        batch_gt_instances = [
            InstanceData(labels=label, masks=mask)
            for label, mask in zip(labels, masks)
        ]
        return batch_gt_instances

    def loss(self, x: Tensor, batch_data_samples: SampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (Tensor): Feature from the transformer decoder, has
                shape (num_decoder_layers, bs, num_queries, cls_out_channels)
                or (num_decoder_layers, num_queries, bs, cls_out_channels).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        mask_features, class_features, norms, saliency = self(
            x, batch_data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)

        losses = self.loss_by_feat(mask_features, class_features, norms,
                                   saliency, batch_gt_instances,
                                   batch_img_metas)

        return losses, mask_features, norms, saliency

    def loss_by_feat(
        self,
        mask_features: Tensor,
        class_features: Tensor,
        norms: Tensor,
        saliency: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """"Loss function.

        Only outputs from the last feature level are used for computing
        losses by default.

        Args:
            all_layers_cls_scores (Tensor): Classification outputs
                of each decoder layers. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            all_layers_bbox_preds (Tensor): Sigmoid regression
                outputs of each decoder layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            'for batch_gt_instances_ignore setting to None.'

        losses_saliency, losses_pairwise, losses_discrete, losses_class = multi_apply(
            self.loss_by_feat_single, mask_features, class_features, norms,
            saliency, batch_gt_instances, batch_img_metas)

        loss_dict = dict()
        loss_dict['loss_saliency'] = losses_saliency
        loss_dict['loss_pairwise'] = losses_pairwise
        loss_dict['loss_discrete'] = losses_discrete
        loss_dict['loss_class'] = losses_class

        # loss from the last decoder layer
        # loss_dict['loss_cls'] = losses_cls[-1]
        # loss_dict['loss_bbox'] = losses_bbox[-1]
        # loss_dict['loss_iou'] = losses_iou[-1]
        # # loss from other decoder layers
        # num_dec_layer = 0
        # for loss_cls_i, loss_bbox_i, loss_iou_i in \
        #         zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1]):
        #     loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
        #     loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
        #     loss_dict[f'd{num_dec_layer}.loss_iou'] = loss_iou_i
        #     num_dec_layer += 1
        return loss_dict

    def discretize(self, feats):
        # feats of shape [c, h, w]
        return self.anchor_conv(feats.unsqueeze(0)).squeeze(0).argmax(dim=0)

    def batch_discretize(self, feats):
        # feats of shape [c, h, w]
        return self.anchor_conv(feats.unsqueeze(0)).squeeze(0).argmax(dim=0)

    def classify(self, feats):
        # feats of shape [c, h, w]
        return torch.nn.functional.conv2d(
            feats.unsqueeze(0),
            weight=self.class_anchors.view(self.class_anchors.shape[0],
                                           self.class_anchors.shape[1], 1,
                                           1)).squeeze(0)

    def loss_by_feat_single(self, mask_feat: Tensor, class_feat: Tensor,
                            norms: Tensor, saliency: Tensor,
                            bathc_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """

        # @TODO DAVID IDEA
        # test difference between using a global saliency mask using CE
        # and then a version where we only apply CE within each ground
        # truth object bbox where each bbox is equally weighted and
        # ignore any other region this should promote smaller
        # objects/more proposals
        # GLOBAL = False
        BG_WEIGHT = 1.0  # reweight background by this area

        # downsample ground truth mask to match predicted shape
        with torch.no_grad():
            gt_masks = bathc_gt_instances.masks
            gt_labels = bathc_gt_instances.labels

            target_shape = mask_feat.shape[-2:]
            has_masks = True
            if gt_masks.shape[0] > 0:
                gt_masks_downsampled: torch.Tensor = F.interpolate(
                    gt_masks.unsqueeze(1).float(),
                    target_shape,
                    mode='nearest').squeeze(1).long()

                # create background "mask"
                # any foreground gt should have value above 0.5
                bg_mask = torch.ones((1, *gt_masks_downsampled.shape[1:]),
                                     device=gt_masks_downsampled.device,
                                     dtype=torch.float) * 0.5

                # get index where 0 is background
                gt_ind_masks = torch.concat(
                    [bg_mask, gt_masks_downsampled.float()]).argmax(
                        dim=0, keepdim=True)
                gt_saliency = (gt_ind_masks > 0).bool()

                # create the instance weighting based on instance area
                mask_areas = gt_masks_downsampled.sum(dim=(1, 2))

                # calculate background area weight
                total_area = np.prod(target_shape)
                bg_area = (total_area - mask_areas.sum()).view((1, ))
                mask_areas = torch.concat([bg_area, mask_areas], dim=0).float()
                mask_weights = mask_areas / torch.linalg.vector_norm(
                    mask_areas, ord=2)  # [gt]
                mask_weights[0] *= BG_WEIGHT

                # mask_weights is currently [gt] convert to single full mask with each weight
                weighted_saliency = mask_weights[
                    gt_ind_masks]  # now [bs, h, w] with each element being the weight of mask from index in gt_ind_masks
            else:
                gt_masks_downsampled: torch.Tensor = gt_masks
                gt_saliency = torch.zeros_like(
                    norms, dtype=torch.bool)  # nothing predicted
                weighted_saliency = torch.ones_like(
                    gt_saliency) * BG_WEIGHT  # all background weighting
                has_masks = False

        # saliency loss
        # print('INPUT', saliency.shape, gt_saliency.shape, mask_weights.shape)
        loss_saliency = torch.binary_cross_entropy_with_logits(
            input=saliency,  # [bs, h, w]
            target=gt_saliency.float(),  # [bs, h, w]
            weight=weighted_saliency  # broadcastable to [bs, h, w]
        )

        # apply foreground terms
        if has_masks:
            sample_points = 1500
            pred_flatten = mask_feat.permute(1, 2, 0) \
                        .view(-1, mask_feat.shape[0])
            cls_flatten = class_feat.permute(1, 2, 0) \
                        .view(-1, class_feat.shape[0])
            norm_flatten = norms.permute(1, 2, 0) \
                        .view(pred_flatten.shape[0], 1)
            gt_ind_flatten = gt_ind_masks.ravel()

            # calculate class norms
            cls_norms = torch.linalg.vector_norm(
                class_feat, ord=2, dim=0, keepdim=True)
            cls_norm_flatten = cls_norms.permute(1, 2, 0) \
                        .view(cls_flatten.shape[0], 1)

            # get agreed foreground pixels. As in where
            # model predicts foreground (high confidence) and ground truth
            # agrees it is foreground we use these points to push instance
            # labels together/apart
            foreground_filter = 0.15
            gt_ind_flatten[torch.sigmoid(norm_flatten).squeeze() <
                           foreground_filter] = 0  # zero out model low scores
            gt_foreground_ind = gt_ind_flatten.nonzero()

            # if non then skip
            num_fg = gt_foreground_ind.numel()
            if num_fg < 3:
                has_masks = False  # in fact no gt masks with agreement
            else:
                # randomly select foreground pixels (up to sample points)
                perm = torch.randperm(
                    num_fg, device=gt_foreground_ind.device)[:sample_points]
                fg_pts = gt_foreground_ind[perm].squeeze(
                )  # get the indices of the sampled points
                # print(perm, perm.shape)
                # print(fg_pts, fg_pts.shape)
                num_fg = perm.shape[0]

                # select the sampled pixels
                # print(pred_flatten.shape)
                pred_fg = pred_flatten[fg_pts].contiguous()
                cls_fg = cls_flatten[fg_pts].contiguous()
                norm_fg = norm_flatten[fg_pts].contiguous()
                cls_norm_fg = cls_norm_flatten[fg_pts].contiguous()
                gt_fg = gt_ind_flatten[fg_pts].contiguous()

                # get pairwise hamming distance as an indicator of
                # within same instance or not (hamming distance simple enough)
                diff_indicator = torch.pdist(
                    gt_fg.unsqueeze(1).float(), p=0).detach()

                # get pairwise indices
                pairwise_ind = torch.triu_indices(
                    num_fg, num_fg, offset=1, device=diff_indicator.device)

                # project fg vectors to hypersphere
                pred_hyper = pred_fg / norm_fg
                cls_hyper = cls_fg / cls_norm_fg
                # norms now are above "foreground_filter" so no epsilon needed

                # compute pairwise cosine similarity
                pred_cossim = pred_hyper.matmul(
                    pred_hyper.t().detach())[pairwise_ind[0],
                                             pairwise_ind[1]].ravel()

                # compute anchor similarity (anchors already on hypersphere)
                # anchor_sim = pred_hyper.matmul(self.anchors.t().detach())

                # get average anchor and move towards an anchor
                # to get a more discrete representation
                usable_ind = torch.unique(gt_fg)
                loss_discrete = 0.0
                loss_class = 0.0
                for ind in usable_ind:
                    # move mask mean embedding to anchor
                    mean = pred_hyper[gt_fg == ind].mean(
                        dim=0, keepdim=True)  # now [1, feat_dim]
                    mean = mean / (
                        torch.linalg.vector_norm(mean, ord=2) + 1e-5)

                    # get most similar anchor
                    # print(mean.shape, self.anchors.shape)
                    anchor_sim_ind = mean.matmul(
                        self.anchors.t().detach()).argmax(dim=1).squeeze()
                    # print(mean.matmul(self.anchors.t().detach()).squeeze().shape, mean.matmul(self.anchors.t().detach()).max())

                    # move mean towards anchor
                    cosim = self.anchors[anchor_sim_ind].unsqueeze(0).matmul(
                        mean.t().detach()).squeeze()
                    # geod = torch.arccos(cosim / (1.0 + 0.05))
                    loss_discrete += 0.1 * (1.0 - cosim)

                    # now move the class points towards the anchor of all foreground points
                    clust = cls_hyper[
                        gt_fg ==
                        ind]  # [N, feat_dim] where N is number of sampled foreground mask points

                    # move towards target class by label
                    target_class = self.class_anchors[gt_labels[
                        ind - 1]]  # backshift by 1 to ignore background
                    target_class = target_class / torch.linalg.vector_norm(
                        target_class, ord=2)
                    cls_cosim = target_class.unsqueeze(0).matmul(
                        clust.t().detach())
                    loss_class += 5.0 * (1.0 - cls_cosim.mean())

                loss_discrete /= usable_ind.numel()
                loss_class /= usable_ind.numel()

                # pairwise weight is instance weighting between pairs
                fg_weights: torch.Tensor = mask_weights[gt_fg].view(num_fg, 1)

                pairwise_weight = fg_weights.matmul(
                    fg_weights.t())[pairwise_ind[0], pairwise_ind[1]].ravel()
                pairwise_weight = (pairwise_weight / torch.linalg.vector_norm(
                    pairwise_weight,
                    ord=2)).detach()  # normalize by pairwise instance weights
                # @ TODO see if this is necessary

                # compute similarity
                # pred_sim = 0.5 * (1 + pred_cossim)

                # hyperspherical energy for diff points
                geodesics = torch.arccos(pred_cossim / (1.0 + 0.05))
                hyper_same_energy = torch.square(geodesics)
                smooth = 1e-7
                hyper_diff_energy = (1.0 + smooth) / (
                    hyper_same_energy + smooth)

                # loss is to max cossim of similar instances and min cossim
                # (with some margin) of different instances
                # margin = np.arccos(0.4)
                # @TODO make the margin part of config
                # print('num sim', (1.0 - diff_indicator).count_nonzero(),
                # 'num diff' , (diff_indicator).count_nonzero())
                weight_same = 1.0
                weight_diff = 1.0
                loss_pairwise = 900 * pairwise_weight * (
                    (1.0 - diff_indicator) * weight_same * hyper_same_energy +
                    (diff_indicator * (pred_cossim < self.margin).detach() *
                     weight_diff * hyper_diff_energy))
                # (((1.0 - diff_indicator) * (1.0 - pred_sim))
                # + (diff_indicator * torch.relu(pred_sim - margin)))
        if not has_masks:
            loss_pairwise = torch.tensor(
                0.0, dtype=torch.float, device=loss_saliency.device)
            loss_discrete = torch.tensor(
                0.0, dtype=torch.float, device=loss_saliency.device)
            loss_class = torch.tensor(
                0.0, dtype=torch.float, device=loss_saliency.device)

        return loss_saliency, loss_pairwise, loss_discrete, loss_class

    def get_targets(self, cls_scores_list: List[Tensor],
                    bbox_preds_list: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict]) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_targets_single,
                                      cls_scores_list, bbox_preds_list,
                                      batch_gt_instances, batch_img_metas)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def _get_targets_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        img_h, img_w = img_meta['img_shape']
        factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)
        num_bboxes = bbox_pred.size(0)
        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        pred_instances = InstanceData(scores=cls_score, bboxes=bbox_pred)
        # assigner and sampler
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)

        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_bboxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def loss_and_predict(
            self, x: Tuple[Tensor],
            batch_data_samples: SampleList) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples. Over-write because
        img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (tuple[Tensor]): Feature from the transformer
                decoder, has shape (num_decoder_layers, bs, num_queries, dim).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: the return value is a tuple contains:

            - losses: (dict[str, Tensor]): A dictionary of loss components.
            - predictions (list[:obj:`InstanceData`]): Detection
              results of each image after the post process.
        """
        batch_gt_instances = []
        batch_img_metas = []
        batch_gt_semantic_segs = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)
            if 'gt_sem_seg' in data_sample:
                batch_gt_semantic_segs.append(data_sample.gt_sem_seg)
            else:
                batch_gt_semantic_segs.append(None)

        # forward
        mask_features, class_features, norms, saliency = self(
            x, batch_data_samples)

        # preprocess ground truth
        batch_gt_instances = self.preprocess_gt(batch_gt_instances,
                                                batch_gt_semantic_segs)

        losses = self.loss_by_feat(mask_features, class_features, norms,
                                   saliency, batch_gt_instances,
                                   batch_img_metas)
        predictions = self.predict_by_feat(mask_features, class_features,
                                           norms, saliency, batch_img_metas)

        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = True) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network. Over-write
        because img_metas are needed as inputs for bbox_head.

        Args:
            hidden_states (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)

        # last_layer_hidden_state = hidden_states[-1].unsqueeze(0)
        # outs = self(last_layer_hidden_state)

        # predictions = self.predict_by_feat(
        #     *outs, batch_img_metas=batch_img_metas, rescale=rescale)

        # forward
        mask_features, class_features, norms, saliency = self(
            x, batch_data_samples)

        predictions = self.predict_by_feat(
            mask_features,
            class_features,
            norms,
            saliency,
            batch_img_metas,
            rescale=rescale)

        return predictions

    def predict_by_feat(self,
                        mask_f: Tensor,
                        class_f: Tensor,
                        norms: Tensor,
                        saliency: Tensor,
                        batch_img_metas: List[dict],
                        rescale: bool = True) -> InstanceList:
        """Transform network outputs for a batch into bbox predictions.

        Args:
            layer_cls_scores (Tensor): Classification outputs of the last or
                all decoder layer. Each is a 4D-tensor, has shape
                (num_decoder_layers, bs, num_queries, cls_out_channels).
            layer_bbox_preds (Tensor): Sigmoid regression outputs of the last
                or all decoder layer. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and shape
                (num_decoder_layers, bs, num_queries, 4).
            batch_img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If `True`, return boxes in original
                image space. Defaults to `True`.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(mask_f[img_id],
                                                   class_f[img_id],
                                                   norms[img_id],
                                                   saliency[img_id], img_meta,
                                                   rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                mask_feat: Tensor,
                                class_feat: Tensor,
                                norms: Tensor,
                                saliency: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        # raise NotImplementedError('not done yet')
        # assert len(cls_score) == len(bbox_pred)  # num_queries
        # max_per_img = self.test_cfg.get('max_per_img', len(cls_score))

        img_shape = img_meta['img_shape']
        forg = torch.sigmoid(saliency)

        # get labels
        out_labels = torch.where(
            forg > 0.5,
            self.discretize(mask_feat / norms) + 1,  # shift one for background
            torch.zeros_like(norms))

        # construct the masks
        inds = torch.unique(out_labels)
        num_inst = inds.numel()
        masks = torch.zeros((num_inst, *out_labels.shape[-2:]),
                            device=mask_feat.device,
                            dtype=torch.bool)
        for i, ind in enumerate(inds):
            masks[i] = out_labels == ind

        # get classification labels by taking the most common label in the mask
        cls_labels = torch.zeros((num_inst, ),
                                 device=mask_feat.device,
                                 dtype=torch.long)
        cls_scores = torch.zeros((num_inst, ),
                                 device=mask_feat.device,
                                 dtype=torch.float)

        cls_ancors = self.class_anchors / torch.linalg.vector_norm(
            self.class_anchors, ord=2, dim=1, keepdim=True)
        scores = torch.nn.functional.conv2d(
            class_feat /
            torch.linalg.vector_norm(class_feat, ord=2, dim=0, keepdim=True),
            weight=cls_ancors.view(cls_ancors.shape[0], cls_ancors.shape[1], 1,
                                   1))
        det_bboxes = torch.zeros((num_inst, 4),
                                 device=mask_feat.device,
                                 dtype=torch.float)

        for i, ind in enumerate(inds):
            mask = out_labels == ind

            # get bbox by min/max of mask
            a = torch.nonzero(mask)
            bbox = torch.min(a[:, 1]), torch.min(a[:, 0]), torch.max(
                a[:, 1]), torch.max(a[:, 0])

            # normalize bbox
            bbox = torch.tensor(
                bbox, device=mask_feat.device, dtype=torch.float)
            bbox = bbox / torch.tensor([
                mask.shape[1],
                mask.shape[0],
                mask.shape[1],
                mask.shape[0],
            ],
                                       device=mask_feat.device)

            # scale to img_shape
            det_bboxes[i, 0::2] *= img_shape[1]
            det_bboxes[i, 1::2] *= img_shape[0]
            det_bboxes[i, 0::2].clamp_(min=0, max=img_shape[1])
            det_bboxes[i, 1::2].clamp_(min=0, max=img_shape[0])

            cls_scores[i], cls_labels[i] = torch.softmax(
                scores[:, mask[0]].mean(dim=1), dim=0).max(0)

        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        # rescale masks to img_shape
        masks = F.interpolate(
            masks.unsqueeze(0).float(),
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False).squeeze(0)

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = cls_scores
        results.labels = cls_labels
        results.masks = masks
        return results
