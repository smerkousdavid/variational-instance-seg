# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.transforms import transforms

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from ..layers import DetrTransformerEncoder, SinePositionalEncoding
# from ..utils import (filter_scores_and_topk, select_single_mlvl,
#                      unpack_gt_instances)
from .base import BaseDetector

cur_img = 0


@MODELS.register_module()
class DVIS(BaseDetector, metaclass=ABCMeta):
    r"""Base class for @TODO add desc


    Args:
        backbone (:obj:`ConfigDict` or dict): Config of the backbone.
        neck (:obj:`ConfigDict` or dict, optional): Config of the neck.
            Defaults to None.
        encoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer encoder. Defaults to None.
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict, optional): Config
            of the positional encoding module. Defaults to None.
        num_queries (int, optional): Number of decoder query in Transformer.
            Defaults to 100.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            the bounding box head module. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            the bounding box head module. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
            ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 encoder: OptConfigType = None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 dvis_head: OptConfigType = None,
                 positional_encoding: OptConfigType = None,
                 num_queries: int = 100,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        # process args
        # bbox_head.update(train_cfg=train_cfg)
        # bbox_head.update(test_cfg=test_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        # self.encoder = encoder
        # self.decoder = decoder
        # self.positional_encoding = positional_encoding
        self.num_queries = num_queries

        # init model layers
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        # self.bbox_head = MODELS.build(bbox_head)
        self.dvis_head = MODELS.build(dvis_head)
        # self._init_layers()

    # @abstractmethod
    # def _init_layers(self) -> None:
    #     """Initialize layers except for backbone, neck and bbox_head."""
    #     self.positional_encoding = SinePositionalEncoding(
    #         **self.positional_encoding)
    #     self.encoder = DetrTransformerEncoder(**self.encoder)
    #     self.embed_dims = self.encoder.embed_dims

    #     num_feats = self.positional_encoding.num_feats
    #     assert num_feats * 2 == self.embed_dims, \
    #         'embed_dims should be exactly 2 times of num_feats. ' \
    #         f'Found {self.embed_dims} and {num_feats}.'

    def pre_transformer(
            self,
            img_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict, Dict]:
        """Prepare the inputs of the Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            img_feats (Tuple[Tensor]): Tuple of features output from the neck,
                has shape (bs, c, h, w).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such as
                `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict, dict]: The first dict contains the inputs of encoder
            and the second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask',
              and 'memory_pos'.
        """

        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        batch_size, feat_dim, _, _ = feat.shape
        # construct binary masks which for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            masks = None
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks, input=feat)
        else:
            masks = feat.new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero values represent
            # ignored positions, while zero values mean valid positions.

            raise NotImplementedError('fix F not found')
            # masks = F.interpolate(
            #     masks.unsqueeze(1),
            #     size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            # [batch_size, embed_dim, h, w]
            pos_embed = self.positional_encoding(masks)

        # use `view` instead of `flatten` for dynamically exporting to ONNX
        # [bs, c, h, w] -> [bs, h*w, c]
        feat = feat.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        pos_embed = pos_embed.view(batch_size, feat_dim, -1).permute(0, 2, 1)
        # [bs, h, w] -> [bs, h*w]
        if masks is not None:
            masks = masks.view(batch_size, -1)

        # prepare transformer_inputs_dict
        encoder_inputs_dict = dict(
            feat=feat, feat_mask=masks, feat_pos=pos_embed)
        decoder_inputs_dict = dict(memory_mask=masks, memory_pos=pos_embed)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor) -> Dict:
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat, query_pos=feat_pos,
            key_padding_mask=feat_mask)  # for self_attn
        encoder_outputs_dict = dict(memory=memory)
        return encoder_outputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        global cur_img

        img_feats = self.extract_feat(batch_inputs)
        # head_inputs_dict = self.forward_embedding
        # (img_feats, batch_data_samples)

        losses, mask_f, norms, sal = self.dvis_head.loss(
            img_feats, batch_data_samples)

        # visualize results
        invTrans = transforms.Compose([
            transforms.Normalize(
                mean=[0., 0., 0.], std=[1 / 58.395, 1 / 57.12, 1 / 57.375]),
            transforms.Normalize(
                mean=[-123.675, -116.28, -103.53], std=[1., 1., 1.]),
        ])
        input_img = batch_inputs[0].detach().clone()
        input_img = invTrans(input_img).detach().permute(
            1, 2, 0).cpu().numpy()[:, :, :]

        forg = torch.sigmoid(sal[0])

        # forg_ind = (forg > 0.5).nonzero()

        # # create blank
        # inst = torch.zeros_like(forg)

        # create random projection to 3d
        norms = torch.where(
            forg > 0.5,
            norms[0],
            torch.ones_like(norms[0]) *
            10000  # some high number bring vector close to 0
        )

        with torch.no_grad():
            # random projection
            for i in range(10):
                weight = torch.empty((3, mask_f.shape[1], 1, 1),
                                     device=norms.device,
                                     dtype=torch.float)
                nn.init.uniform_(weight)
                weight = weight / torch.linalg.norm(
                    weight, ord=2, dim=1, keepdim=True)
                projected = nn.functional.conv2d(mask_f[0], weight)

                # scale between 0-255
                projected = projected + projected.min()
                projected = (projected / projected.max()) * 255.0

                # continue if not white
                if torch.sum(projected.mean(0) > 200).item() < (
                        0.5 * (projected.shape[1] * projected.shape[2])):
                    break

            projected1 = projected
            for i in range(10):
                weight = torch.empty((3, mask_f.shape[1], 1, 1),
                                     device=norms.device,
                                     dtype=torch.float)
                nn.init.uniform_(weight)
                weight = weight / torch.linalg.norm(
                    weight, ord=2, dim=1, keepdim=True)

                projected = nn.functional.conv2d(mask_f[0], weight)

                # scale between 0-255
                projected = projected + projected.min()
                projected = (projected / projected.max()) * 255.0

                # continue if not white
                if torch.sum(projected.mean(0) > 200).item() < (
                        0.5 * (projected.shape[1] * projected.shape[2])):
                    break

            # saliency
            sal = torch.clamp(forg * 255.0, 0,
                              255).to(torch.uint8).detach().clone().permute(
                                  1, 2, 0).cpu().numpy()[:, :, 0]
            sal = cv2.resize(sal, (input_img.shape[1], input_img.shape[0]))
            sal = cv2.cvtColor(sal, cv2.COLOR_GRAY2RGB)

            # rgb projection
            proj1 = torch.clamp(projected1, 0,
                                255).to(torch.uint8).detach().clone().permute(
                                    1, 2, 0).cpu().numpy()[:, :, :]
            proj1 = cv2.resize(proj1, (input_img.shape[1], input_img.shape[0]))

            proj = torch.clamp(projected, 0,
                               255).to(torch.uint8).detach().clone().permute(
                                   1, 2, 0).cpu().numpy()[:, :, :]
            proj = cv2.resize(proj, (input_img.shape[1], input_img.shape[0]))

            # horz stack
            input_img = np.hstack((input_img, sal, proj1, proj))
            cv2.imwrite(f'test_{cur_img}.png', input_img)
            cur_img += 1
            cur_img = cur_img % 100
        # val = torch.tensor(0.0, device='cuda', requires_grad=True)
        # losses = {
        #     'loss_all': val.sum()
        # }

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        return []
        # raise NotImplementedError('not yet')
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_embedding(img_feats,
                                                  batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        return []
        # raise NotImplementedError('not yet')
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_embedding(img_feats,
                                                  batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_embedding(self,
                          img_feats: Tuple[Tensor],
                          batch_data_samples: OptSampleList = None) -> Dict:
        feat = img_feats[-1]  # NOTE img_feats contains only one feature.
        mag = torch.linalg.vector_norm(feat, ord=2, dim=1, keepdim=True)

        # print(feat.shape, mag.shape)

        head_inputs_dict = {'feat': feat, 'mag': mag}

        return head_inputs_dict

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x
