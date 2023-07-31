# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 1024,
    ) -> None:
        """
        Predicts classes of masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer.to('cuda:0')

        self.num_multimask_outputs = num_multimask_outputs

        self.num_mask_class_tokens = num_multimask_outputs
        self.mask_tokens = nn.Embedding(self.num_mask_class_tokens, transformer_dim )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        mask_embeddings: torch.Tensor,
        cls_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          mask_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks --> need this to be embeddings
          torch.Tensor: batched predictions of mask quality
        """
        masks = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            mask_embeddings=mask_embeddings,
            mask_class_embeddings=cls_embed,
        )

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        mask_embeddings: torch.Tensor,
        mask_class_embeddings: torch.Tensor,
        sparse_embeddings: torch.Tensor = None
    ) -> torch.Tensor:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        #print('MASK CLASS EMBED: ', mask_class_embeddings.shape)
        output_tokens = torch.cat([self.mask_tokens.weight], dim=0).to('cuda:0')
        output_tokens = output_tokens.unsqueeze(0).expand(mask_class_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, mask_class_embeddings), dim=1)
        #print('TOKENS: ', tokens.shape)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        #print('src size, image embeddings: ', src.shape, image_embeddings.shape)
        #print('MASK EMBED: ', mask_embeddings.shape)
        src = src + mask_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, s = src.shape
        #print('src size ', src.shape, image_embeddings.shape)

        # Run the transformer
        hs = self.transformer(src, pos_src, tokens)
        mask_tokens_out = hs[:, : (self.num_mask_class_tokens), :]

        return mask_tokens_out


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x