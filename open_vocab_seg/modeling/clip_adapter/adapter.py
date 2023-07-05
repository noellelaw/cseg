# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved
# Modified by Feng Liang from
# https://github.com/MendelXu/zsseg.baseline/blob/master/mask_former/modeling/clip_adapter/adapter.py

from typing import List
from detectron2.structures.image_list import ImageList
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import BitMasks
from .utils import build_clip_model, crop_with_mask
from .text_template import PromptExtractor

PIXEL_MEAN = (0.48145466, 0.4578275, 0.40821073)
PIXEL_STD = (0.26862954, 0.26130258, 0.27577711)

class OneDAff(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones([1, dim]))
        self.beta = nn.Parameter(torch.zeros([1, dim]))

    def forward(self, x):
        x = x * self.alpha + self.beta
        return x

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MLPblock(nn.Module):

    def __init__(self, dim, mlp_dim, dropout = 0.1, init_values=1e-4):
        super().__init__()

        self.pre_affine = OneDAff(dim)
        self.ff = nn.Sequential(
            FeedForward(dim, mlp_dim, dropout),
        )
        self.gamma = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.gamma * self.ff(x)
        return x


class ResMLP(nn.Module):

    def __init__(self, hidden_dim, output_dim, depth, contraction_factor = 4):
        super().__init__()

        # pass in a 1 x 768*2 vector 
        self.mlp_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mlp_blocks.append(MLPblock(hidden_dim, hidden_dim // contraction_factor))

        self.affine = OneDAff(hidden_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):

        for mlp_block in self.mlp_blocks:
            x = mlp_block(x)

        x = self.affine(x)

        x = x.mean(dim=1)
        #print('MLP head shape: ', self.mlp_head(x).shape)

        return self.mlp_head(x)


class ClipAdapter(nn.Module):
  def __init__(
      self, 
      clip_model_name: str, 
      clip_mask_model_name: str, 
      mask_prompt_depth: int, 
      text_templates: PromptExtractor
      ):
    super().__init__()
    # -------------------------------------------------------------------------------
    self.clip_model_reg, self.clip_preprocess = build_clip_model(clip_model_name, 
                                                             mask_prompt_depth = 0)
    self.clip_model, self.clip_mask_preprocess = build_clip_model(clip_mask_model_name, 
                                                                      mask_prompt_depth)
    # -------------------------------------------------------------------------------
    self.text_templates = text_templates
    self.text_templates.init_buffer(self.clip_model)
    self.text_feature_buffer = {}
    # -------------------------------------------------------------------------------
    #self.image_ensembler = VisionEnsembleMLP(768*2, 768, 0.1) # in dim, out dim, dropout 
    '''
    self.image_ensembler = ResMLP(mlp_num_hiddens=(768*2), 
                                  mlp_num_outputs=768, 
                                  depth = 3)
    '''
    # need to pass in (1 x 2 x 1 x 786)
    self.image_ensembler = ResMLP(hidden_dim = (768*2),
                                  output_dim = 768, 
                                  depth = 6)

  def forward(
        self, 
        image: torch.Tensor, 
        masked_image: torch.Tensor, 
        text: List[str], 
        **kwargs
        ):
    # Get image features
    image = self._preprocess_image(image)
    image_features = self.get_image_features(image)
    # Get text features
    text_feature = self.get_text_features(text)  # k,feat_dim
    # Return CLIP get sim logits 
    return self.get_sim_logits(text_feature, image_features)

  # -----------------------------------------------------------------------------
  def _preprocess_image(self, image: torch.Tensor, normalize = True):
    return image

  def _get_text_features(self, noun_list: List[str]):
    left_noun_list = [
        noun for noun in noun_list if noun not in self.text_feature_buffer
    ]
    if len(left_noun_list) > 0:
        left_text_features = self.text_templates(
            left_noun_list, self.clip_model
        )
        self.text_feature_buffer.update(
            {
                noun: text_feature
                for noun, text_feature in zip(
                    left_noun_list, left_text_features
                )
            }
        )
    return torch.stack([self.text_feature_buffer[noun] for noun in noun_list])
  
  # -----------------------------------------------------------------------------
  def get_text_features(self, noun_list: List[str]):
    return self._get_text_features(noun_list)

  # -----------------------------------------------------------------------------
  def get_mask_features(self, regions: torch.Tensor):
    image_features = self.clip_model.visual(regions)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features

  # -----------------------------------------------------------------------------
  def get_image_features(self, image: torch.Tensor):
    image_features = self.clip_model_reg.visual(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


  def get_sim_logits(
      self,
      text_features: torch.Tensor,
      image_features: torch.Tensor,
      temperature: float = 100,
  ):
      return temperature * image_features @ text_features.T

  def normalize_feature(self, feat: torch.Tensor):
      return feat / feat.norm(dim=-1, keepdim=True)
                                


class ClipEnsembler(ClipAdapter):
    def __init__(
        self,
        clip_model_name: str,
        clip_mask_model_name: str, 
        text_templates: PromptExtractor,
        mask_fill: str = "mean",
        mask_expand_ratio: float = 1.0,
        mask_thr: float = 0.5,
        mask_matting: bool = False,
        region_resized: bool = True,
        mask_prompt_depth: int = 0,
        mask_prompt_fwd: bool = False,
    ):
        super().__init__(clip_model_name, 
                         clip_mask_model_name,
                         mask_prompt_depth, 
                         text_templates)
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width ** -0.5,
        )
        # for test
        self.mask_fill = mask_fill
        if self.mask_fill == "zero":
            self.mask_fill = (0.0, 0.0, 0.0)
        elif self.mask_fill == "mean":
            self.mask_fill = [255.0 * c for c in PIXEL_MEAN]
        else:
            raise NotImplementedError(
                "Unknown mask_fill method: {}".format(self.mask_fill)
            )
        self.mask_expand_ratio = mask_expand_ratio
        self.mask_thr = mask_thr
        self.mask_matting = mask_matting
        self.region_resized = region_resized
        self.mask_prompt_fwd = mask_prompt_fwd
        self.register_buffer(
            "pixel_mean", torch.Tensor(PIXEL_MEAN).reshape(1, 3, 1, 1) * 255.0
        )
        self.register_buffer(
            "pixel_std", torch.Tensor(PIXEL_STD).reshape(1, 3, 1, 1) * 255.0
        )

    def forward(
        self,
        image: torch.Tensor,
        text: List[str],
        mask: torch.Tensor,
        normalize: bool = True,
        fwd_w_region_mask: bool = False,
    ):
      
        (regions, unnorm_regions), region_masks, valid_flag = self._preprocess_image(image, mask, normalize=normalize)
        
        if normalize:
            image = (image - self.pixel_mean) / self.pixel_std
        # resize
        images = [
            F.interpolate(image, size=(224, 224), mode="bicubic")
        ]
        image = torch.cat(images)
        #print('IMAGE: ', image.shape)
        if regions is None:
            return None, valid_flag
        if isinstance(regions, list):
            assert NotImplementedError
            mask_features = torch.cat(
                [self.get_image_features(image_i) for image_i in regions], dim=0
            )
        else:
            if self.mask_prompt_fwd:
                masked_features = self.get_mask_features(regions, region_masks)
            else:
                masked_features = self.get_mask_features(regions)

        image_features = self.get_image_features(image)

        #print(f'Ensembling features ...')
        #print('MASKED FEATURES: ', masked_features.shape)
        k, d = masked_features.shape
        ensembled_image_features = torch.cuda.FloatTensor(k, 1, d*2)
        for i in range(k):
          mf = masked_features[i,:,None].permute(1,0) 
          concat_image_features = torch.cat((image_features, mf), 1)
          #print('CONCAT IMAGE FEATURES: ', concat_image_features.shape)
          # print(concat_image_features.shape)
          # Get ensembled features
          ensembled_image_features[i,:] = concat_image_features
        #print('ENSEMBLED_FEATURES: ',ensembled_image_features.shape)
        ensembled_features = self.image_ensembler.forward(ensembled_image_features)
        #print('ENSEMBLED_FEATURES: ', ensembled_features.shape) #k x 768 

        text_feature = self.get_text_features(text)  # k,feat_dim
        return self.get_sim_logits(text_feature, ensembled_features), unnorm_regions, valid_flag
        #return self.get_sim_logits(text_feature, masked_features), unnorm_regions, valid_flag


    def get_mask_features(self, image, region_masks=None):
        image_features = self.clip_model.visual(image, region_masks)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def _preprocess_image(
        self, image: torch.Tensor, mask: torch.Tensor, normalize: bool = True
    ):
        """crop, mask and normalize the image

        Args:
            image ([type]): [C,H,W]
            mask ([type]): [K,H,W
            normalize (bool, optional): [description]. Defaults to True.
        """
        dtype = mask.dtype
        bin_mask = mask > self.mask_thr
        valid = bin_mask.sum(dim=(-1, -2)) > 0 #( 4, 1, h, w) --> h = 0, w = 2

        bin_mask = bin_mask[valid]
        mask = mask[valid]

        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = []
        region_masks = []
        valid_tracker = 0
        for bbox, single_mask in zip(bboxes, mask):
            region, region_mask, not_a_bitch = crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )
            if not_a_bitch:
              regions.append(region.unsqueeze(0))
              region_masks.append(region_mask.unsqueeze(0))
            else: 
              valid[valid_tracker] = False
            valid_tracker += 1
        if len(regions) == 0:
            return None, valid
        unnorm_regions = regions
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            for i, r in enumerate(regions):
              try:
                regions[i] = F.interpolate(r, size=(224, 224), mode="bicubic") 
              except:
                print(f'fail with {r.shape}.')
                regions[i] = torch.zeros((1, 3, 224,224)).to('cuda')
            regions = torch.cat(regions)
            for i, r in enumerate(region_masks):
              try: 
                region_masks[i] = F.interpolate(r, size=(224, 224), mode="nearest") 
              except:
                print(r.shape)
                print(f'fail with {r.shape}.')
                region_masks[i] = torch.zeros((1, 1, 224,224)).to('cuda')
            region_masks = torch.cat(region_masks)

            for i, r in enumerate(unnorm_regions):
              try:
                unnorm_regions[i] = F.interpolate(r, size=(224, 224), mode="bicubic")
              except:
                print(r.shape)
                print(f'fail with {r.shape}.')
                unnorm_regions[i] = torch.zeros((1, 3, 224,224)).to('cuda')

            unnorm_regions = torch.cat(unnorm_regions)
        return (regions, unnorm_regions), region_masks, valid

    def get_text_features(self, noun_list: List[str]):
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=0)
