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

class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim))
        self.b = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        return x * self.g + self.b

class PreAffinePostLayerScale(nn.Module): # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x

def ResMLP(*, mlp_num_hiddens, mlp_num_outputs, depth, expansion_factor = 4):
  dim = 1024 #mlp_num_hiddens
  wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)
  return nn.Sequential(
      nn.Linear(mlp_num_hiddens, dim),
      *[nn.Sequential(
          wrapper(i, nn.Sequential(
              nn.Linear(dim, dim * expansion_factor),
              nn.GELU(),
              nn.Linear(dim * expansion_factor, dim)
          ))
      ) for i in range(depth)],
      Affine(dim),
      nn.Linear(dim, mlp_num_outputs)
  )

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
    self.image_ensembler = ResMLP(mlp_num_hiddens=(768*2), 
                                  mlp_num_outputs=768, 
                                  depth = 3)

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
    '''
    import torchvision.transforms as T
    tfrm = T.ToPILImage()
    pil_image = tfrm(image)
    image = self.clip_preprocess(pil_image)
    image = image[:, :, :, None].permute(3,0,1,2).float().to('cuda')
    '''
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
        ensembled_image_features = []
        k = masked_features.shape[0]
        for i in range(k):
          mf = masked_features[i,:,None].permute(1,0)
          concat_image_features = torch.cat((image_features, mf),1)
          # print(concat_image_features.shape)
          # Get ensembled features
          ensembled_image_features.append(self.image_ensembler.forward(concat_image_features))
        ensembled_image_features = torch.cat(tuple(ensembled_image_features ))
        #print('ENSEMBLED_FEATURES: ', ensembled_image_features.shape) #k x 768 

        text_feature = self.get_text_features(text)  # k,feat_dim
        return self.get_sim_logits(text_feature, ensembled_image_features), unnorm_regions, valid_flag
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
        valid = bin_mask.sum(dim=(-1, -2)) > 0
        bin_mask = bin_mask[valid]
        mask = mask[valid]
        if not self.mask_matting:
            mask = bin_mask
        bin_mask = BitMasks(bin_mask)
        bboxes = bin_mask.get_bounding_boxes()
        # crop,mask
        regions = []
        region_masks = []
        for bbox, single_mask in zip(bboxes, mask):
            region, region_mask = crop_with_mask(
                image.type(dtype),
                single_mask.type(dtype),
                bbox,
                fill=self.mask_fill,
                expand_ratio=self.mask_expand_ratio,
            )
            regions.append(region.unsqueeze(0))
            region_masks.append(region_mask.unsqueeze(0))
        if len(regions) == 0:
            return None, valid
        unnorm_regions = regions
        if normalize:
            regions = [(r - self.pixel_mean) / self.pixel_std for r in regions]
        # resize
        if self.region_resized:
            regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in regions
            ]
            regions = torch.cat(regions)
            region_masks = [
                F.interpolate(r, size=(224, 224), mode="nearest") for r in region_masks
            ]
            region_masks = torch.cat(region_masks)
            unnorm_regions = [
                F.interpolate(r, size=(224, 224), mode="bicubic") for r in unnorm_regions
            ]
            unnorm_regions = torch.cat(unnorm_regions)
        return (regions, unnorm_regions), region_masks, valid

    def get_text_features(self, noun_list: List[str]):
        object_text_features = self._get_text_features(noun_list)
        non_object_text_features = (
            self.non_object_embedding
            / self.non_object_embedding.norm(dim=-1, keepdim=True)
        )
        return torch.cat([object_text_features, non_object_text_features], dim=0)
