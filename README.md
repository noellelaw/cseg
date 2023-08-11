# cseg
Context-aware open-vocabulary semantic segmentation (adapted from [ov-seg](https://github.com/facebookresearch/ov-seg/tree/main))

## Data Preparation
Please see [datasets preparation](https://github.com/facebookresearch/ov-seg/blob/main/datasets/DATASETS.md).

## Getting Started

Evaluation on the KITTI Dataset with OVSeg Pretrained Weights: (Swin-Base + CLIP-ViT-L/14) 
- Download cseg_verification.ipynb, or view [colab file](https://colab.research.google.com/drive/1NVYVUN0K6BFzwwiOWNaggb7Z8gcQUxJF?usp=sharing) directly
- Upload ovseg_swinbase_vitL14_ft_mpt.pth to accessable folder, preferably a google drive folder so that you don't have to re-upload every time. Edit the **model_weights** field under **Build Model** in the ipynb file to the path where ovseg_swinbase_vitL14_ft_mpt.pth is stored.
- Place the [KITTI Test images](https://drive.google.com/drive/folders/1LLKGeYnLXBY1lJXRKOUpTk4GKZaEoNYR?usp=drive_link) to an easily accessible place in drive. Edit the **data_fldr** field under **Read in KITTI test data from drive** in the ipynb file to this path.

Training on COCO or ADE20k dataset:
- The cseg_training.ipynb has the basic training commands and setup, it assumes you have gone through data preparation required for the detectron2 datasets.
- Due to the size of the datasets, the connection between google colab and google drive can time out.
- The /sbatch/ folder features the files required to run this in a slurm environment. It assumes that the virtual environment has all required packages installed.
- The code for full training is in **open_vocab_seg** subfolder.
- More information on training can be found [here](https://github.com/facebookresearch/ov-seg/blob/main/GETTING_STARTED.md).

Fine-tuning the classification stage (CLIP) is adapted from OVSeg: 
- [OpenCLIP Training](https://github.com/facebookresearch/ov-seg/blob/main/open_clip_training/README.md)
- The code for this is in the **open_clip_training** subfolder.
  
## Required weights
- [R103.pkl](https://drive.google.com/file/d/1L36u2_rkEOPHlXLOvy0J_3ztTSoyV6jV/view?usp=share_link)
- [ovseg_swinbase_vitL14_ft_mpt.pth](https://drive.google.com/file/d/1E_ljD_Q7h-LFVcP27UZDuwWCuSh3WAbB/view?usp=share_link)
