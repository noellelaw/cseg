#!/bin/bash -x

#SBATCH --output=train-cseg-e0-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:1

module purge;

singularity exec --nv \
            --overlay /scratch/ntl2689/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; chmod u+x download_coco.sh; ./download_coco.sh; python /scratch/ntl2689/pytorch-example/cseg/train_net.py --num-gpu 1 --config-file /scratch/ntl2689/pytorch-example/cseg/configs/ovseg_swinB_vitL_bs32_coco.yaml --resume MODEL.WEIGHTS /scratch/ntl2689/pytorch-example/output_e0_b64/model_0001999.pth  MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT 0.0 SOLVER.IMS_PER_BATCH 64 OUTPUT_DIR ./output_e0_b64"


