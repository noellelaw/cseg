#!/bin/bash -x

#SBATCH --output=train-cseg-tfrm-b32-%j.out
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
            /bin/bash -c "source /ext3/env.sh; chmod u+x download_coco.sh; ./download_coco.sh; python /scratch/ntl2689/pytorch-example/cseg-tfrm/train_net.py --num-gpu 1 --config-file /scratch/ntl2689/pytorch-example/cseg-tfrm/configs/ovseg_swinB_vitL_bs32_coco.yaml SOLVER.IMS_PER_BATCH 32 OUTPUT_DIR ./output_tfrm_b32"


