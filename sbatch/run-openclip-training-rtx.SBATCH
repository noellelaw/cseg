#!/bin/bash -x

#SBATCH --output=train-openclip-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:1

module purge;

singularity exec --nv \
            --overlay /scratch/ntl2689/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; chmod u+x download_openclip.sh; ./download_openclip.sh; torchrun /scratch/ntl2689/pytorch-example/cseg-tfrm/open_clip_training/src/training/main.py  --train-data $SLURM_TMPDIR/openclip_data/coco_proposal_1cap.csv  --train-num-samples 442117 --lr 0.000005  --warmup 100  --force-quick-gelu --dataset-type csv  --batch-size 1 --precision amp --workers 4  --model  ViT-L-14 --lock-text --zeroshot-frequency 1  --save-frequency 1  --epoch 5  --pretrained  openai  --ade-val $SLURM_TMPDIR/openclip_data/ade_gt_150cls_val"