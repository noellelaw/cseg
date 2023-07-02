#!/bin/bash -x

#SBATCH --output=train-cseg-multi-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:a100:4

module purge;

singularity exec --nv \
            --overlay /scratch/ntl2689/pytorch-example/my_multi_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; chmod u+x download_coco.sh; ./download_coco.sh; python /scratch/ntl2689/pytorch-example/cseg/train_net.py --num-gpu 4 --config-file /scratch/ntl2689/pytorch-example/cseg/ovseg_resnet101c_ade20k.yaml"

