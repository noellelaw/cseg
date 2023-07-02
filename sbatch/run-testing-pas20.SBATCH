#!/bin/bash -x

#SBATCH --output=test-pasvoc-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:rtx8000:1

module purge;

singularity exec --nv \
            --overlay /scratch/ntl2689/pytorch-example/my_multi_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
            /bin/bash -c "source /ext3/env.sh; chmod u+x download_voc.sh; ./download_voc.sh"

