#!/bin/bash -x

#SBATCH --output=train-cseg-multidist-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --mem=128GB
#SBATCH --gres=gpu:4

module purge;

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: tcp://$head_node_ip:8686
export LOGLEVEL=INFO


singularity exec --nv \
            --overlay /scratch/ntl2689/pytorch-example/my_multi_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python /scratch/ntl2689/pytorch-example/ov-seg/train_net.py --num-gpu 4 --config-file /scratch/ntl2689/pytorch-example/ov-seg/ovseg_resnet101c_ade20k.yaml --dist-url tcp://$head_node_ip:8686 --num-machines 2 --machine-rank 0; python /scratch/ntl2689/pytorch-example/ov-seg/train_net.py --num-gpu 4 --config-file /scratch/ntl2689/pytorch-example/ov-seg/ovseg_resnet101c_ade20k.yaml --dist-url tcp://$head_node_ip:8686 --num-machines 2 --machine-rank 1"
