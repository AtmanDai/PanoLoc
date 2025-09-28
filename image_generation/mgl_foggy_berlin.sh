#!/bin/bash
#SBATCH --partition=accelerated
#SBATCH --time=48:00:00            
#SBATCH --nodes=6                   
#SBATCH --ntasks=6                   
#SBATCH --ntasks-per-node=1          
#SBATCH --cpus-per-task=16           
#SBATCH --gres=gpu:4                
#SBATCH --mem=100GB
#SBATCH --constraint=LSDF
#SBATCH --account=hk-project-p0022785
#SBATCH --job-name=mgl_foggy_berlin
#SBATCH --output=mgl_foggy_berlin_%j.out
#SBATCH --error=mgl_foggy_berlin_%j.err

module load devel/cuda/12.4
source /home/hk-project-cvhciass/tj3409/miniconda3/etc/profile.d/conda.sh
conda activate rdai

echo "=== Job Level Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Total nodes requested: $SLURM_NNODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "===================="

export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHON_SCRIPT_PATH="/hkfs/work/workspace/scratch/tj3409-rdaionet/OrienterNet/image_generation/mgl_foggy.py"
SCENE="berlin"

srun bash -c "
echo '=== Node $SLURM_PROCID Info ==='
echo 'SLURM_PROCID: $SLURM_PROCID'
python3 $PYTHON_SCRIPT_PATH --scene '$SCENE' --node-rank \$SLURM_PROCID --total-nodes $SLURM_NTASKS
"

echo "Task $SLURM_PROCID finished."

