#!/bin/bash
#SBATCH --partition=dev_accelerated
#SBATCH --time=00:30:00            
#SBATCH --nodes=2                   
#SBATCH --ntasks=8                   
#SBATCH --ntasks-per-node=4        
#SBATCH --cpus-per-task=10           
#SBATCH --gres=gpu:4                
#SBATCH --mem=100GB
#SBATCH --constraint=LSDF
#SBATCH --account=hk-project-p0022786
#SBATCH --job-name=mgl_final_120
#SBATCH --output=OrienterNet_MGL_final_120_%j.out
#SBATCH --error=OrienterNet_MGL_final_120_%j.err

module load devel/cuda/11.8
source /home/hk-project-cvhciass/tj3409/miniconda3/etc/profile.d/conda.sh
conda activate rdai

export CUDA_VISIBLE_DEVICES=0,1,2,3
DATA="mapillary_final"
EXPERIMENT="OrienterNet_MGL_final_120"
CHECKPOINT_PATH="experiments/orienternet_mgl.ckpt"
FINETUNE_ARG="training.finetune_from_checkpoint='\"$CHECKPOINT_PATH\"'"

srun --jobid $SLURM_JOBID bash -c "
python3 -m maploc.train_120 data=$DATA experiment.name=$EXPERIMENT $FINETUNE_ARG
"
echo "Task $SLURM_PROCID finished."