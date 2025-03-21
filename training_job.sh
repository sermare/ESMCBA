#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=02:10:00
#SBATCH --job-name=ESMCBA_ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2_0.95_30_epitope_0.001_0.001_8_A0201
#SBATCH --output=/global/scratch/users/sergiomar10/logs/ESMCBA_02032025/ESMCBA_ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2_0.95_30_epitope_0.001_0.001_8_A0201_%j.out
#SBATCH --error=/global/scratch/users/sergiomar10/logs/ESMCBA_02032025/ESMCBA_ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2_0.95_30_epitope_0.001_0.001_8_A0201_%j.err
source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh

conda activate ESM_cambrian

python3 /global/scratch/users/sergiomar10/py_files/ESM-C_BA_09022025.py --name_of_model ESMCBA_epitope_0.95_30_ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2_0.001_0.001__8_A0201 --encoding epitope --file_path /global/scratch/users/sergiomar10/models/ESMC_Pretrain/HLAHLAA0201/ESMMASK_epitope_FT_5_0.001_1e-06_AUG_6_HLAA0201_2.pt --train_size 0.95 --blocks_unfrozen 30 --base_block_lr 0.001 --regression_block_lr 0.001 --HLA A0201
