#!/bin/bash
#SBATCH --account=co_nilah
#SBATCH --partition=savio3_gpu
#SBATCH --qos=savio_lowprio
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --time=00:57:00
#SBATCH --job-name=RFD_HP_8E8I_CC8_I50_NS32_10_2212
#SBATCH --output=/clusterfs/nilah/sergio/RFdifussion/logs/EPITOPE_SERIES2/RFD_HP_8E8I_CC8_I50_NS32_10_2212_%j.out
#SBATCH --error=/clusterfs/nilah/sergio/RFdifussion/logs/EPITOPE_SERIES2/RFD_HP_8E8I_CC8_I50_NS32_10_2212_%j.err

source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh
conda activate SE3nv

python /global/scratch/users/sergiomar10/ESMCBA/ESMCBA/ESMCBA/run_RFDiffusionMHC_epitope.py --name RFD_HP_8E8I_CC8_I50_NS32_10_2212 --pdb 8E8I --contigs "A1-275/0 B1-100/0 0/10" --hotspots "A114,A116,A123,A124,A143,A146,A147,A152,A155,A156,A159,A163,A167,A171,A22,A302,A309,A319,A33,A334,A338,A370,A386,A391,A425,A429,A453,A454,A477,A479,A497,A5,A506,A521,A550,A567,A59,A62,A63,A646,A657,A66,A662,A664,A67,A680,A69,A699,A7,A70,A73,A74,A76,A77,A80,A81,A84,A9,A95,A97,A99" --chains A,B,C --epitope_chain C --iterations 50 --num_seqs 32
