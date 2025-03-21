import pandas as pd
pre_trained_models = pd.read_csv('ESMCBA_models.csv')

################################################
# 6. Define a simple FASTA parser
################################################
def parse_fasta(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        header = None
        seq = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq:
                    sequences.append((header, seq))
                    seq = ""
                header = line[1:]
            else:
                seq += line
        # Append the last sequence if present
        if seq:
            sequences.append((header, seq))
    return sequences


############################################
# 7. Load and filter the data
############################################
train_fasta = "/global/scratch/users/sergiomar10/jupyter_notebooks/hla_protein_sequences.fasta"
all_data = parse_fasta(train_fasta)

filtered_data = []
for header, sequence in all_data:
    if sequence[0] != 'M':
        continue
    if len(sequence) < 50:
        continue
    if 'X' in sequence:
        continue

    header = header.split('|')[1][:7].replace('*','').replace(':','')

    filtered_data.append((header))

import itertools
import os
import glob

script_path = '/global/scratch/users/sergiomar10/py_files/ESM-C_BA_09022025.py'
sh_file_dir = '/global/scratch/users/sergiomar10/slurm_jobs/ESMCBA_02032025'
os.makedirs(sh_file_dir, exist_ok=True)

trials = [5, 6, 7, 8]
encoding = ['HLA', 'epitope']
training_prop = [0.5, 0.8, 0.95]
blocks_unfrozen = [20, 30]
last_block_lr = [1e-3, 1e-4]
regression_block_lr = [1e-3, 1e-5, 1e-6]
HLAs = filtered_data

for trial_n, encode, train_prop, num_block, lr_base, lr_reg, HLA  in itertools.product(trials, encoding, training_prop, blocks_unfrozen, last_block_lr, regression_block_lr, HLAs):

    path_model = glob.glob(f'/global/scratch/users/sergiomar10/models/ESMC_Pretrain/HLAHLA{HLA}/*_{encode}_*.pt')

    if len(path_model) < 1:
        continue
    
    pre_trained_models_path = path_model[0]
    pretrain_filepath_name =  pre_trained_models_path.split('/')[-1].replace('.pt','')
    
    file_name = f'ESMCBA_{pretrain_filepath_name}_{train_prop}_{num_block}_{encode}_{lr_base}_{lr_reg}_{trial_n}_{HLA}'

    sh_filename = f'{file_name}.sh'
    sh_filepath = os.path.join(sh_file_dir, sh_filename)
    
    # Construct the command to run the script with the current parameters
    cmd = f'python3 {script_path} --name_of_model ESMCBA_{encode}_{train_prop}_{num_block}_{pretrain_filepath_name}_{lr_base}_{lr_reg}__{trial_n}_{HLA} --encoding {encode} --file_path {pre_trained_models_path} --train_size {train_prop} --blocks_unfrozen {num_block} --base_block_lr {lr_base} --regression_block_lr {lr_reg} --HLA {HLA}'

    with open(sh_filepath, 'w') as sh_file:
        sh_file.write('#!/bin/bash\n')
        sh_file.write('#SBATCH --account=co_nilah\n')
        sh_file.write('#SBATCH --partition=savio2_1080ti\n')
        sh_file.write('#SBATCH --qos=savio_lowprio\n')
        sh_file.write('#SBATCH --cpus-per-task=4\n')
        sh_file.write('#SBATCH --gres=gpu:1\n')
        sh_file.write('#SBATCH --requeue\n')
        sh_file.write('#SBATCH --time=02:10:00\n')
        sh_file.write(f'#SBATCH --job-name={file_name}\n')
        sh_file.write(f'#SBATCH --output=/global/scratch/users/sergiomar10/logs/ESMCBA_02032025/{file_name}_%j.out\n')
        sh_file.write(f'#SBATCH --error=/global/scratch/users/sergiomar10/logs/ESMCBA_02032025/{file_name}_%j.err\n')
        sh_file.write('source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh\n')
        sh_file.write('\n')
        sh_file.write('conda activate ESM_cambrian\n')
        sh_file.write('\n')

        sh_file.write(cmd + '\n')
        
    # Make the shell script executable
    os.chmod(sh_filepath, 0o755)
    
    print(f'Created shell script: {file_name}')
    # if 'HLA' == 
