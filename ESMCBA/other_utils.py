# other_utils.py

from ESMCBA.imports import *

def split_dataset(df, test_size=0.2, random_state=42):
    """
    Example function that splits a DataFrame into train/test sets.
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_df, test_df

def compute_correlations(x, y):
    """
    Example function returning Spearman and Pearson correlations.
    """
    return spearmanr(x, y)[0], pearsonr(x, y)[0]


def get_all_evaluations(path = '/global/scratch/users/sergiomar10/losses/ESMCBA_02032025/*.csv'):
   
    evaluations_df = []

    for path in glob.glob(path):

        creation_time = os.path.getctime(path)
        creation_date = datetime.datetime.fromtimestamp(creation_time)

        df = pd.read_csv(path)

        num_evaluations = len(df)
        # Compute correlation metrics
        spearman_r, _ = spearmanr(df['measured'], df['prediction'])
        pearson_r, _ = pearsonr(df['measured'], df['prediction'])
        
        # Compute regression metrics
        mse = mean_squared_error(df['measured'], df['prediction'])
        mae = mean_absolute_error(df['measured'], df['prediction'])
        r2 = r2_score(df['measured'], df['prediction'])
        rmse = np.sqrt(mse)


        if '_MSE_' in path:
            loss = 'MSE'
        else:
            loss = 'Hubber' 
        
        name = path.split('_')
        if len(name) < 14:
            continue

        HLA = name[14]

        evaluations_df.append([
            HLA,
            loss,
            name[3],
            name[4],
            name[5],
            name[16],
            name[17],
            num_evaluations,
            spearman_r,
            pearson_r,
            mse,
            mae,
            r2,
            rmse,
            creation_date,
            path
        ])

    columns_to_name = ['HLA','Losses','encoding','data_prop','trained_blocks','lr_transformer','lr_regression','n_evaluations','spearman','pearsonr','mse','mae','r2','rmse','time','path']

    evaluations_df = pd.DataFrame(evaluations_df, columns=columns_to_name)

    return evaluations_df

def mhcflurry_predict(evaluations_dt_sorted):

    hla_sequences = []

    for HLA, path in evaluations_dt_sorted[['HLA', 'path']].values:
        dt = pd.read_csv(path)
        sequences = dt['sequence'].values
        HLA = HLA.replace('HLA','HLA-')
        for seq in sequences:

            if len(seq) < 8:
                continue
            if len(seq) > 15:
                continue
            
            hla_sequences.append([HLA, seq])
    hla_sequences = pd.DataFrame(hla_sequences, columns = ['HLA', 'sequence'])#.to_csv('hla_sequences.csv', index = False)

    # Directory to save the shell script files
    sh_file_dir = '/global/scratch/users/sergiomar10/slurm_jobs/MHCF_ESMCBA'
    os.makedirs(sh_file_dir, exist_ok=True)
    unique = hla_sequences['HLA'].unique()

    for allele in unique:

        allele, peptides = allele, hla_sequences[hla_sequences['HLA'] == allele].values.T[1]
        
        allele_clean = allele.replace('*', '').replace(':', '').replace(' ', '')
        run_id = f"EVAL_MHCFLURRY_FUSIONS_{allele_clean}"
        
        aa = set("ACDEFGHIKLMNPQRSTVWY")

    # Assume peptides is a list of peptide strings.
        # First, remove occurrences of '<unk>' then filter out any non-amino acid letters.
        cleaned_peptides = [
            ''.join(ch for ch in peptide.replace('<unk>', '') if ch.upper() in aa)
            for peptide in peptides
        ]

    # Convert the cleaned peptides list into a NumPy array if needed.
        peptides = np.array(cleaned_peptides)

        # Convert the peptides array into a plain string without extra quotes.
        # You can choose a delimiter; here we use a space.
  
  
        peptides_str = " ".join(peptides.tolist())
        output_dir = '/global/scratch/users/sergiomar10/ESMCBA/ESMCBA/performances/benchmark/'
        fasta_filename = os.path.join(output_dir, f"{allele_clean}.pep")
        with open(fasta_filename, "w") as fout:
            for idx, peptide in enumerate(peptides, 1):
                fout.write(f">pep{idx}\n{peptide}\n")
        
        print(f"Saved {len(peptides)} peptides for allele {allele} in {fasta_filename}")
        # Create the command. Now the --peptides argument is a plain string.
        sql_query = f"""mhcflurry-predict --alleles {allele_clean} --peptides {peptides_str} --out /global/scratch/users/sergiomar10/ESMCBA/ESMCBA/performances/benchmark/MHCFlurry/MHCFlurry{allele_clean}_mhc_flurry.csv
    """
        try:
            # Create the full path for the SLURM job script.
            sh_filepath = os.path.join(sh_file_dir, f"{run_id}.sh")
            
            # Write the SLURM script.
            with open(sh_filepath, 'w') as sh_file:
                sh_file.write('#!/bin/bash\n')
                sh_file.write('#SBATCH --account=co_nilah\n')
                sh_file.write('#SBATCH --partition=savio3_gpu\n')
                sh_file.write('#SBATCH --qos=savio_lowprio\n')
                sh_file.write('#SBATCH --cpus-per-task=4\n')
                sh_file.write('#SBATCH --gres=gpu:1\n')
                sh_file.write('#SBATCH --time=00:20:00\n')
                sh_file.write(f'#SBATCH --job-name={run_id}\n')
                sh_file.write(f'#SBATCH --output=/global/scratch/users/sergiomar10/logs/MHCFlurry_evals/{run_id}_%j.out\n')
                sh_file.write(f'#SBATCH --error=/global/scratch/users/sergiomar10/logs/MHCFlurry_evals/{run_id}_%j.err\n')
                sh_file.write('\n')
                sh_file.write('source /clusterfs/nilah/sergio/miniconda3/etc/profile.d/conda.sh\n')
                sh_file.write('conda activate /clusterfs/nilah/sergio/miniconda3/envs/cooltools\n')
                sh_file.write('\n')
                sh_file.write("cd /global/scratch/users/sergiomar10/data/MHCFlurry_evals\n")
                sh_file.write(sql_query)
            
            # Make the SLURM script executable.
            os.chmod(sh_filepath, 0o755)
        except Exception as e:
            print(f"Error creating script for {run_id}: {e}")
            
        # break

    print('Done.')
