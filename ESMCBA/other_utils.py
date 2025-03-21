# other_utils.py

from imports import *

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


def get_all_evaluations():
    evaluations_df = []

    for path in glob.glob('/global/scratch/users/sergiomar10/losses/ESMCBA_02032025/*.csv'):

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

    returan evaluations_df

