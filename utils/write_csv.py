from pandas.core.frame import DataFrame


def write_performance(performance, type, dir):
    df = DataFrame(performance)
    if type == 'pred':
        df_dir = dir + 'pred.csv'
    elif type == 'actual':
        df_dir = dir + 'actual.csv'
    else:
        df_dir = dir + 'cog.csv'
    df.to_csv(df_dir, sep='\t',header=False, index = False)
