# CPU only: print the per-run CSV files written by nb08b-eval (output files cannot be downloaded from this network).
import glob, os
import pandas as pd
pd.set_option('display.width', 400); pd.set_option('display.max_columns', 80); pd.set_option('display.max_rows', 500)
files = sorted(glob.glob('/kaggle/input/**/nb08/*.csv', recursive=True))
print('files:', files)
for f in files:
    df = pd.read_csv(f)
    print('\n=====', os.path.basename(f), df.shape)
    print(df.to_string(index=False))
print('PRINT FINISHED')
