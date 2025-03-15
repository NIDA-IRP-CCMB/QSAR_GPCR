import sys
import pandas as pd
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
sys.path.insert(0, core_dir)
from filters import *

def update_calc_pchembl_values(buffer, Verbose = False):
    nan_indexes = buffer.index[buffer['pchembl_value'].isna()]
    nan_indexes = nan_indexes.tolist()

    for i in range(len(nan_indexes)):
        try:
            index = nan_indexes[i]
            if buffer.loc[index, 'standard_units'] in ['M', 'mM', 'uM', 'nM', 'pM', 'fM'] and \
                    pd.notna(buffer.loc[index, 'standard_value']) and buffer.loc[index, 'standard_value'] != 0:

                buffer.loc[index, 'pchembl_value'] = calc_pscale(buffer['standard_value'][index],
                                                                             buffer['standard_units'][index])
        except Exception as e:
            print(f"An error occurred at index {nan_indexes[i]}: {e}")
    if Verbose:
        print('Number of pharmacological activity after calculating pchembl values:', len(buffer))

    return buffer
