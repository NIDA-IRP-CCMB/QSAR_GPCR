import sys
import numpy as np
import pandas as pd
import itertools

# Storing the arguments
start = int(sys.argv[1]) # e.g. 0
end = int(sys.argv[2]) # e.g. 1000

# Define parameter lists
dict_para = {
    'epochs': [160, 640, 1280],
    'hidden_layers': [1, 2],
    'neurons': [6144, 7168, 8192],
    'learning_rate': [10**(-5.5), 1e-05, 10**(-4.5)],
    'batch_size': [256, 512, 1024],
    'dropout': [0.0, 0.1]
}

# Create a list of all unique combinations
all_combinations = list(itertools.product(
    dict_para['epochs'],
    dict_para['hidden_layers'],
    dict_para['neurons'],
    dict_para['learning_rate'],
    dict_para['batch_size'],
    dict_para['dropout']
))

# Ensure 'end' is within the number of unique combinations
if end > len(all_combinations):
    raise ValueError("Requested number of samples exceeds the total number of unique combinations.")

# Randomly sample the desired number of unique combinations
np.random.seed(2023)
sampled_combinations = np.random.choice(range(len(all_combinations)), end, replace=False)
sampled_combinations = [all_combinations[i] for i in sampled_combinations]

# Create DataFrame
cols = ['epochs', 'hidden_layers', 'neurons', 'learning_rate', 'batch_size', 'dropout']
df = pd.DataFrame(sampled_combinations, columns=cols)

# Verify no duplicates
assert df.duplicated().any() == False

# name index, then move it to first col
# df['index_col'] = df.index
# last_col = df.pop('index_col')
# df.insert(0, 'index_col', last_col)
# remove index 0 to 'start' variable set initially
df = df.drop(df.index[:start])
# write combinations to file
df.to_csv('parameters_list.txt', header=True, index=False, sep=' ', mode='w')
