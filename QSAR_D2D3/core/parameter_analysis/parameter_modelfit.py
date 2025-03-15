import sys, os
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
conf_dir = core_dir+"/conf"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)
DR_dir = home+'/repositories/ai-DR/core'
sys.path.insert(0, DR_dir)
import os.path
from deeplearning import *
import pickle

i = int(sys.argv[1])
print(i)


def build_dnn_model(mode, rand_state, hidden_layers, neurons, dropout, learning_rate, num_features):
    set_tf_seed(rand_state)
    model = keras.Sequential()

    model.add(layers.InputLayer(input_shape=(num_features,)))

    # hidden layers
    for i in range(hidden_layers):
        model.add(layers.Dense(neurons, activation='relu'))  # tanh? relu?
        if i < hidden_layers - 1:
            model.add(layers.Dropout(rate=dropout, seed = rand_state))
    # output layer
    if mode == 'reg':
        model.add(layers.Dense(1))
        model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
    elif mode == 'class':
        model.add(layers.Dense(1, activation="sigmoid"))
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
    return model


# initial values
stage = 'buildmodel'
mode = 'reg'
method = 'dnn'
tp = 0.15
num_splits = 10
n_rand_stat = 35

# while True:
  # Make sure parameter file exists or isn't blank. Read parameter file in to choose a combination from
filename_list = f'parameters_list.txt'
if os.path.isfile(filename_list):
    df = pd.read_csv(filename_list, sep=' ')
else:
    raise SystemExit(filename_list + " does not exist. Please create a list using parameter_lists.py")
if len(df) == 0:
    raise SystemExit(filename_list + " is blank. Modelbuilding is finished?")

# retrieve prepared training data
with open(f'../dict_data.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)

train_descs_norm = dict_data['train_descs_norm']
test_descs_norm = dict_data['test_descs_norm']
train_acts = dict_data['train_acts']
test_acts = dict_data['test_acts']
train_names = dict_data['train_names']
rand_state = dict_data['rand_state']
random_split = dict_data['random_split']

# randomly choose a row from the text file
# random_int = random.randint(0, len(df))
sample = df.iloc[i-1]
# print(random_int, list(sample))
epochs = int(sample['epochs'])
hidden_layers = int(sample['hidden_layers'])
neurons = int(sample['neurons'])
learning_rate = sample['learning_rate']
batch_size = int(sample['batch_size'])
dropout = sample['dropout']
sample = list(sample)
print('building', sample)

# # double check that we're not repeating a parameter that's already been built and added to results.txt
filename_results = f'results_list.txt'
# if os.path.isfile(filename_results):
#     df_results = pd.read_csv(filename_results, header = None, sep = ' ')
#     cols = df.columns
#     cols = cols.insert(0, "i")
#     df_results = df_results.iloc[:, :len(cols)]
#     df_results.columns = cols
#     df_results = df_results[df.columns]
#     results_list = df_results.values.tolist() # 2D array of the dfs
#     for x in results_list:
#         if x == sample:
#             raise SystemExit(str(sample) + ' already built and in the results list.')

# # remove parameter from the original list
# df = df.drop(index=[random_int])
# df.to_csv(filename_list, header=True, index=False, sep=' ', mode='w')

# model building
start_time = time.time()
model = build_dnn_model(mode, rand_state, hidden_layers, neurons, dropout, learning_rate, train_descs_norm.shape[1])
model.fit(train_descs_norm, train_acts, epochs=epochs, batch_size=batch_size, verbose=0)
end_time = time.time()
minutes = (end_time - start_time) / 60

# prediction
y_true, y_pred = test_acts, model.predict(test_descs_norm).flatten()
R2, Rp, Rs, MSE, RMSE = stats(y_true, y_pred)


# write results to file
with open(filename_results, 'a') as f:
    f.write(f'{i} {epochs} {hidden_layers} {neurons} {learning_rate} {batch_size} {dropout} {minutes} {R2} {Rp} {Rs} {MSE} {RMSE}')
    f.write('\n')
