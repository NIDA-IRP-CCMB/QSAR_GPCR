#!/usr/bin/python

## define environment
import sys, os
from pathlib import Path
# Repository - General folder, with Machine Learning codes
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
conf_dir = core_dir+"/conf"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)
from filters import *
from buildmodel import *
from misc import *
# Repository - Dopamine Receptor, with deep learning codes
core_DR = home+'/repositories/ai-DR/core'
sys.path.insert(0, core_DR)
from deeplearning import *
import time
from descriptor_setup import dnames, dlist
import getopt

# initialize global variables
stages = ['buildmodel', 'buildmodel_chiral', 'prediction', 'prediction_chiral','shap', "train_data"]

def show_msg(arg):
    msg = ''' 
        -s <stage>              buildmodel, prediction, or shap
        -m <model>              regression or classification model
        -x <method>             xgb or rf (random forest) machine learning methods
        -t <testing_percent>    proportion of dataset to reserve for testing; decimal range [0,1)
        -r <random_state>       number of random states/seeds to use to build the model; int range [1,inf)
        -n <num_splits>         number of times to randomly split your data into a testing set; int range [1,inf)
        -i <input_basename>     location of your input data with its basename (without ext ex. .act or .smi)
        -d <validation_data>    dataset used to benchmark the model prediction

        If -t is set to 0, then -n must be set to 1 as there is only one way to split the data into 100% training
        and 0% testing data. If -n is greater than 1, then -t must be in range (0,1).

        If stage is set to do prediction then -d <validation_data> must also be given. Can input nothing for -d if build
        model.
        '''

    print(arg + msg)
    sys.exit()

def main(argv):

    in_validation_set = None
    in_filename = None

    try:
        opts, args = getopt.getopt(argv,"hs:m:x:t:r:n:e:i:d:",
                                   ["stage=", "model=", "method=", "testing_percent=", "n_rand_stat", "num_splits=",
                                    "indexes=", "in_filename=", "in_validation_set="])
    except getopt.GetoptError:
        show_msg(sys.argv[0])

    if len(opts) == 0:
        show_msg(sys.argv[0])

    for opt, arg in opts:
        if opt == '-h':
            show_msg(sys.argv[0])
        elif opt in ("-s", "--stage"):
            stage = arg.lower()
            if stage not in stages:
                show_msg("Invalid stage option")
        elif opt in ("-m", "--model"):
            mode = arg.lower()
        elif opt in ("-x", "--method"):
            method = arg.lower()
        elif opt in ('-t', '--tp'):      # testing dataset ratio
            tp = float(arg)
            if tp < 0 or tp >= 1:
                show_msg("-t option must be set as a decimal between 0 inclusive and 1")
        elif opt in ('-r', '--n_rand_stat'):  # number of random_stat for machine learning XGB/RF
            n_rand_stat = int(arg)
            if n_rand_stat < 1:
                show_msg("-r option must be at least 1")
        elif opt in ('-n', '--num_splits'):  # number of random_splits for splitting dataset into training/testing
            num_splits = int(arg)
            if num_splits < 1:
                show_msg("-n option must be at least 1")
        elif opt in ('-e', '--indexes'):  # index of models we want to build, in str, e.g. '0:3'
            indexes = arg
            if not ":" in indexes:
                show_msg("-r option must contain ':'")
        elif opt in ("-i", "--in_filename"):   # input filename, eg. dir/filename
            in_filename = arg
            if not os.path.exists(get_dir(arg)):
                show_msg("Input filename path does not exist")
        elif opt in ("-d", "--in_validation_set"):
            in_validation_set = arg

    if in_validation_set is None and stage.startswith('p'):
        show_msg('Need -d option')
        sys.exit()
    if in_filename is None and stage.startswith('b'):
        show_msg('Need -i option')
        sys.exit()

    return stage, mode, method, tp, num_splits, n_rand_stat, indexes, in_filename, in_validation_set


if __name__ == "__main__":
    stage, mode, method, tp, num_splits, n_rand_stat, indexes, in_filename, in_validation_set = main(sys.argv[1:])
    print("Stage:", stage)

    chiral_descs = False  # Defaults to not using chiral descriptors

    if stage in ["buildmodel", "buildmodel_chiral", "shap", "train_data"]:
        if stage == "buildmodel_chiral":
            chiral_descs = True

        train_dataset_prefix = in_filename
        output_dir = get_output_dir(mode, method, tp)  # reg_dnn_0.15
        check_misc(get_dir(train_dataset_prefix))  # checking and adding appropriate directories
        check_output_dir(output_dir, keep_old=False)
        rand_states, rand_splits = rand(num_splits, n_rand_stat, indexes)  # e.g. rand_splits[index1:index2]
        for rand_state in rand_states:
            for random_split in rand_splits:
                output_ext = get_output_ext(mode, method, tp, rand_state, random_split)
                print('output_ext', output_ext)
                print("dataset info", train_dataset_prefix)
                if stage in ["buildmodel", "buildmodel_chiral", "train_data"]:
                    train_names, test_names, train_descs, train_acts, test_descs, test_acts, topo_names, phore_names\
                        = get_training_dataset(mode, tp, stage, train_dataset_prefix, output_dir, output_ext, rand_state,
                                               random_split, all_descs=True, extra_features = True, remove_ipc = True,
                                               chiral_descs=chiral_descs)
                    if tp > 0:
                        train_descs_norm, test_descs_norm = convert_to_zscore2(train_descs, test_descs)
                    else:
                        train_descs_norm, test_descs_norm = convert_to_zscore2(train_descs) # Only pass train_descs for tp=0.00

                    # Define the file paths
                    train_descs_file = output_dir + "/train_descs_%s.dat" % output_ext
                    feature_names_file = output_dir + "/feature_names_%s.dat" % output_ext
                    with open(train_descs_file, "wb") as train_descs_file_obj:
                        pickle.dump(train_descs_norm, train_descs_file_obj)
                    with open(feature_names_file, "wb") as feature_names_file_obj:
                        feature_names = topo_names + phore_names
                        pickle.dump(feature_names, feature_names_file_obj)

                    # save to pickle file
                    dict_data = {'train_descs_norm': train_descs_norm, 'test_descs_norm': test_descs_norm,
                                 'train_descs': train_descs, 'test_descs': test_descs, "test_names": test_names,
                                 'train_acts': train_acts, 'test_acts': test_acts, 'train_names': train_names,
                                 'rand_state': rand_state, 'random_split': random_split, "topo_names": topo_names,
                                 "phore_names": phore_names, "feature_names": feature_names
                                 }
                    with open(f'{output_dir}/dict_data_{rand_state}_{random_split}.pickle', 'wb') as handle:
                        pickle.dump(dict_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    if stage in ["buildmodel", "buildmodel_chiral"]:
                        best_model, best_params, best_score, model_time = \
                            buildmodel_RandomSearchCV(mode, rand_state, train_descs_norm, train_acts, output_ext)
                        if tp > 0:
                            predict_model(best_model, test_names, test_descs_norm, test_acts, output_ext)

                if stage in ["buildmodel", "buildmodel_chiral", "shap"]:
                    filename = f'{output_dir}/dict_data_{rand_state}_{random_split}.pickle'
                    train_descs, test_descs, train_descs_norm, test_descs_norm, train_acts, test_acts, train_names, \
                        test_names, topo_names, phore_names, feature_names = retrieve_training_data(filename, tp)

                    # reload model
                    model = keras.models.load_model(f'{output_dir}/model_{output_ext}')
                    from feature_importance import FeatureImportance
                    # Create an instance of FeatureImportance and run the analysis
                    start_time = time.time()
                    print("Started SHAP")
                    feature_imp = FeatureImportance(method, model, train_descs_norm, feature_names, output_dir,
                                                    rand_state, random_split, output_ext)
                    # get SHAP values for the current model
                    feature_imp.run_analysis()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print("Time taken:", elapsed_time, "seconds")

                if tp > 0:
                    if mode.startswith('reg'):
                        predict_model(best_model, test_names, test_descs_norm, test_acts, output_ext)
                    elif mode.startswith('class'):
                        predict_model(best_model, test_names, train_acts, test_descs_norm, test_acts,
                                                         output_dir, output_ext)

    if stage == "prediction" or stage == "prediction_chiral":
        print("Prediction stage")
        output_dir = get_output_dir(mode, method, tp)
        check_misc(get_dir(in_validation_set))  # checking and adding appropriate directories
        rand_states, rand_splits = rand(num_splits, n_rand_stat, indexes)
        df = pd.DataFrame()
        df_pred = pd.DataFrame()
        result_list = []
        count = 0

        if stage == "prediction_chiral":
            chiral_descs = True

        for rand_state in rand_states:
            for random_split in rand_splits:
                # retrieve training dataset built during buildmodel, tp = 0.15 stage
                filename = f'{output_dir}/dict_data_{rand_state}_{random_split}.pickle'
                train_descs, test_descs, train_descs_norm, test_descs_norm, train_acts, test_acts, train_names, test_names, \
                topo_names, phore_names, feature_names = retrieve_training_data(filename, tp)

                # reload model
                output_ext = get_output_ext(mode, method, tp, rand_state, random_split)
                model = keras.models.load_model(f'{output_dir}/model_{output_ext}')
                if tp > 0:
                    if mode.startswith('reg'):
                        predict_model(model, test_names, test_descs_norm, test_acts, output_ext)
                    elif mode.startswith('class'):
                        predict_class(model, test_names, train_acts, test_descs_norm, test_acts, output_dir, output_ext)
                elif tp == 0:
                    validation_basename = in_validation_set.split('/')[-1]
                    in_validation_dir = get_dir(in_validation_set)
                    output_ext = get_output_ext(mode, method, tp, rand_state, random_split)
                    input_data = read_mols_dnn(mode, method, output_ext, validation_basename, datadir=in_validation_dir,
                                               modeldir=output_dir)
                    molnames = input_data['molnames']
                    mols = input_data['molecules']
                    #inds = input_data['inds']
                    sigbits = input_data['sigbits']
                    ad_fps = input_data['ad_fps']
                    ad_radius = input_data['ad_radius']
                    # feature_names = input_data['feature_names']
                    # Check Applicability Domain
                    appdom_results = check_appdom(ad_fps, ad_radius, mols, molnames, step=stage)
                    mols = appdom_results['test_mols']
                    molnames = appdom_results['test_names']
                    molecules_rej = appdom_results['rej_mols']
                    molnames_rej = appdom_results['rej_names']

                    # validation descriptors
                    descriptors = calc_topo_descs(mols)
                    if chiral_descs:
                        chir_descs = calc_chir_descs(mols)
                        descriptors = np.concatenate((descriptors, chir_descs), axis=1)
                    phore_descriptors = calc_phore_descs(mols, sigbits)
                    descriptors = np.concatenate((descriptors, phore_descriptors), axis=1)

                    if "Ipc" not in topo_names:
                        feature_index = dnames.index("Ipc")
                        descriptors = np.delete(descriptors, feature_index, axis=1)
                    train_descs_norm, descriptors_norm = convert_to_zscore2(train_descs, descriptors)

                    if mode.startswith('reg'):
                        pred_results = make_preds(molnames, descriptors_norm, model, random_split, mode=mode)
                        result_list.append(pred_results['predictions'])
                        pred_results_predictions = []
                        for i in pred_results['predictions']:
                            pred_results_predictions.append(i[0])  # pred_results['predictions'] list without the []
                        df_pred[f'split_{count}'] = pred_results_predictions
                        count += 1
                    elif mode.startswith('class'):
                        filepath = in_validation_set + '.act'
                        if os.path.exists(filepath):
                            df_act = pd.read_csv(filepath, header=None, sep='\t')
                            y_true = df_act[1]
                            pred_results = make_preds(molnames, descriptors_norm, model, y_true, mode=mode)
                        else:
                            pred_results = make_preds(molnames, descriptors_norm, model, mode=mode)
                        if result_list == []:
                            result_list = pred_results['predictions']
                        else:
                            result_list += pred_results['predictions']

        if mode.startswith('reg'):
            compound, pred_mean, pred_error = summarize_preds(molnames, result_list)
            data = {"compound": compound, "mean": pred_mean, "stdev": pred_error}
        elif mode.startswith('class'):
            activity = []
            for prediction in result_list:
                if prediction > 0.5 * len(rand_splits):
                    activity.append(1)
                else:
                    activity.append(0)
            data = {"compound": molnames, "sum_activity": list(result_list), "pred_activity": activity}

        df = pd.DataFrame(data=data)

        # makes the output directory if it does not already exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_filename = in_validation_set.split('/')[-1]

        try:
            # merge dataframe with experimental data
            df_act = pd.read_csv(in_validation_set + '.act', header=None, sep='\t')
            df_act.columns = ['compound', 'exp_mean']
            df = df.merge(df_act, on='compound')
        except:
            print("skip merging dataframe with experimental data")


        check_output_dir(f'pred_{output_dir}')
        df.to_csv(f'pred_{output_dir}/{output_dir}', header=True, index=False, sep='\t')
        # individual predictions included
        df2 = pd.concat([df, df_pred], axis = 1)
        df2.to_csv(f'pred_{output_dir}/pred_splits', header=True, index=False, sep='\t')
