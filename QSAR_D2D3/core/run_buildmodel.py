#!/usr/bin/python


## define enviroment
import sys, os
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
conf_dir = core_dir+"/keywords"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)


## import all
from filters import *
from buildmodel import *
from misc import *

import getopt



# initialize global variables
stages = ["buildmodel", "buildmodel_chiral", "same_buildmodel", "shap", "prediction", "prediction_chiral", "train_data"]
# settings = 0  # 0 is buildmodel only, 1 is buildmodel + shap, 2 is shap analysis only (models already in), 3 is internal prediction only


def show_msg(arg):
    msg = ''' 
        
        -s <stage>              buildmodel, buildmodel_chiral, prediction, shap, same_buildmodel
                                buildmodel --> building model (using new folder)
                                buildmodel_chiral --> includes chiral descriptors
                                same_buildmodel --> retrain data and rebuild model (using same folder)
                                prediction --> making predictions for either internal or external
                                prediction_chiral --> for tp=0; includes chiral descriptors
                                shap --> making shap predictions (once models are finished)
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
    indexes = None

    try:
        opts, args = getopt.getopt(argv,"hs:m:x:t:r:n:e:i:d:",
                                   ["stage=", "model=", "method=", "testing_percent=", "n_rand_stat",
                                    "num_splits=", "indexes=", "in_filename=", "in_validation_set="])

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
        elif opt in ('-n', '--num_splits'):   # number of random_splits for splitting dataset into training/testing
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

    if in_validation_set is None and stage.startswith('p') and tp is 0:
        show_msg('Need -d option')
        sys.exit()
    if in_filename is None and stage.startswith('b'):
        show_msg('Need -i option')
        sys.exit()

    return stage, mode, method, tp, num_splits, n_rand_stat, indexes, in_filename, in_validation_set


if __name__ == "__main__":
    # return argvs 
    stage, mode, method, tp, num_splits, n_rand_stat, indexes, in_filename, in_validation_set = main(sys.argv[1:])

    chiral_descs = False # Defaults to not using chiral descriptors

    print("Stage:", stage)
    # initializing
    if stage in ["buildmodel", "buildmodel_chiral", "same_buildmodel", "shap", "train_data"]:
        # define output_dir for output & log file
        output_dir = get_output_dir(mode, method, tp)
        writer = MyWriter(sys.stdout, output_dir+'.log')
        sys.stdout = writer

        # check required files/directories
        in_dataset_dir = get_dir(in_filename)
        check_misc(in_dataset_dir)

        if stage == "buildmodel_chiral":
            chiral_descs = True

        if stage == "buildmodel" or stage == "buildmodel_chiral":
            check_required(in_filename, output_dir) # makes new directory
        elif stage == "same_buildmodel" or stage == "same_buildmodel_chiral":
            check_required(in_filename, output_dir, keep_old = False) # keeps same directory
        print("indexes", indexes)

        if indexes is None:
            print("-e option not added")
            # number of rand_splits/rand_states
            rand_splits = gen_random_splits(control_seed=2020, num_splits=num_splits)
            if tp == 0.0:
                rand_splits = [rand_splits[0]]
            rand_states = gen_random_splits(control_seed=501, num_splits=n_rand_stat)
        elif indexes is not None:
            rand_states, rand_splits = rand(num_splits, n_rand_stat, indexes)  # e.g. rand_splits[index1:index2]
    # actual runs
    if stage in ["buildmodel", "buildmodel_chiral", "same_buildmodel", "train_data"]:
        # loop do build models
        for rand_state in rand_states:
            for random_split in rand_splits:
                print("(rand_state)_(random_split)", f"{rand_state}_{random_split}")
                output_ext = get_output_ext(mode, method, tp, rand_state, random_split)

                # if training dict_data exists (which is created at the end of the data generation), skip training data
                filename = f'{output_dir}/dict_data_{rand_state}_{random_split}.pickle'
                if os.path.isfile(filename) and stage in ["buildmodel", "buildmodel_chiral", "same_buildmodel"]:
                    print("Training data exists. Retrieving from previous data")
                    train_descs, test_descs, train_acts, test_acts, train_names, test_names, \
                        topo_names, phore_names, feature_names = retrieve_training_data(filename, tp)
                # else get the training data
                else:
                    print("Generating training data")
                    train_descs, train_acts, train_names, test_descs, test_acts, test_names = \
                        get_training_dataset_ML(in_filename, mode, method, tp, output_dir, rand_state, random_split, stage, chiral_descs)

                if stage in ["buildmodel", "buildmodel_chiral", "same_buildmodel"]:
                    # buildmodel
                    print("buildmodel started")
                    model, model_score, best_params = build_model(mode, method, rand_state, train_descs, train_acts, output_dir, output_ext)
                    print("buildmodel finished")

                    if tp > 0:
                        print("prediction started")
                        if mode.startswith('reg'):
                            test_r2, test_rmse, test_mse = predict_model(model, train_names, train_acts, test_descs, test_acts, test_names,\
                                                                        random_split, output_dir, mode, method, rand_state)
                        elif mode.startswith('class'):
                            predictions, acc = predict_model(model, train_names, train_acts, test_descs, test_acts, test_names,\
                                                             random_split, output_dir, mode, method, rand_state)
                        print("prediction finished")
    elif stage == "shap":
        # define output_dir for output & log file
        output_dir = get_output_dir(mode, method, tp)
        writer = MyWriter(sys.stdout, output_dir + '.log')
        sys.stdout = writer

        # check required files/directories
        in_dataset_dir = get_dir(in_filename)
        check_misc(in_dataset_dir)
        # number of rand_splits/rand_states
        # rand_splits = gen_random_splits(control_seed=2020, num_splits=num_splits)
        # if tp == 0.0:
        #     rand_splits = [rand_splits[0]]
        # rand_states = gen_random_splits(control_seed=501, num_splits=n_rand_stat)

        for rand_state in rand_states:
            for random_split in rand_splits:
                print("(rand_state)_(random_split)", f"{rand_state}_{random_split}")
                output_ext = get_output_ext(mode, method, tp, rand_state, random_split)

                # Loading back data and model
                filename = f'{output_dir}/dict_data_{rand_state}_{random_split}.pickle'
                train_descs, test_descs, train_acts, test_acts, \
                                train_names, test_names, topo_names, phore_names, feature_names = retrieve_training_data(filename, tp)
                model = pickle.load(open(output_dir + "/model_%s.dat" % output_ext, "rb"))

                # Check the settings flag and conditionally run the feature importance analysis
                from feature_importance import FeatureImportance
                # Create an instance of FeatureImportance and run the analysis
                print("SHAP analysis")
                feature_imp = FeatureImportance(method, model, train_descs, feature_names, output_dir,
                                                rand_state, random_split, output_ext)
                # get SHAP values for the current model
                feature_imp.run_analysis()
                print("SHAP finished")

    if stage == "prediction" or stage == "prediction_chiral":
        if tp > 0:  # internal prediction, if you want to rerun the internal predictions for 85% models
            rand_splits = gen_random_splits(control_seed=2020, num_splits=num_splits)
            rand_states = gen_random_splits(control_seed=501, num_splits=n_rand_stat)
            for rand_state in rand_states:
                for random_split in rand_splits:
                    print("(rand_state)_(random_split)", f"{rand_state}_{random_split}")
                    output_dir = get_output_dir(mode, method, tp)
                    output_ext = get_output_ext(mode, method, tp, rand_state, random_split)

                    # Loading back data and model
                    filename = f'{output_dir}/dict_data_{rand_state}_{random_split}.pickle'
                    train_descs, test_descs, train_acts, test_acts, \
                    train_names, test_names, topo_names, phore_names, feature_names = retrieve_training_data(filename, tp)
                    model = pickle.load(open(output_dir + "/model_%s.dat" % output_ext, "rb"))
                    if mode.startswith('reg'):
                        test_r2, test_rmse, test_mse = predict_model(model, test_descs, test_acts, test_names,\
                                    random_split, output_dir, mode, method, rand_state)
                    elif mode.startswith('class'):
                        predictions, acc = predict_model(model, train_names, train_acts, test_descs, test_acts,
                                                         test_names, \
                                                         random_split, output_dir, mode, method, rand_state)

        if stage == "prediction_chiral":
            chiral_descs = True

        if tp == 0: # external prediction
            # hacking ARGV to old format
            in_model_dir = get_output_dir(mode, method, tp)
            splits = in_validation_set.split('/')
            in_validation_dir = get_dir(in_validation_set)
            in_validation_filename = splits[-1]

            output_dir = 'pred_' + get_output_dir(mode, method, tp)
            result_list = []
            # number of rand_state

            rand_states = gen_random_splits(control_seed=501, num_splits=n_rand_stat)
            rand_splits = gen_random_splits(num_splits=num_splits)

            df_pred = pd.DataFrame()
            count = 0


            for rand_state in rand_states:
                for random_split in rand_splits:
                    filename = f'{in_model_dir}/dict_data_{rand_state}_{random_split}.pickle'
                    train_descs, test_descs, train_acts, test_acts, \
                        train_names, test_names, topo_names, phore_names, feature_names = retrieve_training_data(filename, tp)

                    output_ext = get_output_ext(mode, method, tp, rand_state, random_split)

                    input_data = read_mols(mode, method, in_validation_filename, output_ext, datadir=in_validation_dir, modeldir=in_model_dir)
                    molnames = input_data['molnames']
                    mols = input_data['molecules']
                    model = input_data['model']
                    inds = input_data['inds']

                    # if mode.startswith('reg') and method == 'xgb':
                    sigbits = input_data['sigbits']
                    ad_fps = input_data['ad_fps']
                    ad_radius = input_data['ad_radius']
                    feature_names = input_data['feature_names']
                    # Check Applicability Domain
                    appdom_results = check_appdom(ad_fps, ad_radius, mols, molnames, step=stage)
                    mols = appdom_results['test_mols']
                    molnames = appdom_results['test_names']
                    molecules_rej = appdom_results['rej_mols']
                    molnames_rej = appdom_results['rej_names']
                    descriptors = calc_topo_descs(mols, inds)
                    if chiral_descs:
                        chir_descs = calc_chir_descs(mols)
                        descriptors = np.concatenate((descriptors, chir_descs), axis=1)
                    phore_descriptors = calc_phore_descs(mols, sigbits)
                    descriptors = np.concatenate((descriptors, phore_descriptors), axis=1)

                    if mode.startswith('reg'):
                        pred_results = make_preds(molnames, descriptors, model, random_split, mode=mode)
                        result_list.append(pred_results['predictions'])
                        # individual predictions, to be saved separately
                        pred_results_predictions = []
                        for i in pred_results['predictions']:
                            pred_results_predictions.append(i)  # pred_results['predictions'] list without the []
                        df_pred[f'split_{count}'] = pred_results_predictions
                        count += 1
                    elif mode.startswith('class'):
                        filepath = in_validation_set + '.act'
                        if os.path.exists(filepath):
                            df_act = pd.read_csv(filepath, header=None, sep='\t')
                            y_true = df_act[1]
                            pred_results = make_preds(molnames, descriptors, model, y_true, mode=mode)
                        else:
                            pred_results = make_preds(molnames, descriptors, model, mode=mode)
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

            ## TODO, the following will case issue. if you have input directory as tetst/, it will return nothing as output_filename
            output_filename = in_model_dir.split('/')[-1]
            # merge dataframe with experimental data
            print('in_validation_filename', in_validation_filename)
            try:
                # if mode.startswith('reg'):
                df_act = pd.read_csv(in_validation_set + '.act', header=None, sep='\t')
                # elif mode.startswith('class'):
                print(in_validation_set, 'in_validation_set')
                df_act = pd.read_csv(in_validation_set + '.act', header=None, sep='\t')
                df_act.columns = ['compound', 'exp_mean']
                df = df.merge(df_act, on='compound')
                print('merging df with experimental df')
            except:
                print('.act file missing, standalone predictions is outputted')
            # df.to_csv(output_dir + '/' + output_filename, header=True, index=False, sep='\t')

            # individual predictions
            df2 = pd.concat([df, df_pred], axis=1)
            df2.to_csv(output_dir + '/pred_splits_'+in_validation_filename, header=True, index=False, sep='\t')

