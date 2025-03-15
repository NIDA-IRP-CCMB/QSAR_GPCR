"""
    SHAP feature analysis code. This tool combines SHAP explanations from several models.
    Currently, this works for models built using 100% data to explain how the model makes
    predictions from the training data. Results from this analysis can be interpreted as
    the consensus SHAP interpretation. This currently handles methods from XGBoost, RF,
    and DNN.
"""

import os
import pandas as pd
import numpy as np
import keras
import pickle
import shap
import shutil

def get_combined_descs_and_featnames(mods, tp, base_path):
    if tp != '0.00':
        raise ValueError("Can't handle different datasets yet for consensus.")

    # just take one since it will be the same for both rf and xgb as well as between seeds
    ml_descs_path = os.path.join(base_path, f'reg_xgb_{tp}', f'train_descs_87_864.dat')
    ml_featnames_path = os.path.join(base_path, f'reg_xgb_{tp}', f'feature_names_87_864.dat')
    with open(ml_descs_path, 'rb') as file:
        ml_descs = pickle.load(file)
    with open(ml_featnames_path, 'rb') as file:
        ml_featnames = pickle.load(file)
    df_ml = pd.DataFrame(ml_descs, columns=ml_featnames)

    dnn_descs, dnn_featnames = None, None

    if 'dnn' in mods:
        # just take one since it will be the same between seeds
        dnn_descs_path = os.path.join(base_path, f'reg_dnn_{tp}', f'train_descs_reg_dnn_{tp}_87_864.dat')
        dnn_featnames_path = os.path.join(base_path, f'reg_dnn_{tp}', f'feature_names_reg_dnn_{tp}_87_864.dat')
        with open(dnn_descs_path, 'rb') as file:
            dnn_descs = pickle.load(file)
        with open(dnn_featnames_path, 'rb') as file:
            dnn_featnames = pickle.load(file)
        df_dnn = pd.DataFrame(dnn_descs, columns=dnn_featnames)

        combined_columns = df_dnn.columns.union(df_ml.columns)
        df_ml_reindexed = df_ml.reindex(columns=combined_columns)
        combined_df = df_ml_reindexed.combine_first(df_dnn)
        combined_descs = combined_df.values
        combined_featnames = combined_df.columns.tolist()
    else:
        combined_descs = ml_descs
        combined_featnames = ml_featnames

    return combined_descs, combined_featnames, dnn_descs, dnn_featnames, ml_descs, ml_featnames


def get_all_shap(label, mods, tp, base_path, dnn_descs=None, dnn_featnames=None,
                 ml_descs=None, ml_featnames=None, updateDNN=False, updateML=False):
    print(f'... Processing {label}')

    shap_values_all = {}
    shap_explainers_dir = os.path.join(base_path, f'shap_explainers_{tp}')
    os.makedirs(shap_explainers_dir, exist_ok=True)

    for mod in mods:
        print(f'Processing {mod} models')
        model_dir = f'{base_path}/reg_{mod}_{tp}'

        if mod == 'dnn':
            model_dirs = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))
                          and d.startswith('model')]
            for sub_dir in model_dirs:
                model_path = os.path.join(model_dir, sub_dir)
                key = f'{mod}_{sub_dir.split("_")[-2]}'
                split_state = f'{sub_dir.split("_")[-2]}'
                rand_state = f'{sub_dir.split("_")[-1]}'

                built_explainer = os.path.join(model_dir, f'explainer_{mod}_{split_state}_{rand_state}_shap.pkl')
                if os.path.exists(built_explainer):
                    shap_file_path = os.path.join(shap_explainers_dir,
                                                  f'explainer_{mod}_{split_state}_{rand_state}_shap.pkl')
                    shutil.copy(built_explainer, shap_file_path)
                else:
                    shap_file_path = os.path.join(shap_explainers_dir,
                                                  f'explainer_{mod}_{split_state}_{rand_state}_shap.pkl')

                if os.path.exists(shap_file_path) and not updateDNN:
                    with open(shap_file_path, 'rb') as file:
                        shap_values_all[key] = pickle.load(file)
                else:
                    model = keras.models.load_model(model_path)
                    explainer = shap.DeepExplainer(model, dnn_descs)
                    shap_val = explainer.shap_values(dnn_descs)[0]
                    base_value = explainer.expected_value
                    base_values = np.full((shap_val.shape[0],),
                                          base_value if isinstance(base_value, float) else base_value[0])

                    shap_dnn = shap.Explanation(values=shap_val,
                                                base_values=base_values,
                                                data=dnn_descs,
                                                feature_names=dnn_featnames,
                                                )

                    shap_values_all[key] = shap_dnn
                    with open(shap_file_path, 'wb') as file:
                        pickle.dump(shap_dnn, file)

                print(f'Total models processed so far: {len(shap_values_all)}')

        else:
            model_files = [f for f in os.listdir(model_dir) if f.startswith(f'model_reg_{mod}')
                           and f.endswith('.dat')]
            for filename in model_files:
                model_path = f'{model_dir}/{filename}'
                key = f'{mod}_{filename.split("_")[-2].split(".")[0]}'
                split_state = f'{filename.split("_")[-2].split(".")[0]}'
                rand_state = f'{filename.split("_")[-1].split(".")[0]}'
                built_explainer = os.path.join(model_dir, f'explainer_{mod}_{split_state}_{rand_state}_shap.pkl')
                if os.path.exists(built_explainer):
                    shap_file_path = os.path.join(shap_explainers_dir,
                                                  f'explainer_{mod}_{split_state}_{rand_state}_shap.pkl')
                    shutil.copy(built_explainer, shap_file_path)
                else:
                    shap_file_path = os.path.join(shap_explainers_dir,
                                                  f'explainer_{mod}_{split_state}_{rand_state}_shap.pkl')

                if os.path.exists(shap_file_path) and not updateML:
                    with open(shap_file_path, 'rb') as file:
                        shap_values_all[key] = pickle.load(file)
                else:
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                    explainer = shap.TreeExplainer(model, ml_descs)
                    shap_val = explainer(ml_descs)
                    base_value = explainer.expected_value
                    base_values = np.full((shap_val.shape[0],),
                                          base_value if isinstance(base_value, float) else base_value[0])

                    shap_ml = shap.Explanation(values=shap_val,
                                               base_values=base_values,
                                               data=ml_descs,
                                               feature_names=ml_featnames,
                                               )

                    shap_values_all[key] = shap_ml
                    with open(shap_file_path, 'wb') as file:
                        pickle.dump(shap_ml, file)

                print(f'Total models processed so far: {len(shap_values_all)}')
    return shap_values_all


def mean_shap_interpretations(shap_dict, combined_featnames, combined_descs):
    num_entries = len(shap_dict)

    first_dnn_key = next((key for key in shap_dict if key.startswith('dnn_')), None)

    if first_dnn_key is not None:
        first_key = first_dnn_key
    else:
        first_key = next(iter(shap_dict))

    num_features = len(combined_featnames)
    sample_size = shap_dict[first_key].values.shape[0]
    total_shap_values = np.zeros((sample_size, num_features))
    total_base_values = np.zeros_like(shap_dict[first_key].base_values)

    feat_index = {feat: idx for idx, feat in enumerate(combined_featnames)}

    for shap_explanation in shap_dict.values():
        for i, feat in enumerate(shap_explanation.feature_names):
            if feat in feat_index:
                idx = feat_index[feat]
                total_shap_values[:, idx] += shap_explanation.values[:, i]
        total_base_values += shap_explanation.base_values

    mean_shap_values = total_shap_values / num_entries
    mean_base_values = total_base_values / num_entries

    mean_explainer = shap.Explanation(
        values=mean_shap_values,
        base_values=mean_base_values,
        data=combined_descs,
        feature_names=combined_featnames,
    )

    return mean_explainer


def process_all(labels, mods, base_path, tp, updateDNN=False, updateML=False):
    all_shap_values = {}
    all_mean_explainers = {}

    for label in labels:
        path = os.path.join(base_path, label)
        combined_descs, combined_featnames, dnn_descs, \
            dnn_featnames, ml_descs, ml_featnames = get_combined_descs_and_featnames(mods, tp, path)

        shap_values_all = get_all_shap(label, mods, tp, path, dnn_descs, dnn_featnames,
                                       ml_descs, ml_featnames, updateDNN, updateML)
        mean_explainer = mean_shap_interpretations(shap_values_all, combined_featnames, combined_descs)

        all_shap_values[label] = shap_values_all
        all_mean_explainers[label] = mean_explainer

    return all_shap_values, all_mean_explainers


def split_explainer(original_explainer, label):
    explainer = original_explainer[label]
    feature_names = explainer.feature_names

    ph2d_indices = [i for i, name in enumerate(feature_names) if name.startswith('Ph2D_')]
    non_ph2d_indices = [i for i in range(len(feature_names)) if i not in ph2d_indices]

    def sub_explainers(explainer, indices, feature_names):
        new_explainer = shap.Explanation(
            values=explainer.values[:, indices],
            base_values=explainer.base_values,
            data=explainer.data[:, indices],
            feature_names=[feature_names[i] for i in indices]
        )
        return new_explainer

    ph2d_explainer = sub_explainers(explainer, ph2d_indices, feature_names)
    non_ph2d_explainer = sub_explainers(explainer, non_ph2d_indices, feature_names)

    original_explainer[f'{label}_ph2d'] = ph2d_explainer
    original_explainer[f'{label}_non_ph2d'] = non_ph2d_explainer

    return original_explainer