"""
    SHAP feature importance. Tested for both xgb and rf. This generates two plots of features ordered in terms
    of their effect to the prediction value.
    1. A bar plot showing the absolute SHAP values of features. It does not show whether it affects the
       prediction in a positive or negative way.
    2. A violin plot. The horizontal axis represents the SHAP value, while the color of the point shows
       us if that observation has a higher or a lower value, when compared to other observations.
    These plots only show the top 30 of the total number of features. The full list for both raw and
    normalized SHAP values are stored in their corresponding .dat files

    If you already have a data set, feature names, and a model, you can run this on its own using
    python ${COREPATH}/feature_importance.py -x {rf, xgb, or dnn} -mp {/path/to/model} -td {/path/to/training_data/file} \
                                             -fn {/path/to/feature_names/file} -od {ouput directory} -r {Rand state} -s {Split of Model}
    e.g.
    python ${COREPATH}/feature_importance.py -x rf -mp ../model_reg_rf_0.15_87_864.dat -td train_descs_864.dat \
                                             -fn feature_names_864.dat -od . -r 87 -s 864
    The production of train_descs_{Rand}.dat and feature_names_{Rand}.dat is toggled in run_buildmodel.py
"""

class FeatureImportance:
    def __init__(self, method, model, training_descs, features, output_dir, state, split, output_ext=''):
        self.method = method
        self.model = model
        self.training_descs = training_descs
        self.features = features
        self.output_ext = output_ext
        self.output_dir = output_dir
        self.state = state
        self.split = split


    def run_analysis(self):
        # Import needed libraries
        import shap
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.special import softmax
        import pandas as pd
        import pickle

        print('Training Data Shape:', self.training_descs.shape, 'Number of features:', len(self.features))
        if self.method == 'xgb' or self.method == 'rf':
            explainer = shap.TreeExplainer(self.model, self.training_descs)
            # shap_values = explainer(self.training_descs)
            # shap_values_values = shap_values.values
            # shap_base_values = shap_values.base_values

            shap_values = explainer.shap_values(self.training_descs, check_additivity=False)
            base_value = explainer.expected_value
            base_values = np.full((shap_values.shape[0],),
                                  base_value if isinstance(base_value, float) else base_value[0])
            shap_file_path = self.output_dir + f'/explainer_{self.method}_{self.state}_{self.split}_shap.pkl'
            shap_ml = shap.Explanation(values=shap_values,
                                       base_values=base_values,
                                       data=self.training_descs,
                                       feature_names=self.features,
                                       )
            with open(shap_file_path, 'wb') as file:
                pickle.dump(shap_ml, file)

            plt.figure()
            # Bar plot: Include only a third of the features for plotting
            shap.summary_plot(shap_values, feature_names=self.features, max_display=30, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir + f"/shap_train_box_{self.split}.png", bbox_inches='tight', dpi=300)
            if self.output_ext:
                plt.savefig(self.output_dir + f"/shap_train_box_{self.output_ext}.png", bbox_inches='tight', dpi=300)
            plt.clf()
            plt.figure()
            # Violin plot: Include only a third of the features for plotting
            shap.summary_plot(shap_values, feature_names=self.features, max_display=30, plot_type="violin", show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir + f"/shap_train_violin_{self.split}.png", bbox_inches='tight', dpi=300)
            if self.output_ext:
                plt.savefig(self.output_dir + f"/shap_train_violin_{self.output_ext}.png", bbox_inches='tight', dpi=300)
            plt.clf()

            pickle.dump(shap_values, open(self.output_dir + f"/shap_values_{self.state}_{self.split}.dat", "wb"))
            if self.output_ext:
                pickle.dump(shap_values, open(self.output_dir + f"/shap_values_{self.output_ext}.dat", "wb"))


            # Get full list of SHAP values and store the raw and normalized values for each feature
            importances = []
            for i in range(shap_values.shape[1]):
                importances.append(np.mean(np.abs(shap_values[:, i])))
            # Calculates the normalized version
            importances_norm = softmax(importances)
            # Organize the importances and columns in a dictionary
            feature_importances = {fea: imp for imp, fea in zip(importances, self.features)}
            feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, self.features)}
            # Sorts the dictionary
            feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
            feature_importances_norm = {k: v for k, v in
                                        sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
            # Save the feature importances to a pickled .dat file
            pickle.dump(feature_importances, open(self.output_dir + f"/feature_importances_{self.state}_{self.split}.dat", "wb"))
            if self.output_ext:
                pickle.dump(feature_importances,
                            open(self.output_dir + f"/feature_importances_{self.output_ext}.dat", "wb"))

            # Save the normalized feature importances to a pickled .dat file
            pickle.dump(feature_importances_norm,
                        open(self.output_dir + f"/feature_importances_norm_{self.state}_{self.split}.dat", "wb"))
            if self.output_ext:
                pickle.dump(feature_importances_norm,
                            open(self.output_dir + f"/feature_importances_norm_{self.output_ext}.dat", "wb"))
        if self.method == 'dnn':
            explainer = shap.DeepExplainer(self.model, self.training_descs)
            shap_values = explainer.shap_values(self.training_descs)[0]
            base_value = explainer.expected_value
            base_values = np.full((shap_values.shape[0],),
                                  base_value if isinstance(base_value, float) else base_value[0])
            shap_file_path = self.output_dir + f'/explainer_{self.method}_{self.state}_{self.split}_shap.pkl'
            shap_dnn = shap.Explanation(values=shap_values,
                                       base_values=base_values,
                                       data=self.training_descs,
                                       feature_names=self.features,
                                       )
            with open(shap_file_path, 'wb') as file:
                pickle.dump(shap_dnn, file)

            #shap_values = explainer(self.training_descs[:100])
            plt.figure()
            # Bar plot: Include only a third of the features for plotting
            shap.summary_plot(shap_values, feature_names=self.features, max_display=30, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir + f"/shap_train_box_{self.state}_{self.split}.png", bbox_inches='tight', dpi=300)
            if self.output_ext:
                plt.savefig(self.output_dir + f"/shap_train_box_{self.output_ext}.png", bbox_inches='tight',
                            dpi=300)
            plt.clf()
            plt.figure()

            pickle.dump(shap_values,
                        open(self.output_dir + f"/shap_values_{self.state}_{self.split}.dat", "wb"))
            if self.output_ext:
                pickle.dump(shap_values,
                            open(self.output_dir + f"/shap_values_{self.output_ext}.dat", "wb"))

            # Get full list of SHAP values and store the raw and normalized values for each feature
            importances = []
            for i in range(shap_values.shape[1]):
                importances.append(np.mean(np.abs(shap_values[:, i])))
            # Calculates the normalized version
            importances_norm = softmax(importances)
            # Organize the importances and columns in a dictionary
            feature_importances = {fea: imp for imp, fea in zip(importances, self.features)}
            feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, self.features)}
            # Sorts the dictionary
            feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)}
            feature_importances_norm = {k: v for k, v in
                                        sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse=True)}
            # Save the feature importances to a pickled .dat file
            pickle.dump(feature_importances, open(self.output_dir + f"/feature_importances_{self.state}_{self.split}.dat", "wb"))
            if self.output_ext:
                pickle.dump(feature_importances,
                            open(self.output_dir + f"/feature_importances_{self.output_ext}.dat", "wb"))

            # Save the normalized feature importances to a pickled .dat file
            pickle.dump(feature_importances_norm, open(self.output_dir + f"/feature_importances_norm_{self.state}_{self.split}.dat", "wb"))
            if self.output_ext:
                pickle.dump(feature_importances_norm,
                            open(self.output_dir + f"/feature_importances_norm_{self.output_ext}.dat", "wb"))



if __name__ == "__main__":

    import argparse
    import os
    import pickle

    def parse_args():
        parser = argparse.ArgumentParser(description="SHAP Feature Importance Analysis")

        # Required arguments for running as standalone
        parser.add_argument("-x", "--method", required=True, help="Method: rf, xgb, or dnn")
        parser.add_argument("-mp", "--model_path", required=True, help="Path to the machine learning model file")
        parser.add_argument("-td","--training_descs", required=True, help="Path to the pickled training data set")
        parser.add_argument("-fn","--feature_names", required=True, help="Path to the pickled features names file")
        parser.add_argument("-od","--output_dir", required=False, help="Output directory for plots and data files")
        parser.add_argument("-r","--rand", type=int, required=True, help="Random state of the built model, e.g. 87")
        parser.add_argument("-s","--split", type=int, required=True, help="Model identifier")

        return parser.parse_args()

    # Parse command-line arguments and check flags
    args = parse_args()

    if args.method not in ('rf', 'xgb', 'dnn'):
        print("Error: Method must be 'rf', 'xgb', or 'dnn'.")
        exit(1)

    if not os.path.isfile(args.model_path):
        print(f"Error: Model file '{args.model_path}' not found.")
        exit(1)

    if not os.path.isfile(args.training_descs):
        print(f"Error: Training data file '{args.training_descs}' not found.")
        exit(1)

    if not os.path.isfile(args.feature_names):
        print(f"Error: Feature names file '{args.feature_names}' not found.")
        exit(1)

    if args.output_dir is None:
        args.output_dir = os.getcwd()
        print(f"Output directory not specified. Saving results in the current working directory: {args.output_dir}")

    if not isinstance(args.rand, int):
        print("Error: Rand state must be an integer.")
        exit(1)

    if not isinstance(args.split, int):
        print("Error: Split must be an integer.")
        exit(1)

    # Load pickled training set, feature names, and model
    training_descs = pickle.load(open(args.training_descs, "rb"))
    feature_names = pickle.load(open(args.feature_names, "rb"))
    model = pickle.load(open(args.model_path, "rb"))

    # Create an instance of the FeatureImportance class using the parsed arguments
    feature_imp = FeatureImportance(args.method, model, training_descs, feature_names,
                                    args.output_dir, args.rand, args.split)

    # Run the analysis
    feature_imp.run_analysis()