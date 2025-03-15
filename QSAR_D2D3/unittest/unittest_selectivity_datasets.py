import unittest
import sys
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
sys.path.insert(0, core_dir)
selectivity_dir = core_dir + "/selectivity"
sys.path.insert(0, selectivity_dir)

from selectivity import *

chembl_version = "C31"
n = 200
test_data = home+"/repositories/ai-x/unittest/unittest_data/test_selectivity_data/"
path1 = test_data+"C31_D2D3/dataset_D2_antagonist_Ki/pubdata"
path2 = test_data+"C31_D2D3/dataset_D3_antagonist_Ki/pubdata"

class TestSelectivity(unittest.TestCase):
    def setUp(self):
        self.i = 0

    def tearDown(self):
        pass

    def startUp(self):
        d2 = get_df(path1)
        d3 = get_df(path2)
        d2 = check_similarity_within_df(d2)  # checks if there are any similar compounds within its own df
        d3 = check_similarity_within_df(d3)  # checks if there are any similar compounds within its own df
        mol_dup_pairs = get_mol_dup_pairs(d2, d3)  # a paired set, referring to similar indexes between d2 and d3
        df_overlap = get_overlap(d2, d3, mol_dup_pairs)  # combines d2 and d3 (horizontally), only returns similar compounds

        return d2, d3, mol_dup_pairs, df_overlap

    def startUp2(self):
        d2, d3, mol_dup_pairs, df_overlap = self.startUp()
        np.random.seed(self.i)
        # calculated value
        ls_d2_indexes, ls_d3_indexes, ls_overlap_indexes = get_similar_indexes(mol_dup_pairs, n)
        d2_train, d2_val = training_validation_datasplit(d2, ls_d2_indexes)
        d3_train, d3_val = training_validation_datasplit(d3, ls_d3_indexes)

        return d2_train, d2_val, d3_train, d3_val, ls_overlap_indexes

    def test_01_startingSize(self):
        d2, d3, mol_dup_pairs, df_overlap = self.startUp()
        self.assertEqual(len(d2), 3507)
        self.assertEqual(len(d3), 2305)
        self.assertEqual(len(df_overlap), 1758)

    def test_02_d2_values(self):
        d2_train, d2_val, d3_train, d3_val, ls_overlap_indexes= self.startUp2()
        path = test_data+'dataset_D2_antagonist_Ki/'
        df_train = pd.read_csv(path+"pubdata"+str(self.i)+".act", sep='\t', header=None)
        df_val = pd.read_csv(path+"val"+str(self.i)+".act", sep='\t', header=None)

        # unittest
        self.assertEqual(list(d2_train['CHEMBL']), list(df_train[0]))
        self.assertEqual(list(d2_train['pKi']), list(df_train[1]))
        self.assertEqual(list(d2_val['CHEMBL']), list(df_val[0]))
        self.assertEqual(list(d2_val['pKi']), list(df_val[1]))

        # unittest
        self.assertEqual(len(d2_train), 3307)
        self.assertEqual(len(d2_val), 200)

    def test_03_d3_values(self):
        d2_train, d2_val, d3_train, d3_val, ls_overlap_indexes= self.startUp2()
        path = test_data+'dataset_D3_antagonist_Ki/'
        df_train = pd.read_csv(path+"pubdata"+str(self.i)+".act", sep='\t', header=None)
        df_val = pd.read_csv(path+"val"+str(self.i)+".act", sep='\t', header=None)

        # unittest
        self.assertEqual(list(d3_train['CHEMBL']), list(df_train[0]))
        self.assertEqual(list(d3_train['pKi']), list(df_train[1]))
        self.assertEqual(list(d3_val['CHEMBL']), list(df_val[0]))
        self.assertEqual(list(d3_val['pKi']), list(df_val[1]))

        # unittest
        self.assertEqual(len(d3_train), 2105)
        self.assertEqual(len(d3_val), 200)


    def test_04_d2_overlap_values(self):
        d2, d3, mol_dup_pairs, df_overlap = self.startUp()
        d2_train, d2_valid, d3_train, d3_valid, ls_overlap_indexes = self.startUp2()
        df_training, df_validation = training_validation_datasplit(df_overlap, ls_overlap_indexes)
        df_train_overlap1, df_val_overlap1, df_train_overlap2, df_val_overlap2 = split_overlapped_df(df_training,
                                                                                                     df_validation)

        path = test_data + 'dataset_D2_overlap_antagonist_Ki/'
        data_train = pd.read_csv(path + "pubdata" + str(self.i) + ".act", sep='\t', header=None)
        data_val = pd.read_csv(path + "val" + str(self.i) + ".act", sep='\t', header=None)

        # unittest - target 1
        self.assertEqual(list(df_train_overlap1['CHEMBL']), list(data_train[0]))
        self.assertEqual(list(df_train_overlap1['pKi']), list(data_train[1]))
        self.assertEqual(list(df_val_overlap1['CHEMBL']), list(data_val[0]))
        self.assertEqual(list(df_val_overlap1['pKi']), list(data_val[1]))
        # unittest
        self.assertEqual(len(df_train_overlap1), 1558)
        self.assertEqual(len(df_val_overlap1), 200)

    def test_05_d3_overlap_values(self):
        d2, d3, mol_dup_pairs, df_overlap = self.startUp()
        d2_train, d2_valid, d3_train, d3_valid, ls_overlap_indexes = self.startUp2()
        df_training, df_validation = training_validation_datasplit(df_overlap, ls_overlap_indexes)
        df_train_overlap1, df_val_overlap1, df_train_overlap2, df_val_overlap2 = split_overlapped_df(df_training,
                                                                                                     df_validation)

        path = test_data + 'dataset_D3_overlap_antagonist_Ki/'
        data_train = pd.read_csv(path + "pubdata" + str(self.i) + ".act", sep='\t', header=None)
        data_val = pd.read_csv(path + "val" + str(self.i) + ".act", sep='\t', header=None)

        # unittest - target 1
        self.assertEqual(list(df_train_overlap2['CHEMBL']), list(data_train[0]))
        self.assertEqual(list(df_train_overlap2['pKi']), list(data_train[1]))
        self.assertEqual(list(df_val_overlap2['CHEMBL']), list(data_val[0]))
        self.assertEqual(list(df_val_overlap2['pKi']), list(data_val[1]))
        # unittest
        self.assertEqual(len(df_train_overlap2), 1558)
        self.assertEqual(len(df_val_overlap2), 200)

    def test_06_d2d3_diff_regression_data(self):
        # df_training, df_validation = training_validation_datasplit(df_overlap, ls_overlap_indexes)
        path = test_data + 'dataset__ratio_D2_antagonist_Ki_D3_antagonist_Ki/'
        data_train = pd.read_csv(path + "pubdata" + str(self.i) + ".act", sep='\t', header=None)
        data_val = pd.read_csv(path + "val" + str(self.i) + ".act", sep='\t', header=None)

        d2, d3, mol_dup_pairs, df_overlap = self.startUp()
        d2_train, d2_valid, d3_train, d3_valid, ls_overlap_indexes = self.startUp2()
        df_training, df_validation = training_validation_datasplit(df_overlap, ls_overlap_indexes)

        # unittest
        self.assertEqual(list(df_training['ratio']), list(data_train[1]))
        self.assertEqual(list(df_validation['ratio']), list(data_val[1]))
        self.assertEqual(len(df_training), 1558)
        self.assertEqual(len(df_validation), 200)

    def test_07_d2d3_diff_classification_data(self):
        path = test_data + 'dataset__ratio_D2_antagonist_Ki_D3_antagonist_Ki/'
        data_train = pd.read_csv(path + "pubdata_class" + str(self.i) + ".act", sep='\t', header=None)
        data_val = pd.read_csv(path + "val_class" + str(self.i) + ".act", sep='\t', header=None)

        d2, d3, mol_dup_pairs, df_overlap = self.startUp()
        d2_train, d2_valid, d3_train, d3_valid, ls_overlap_indexes = self.startUp2()
        df_training, df_validation = training_validation_datasplit(df_overlap, ls_overlap_indexes)
        df_training_class = classify(df_training)
        df_validation_class = classify(df_validation)

        lis = list(df_training_class['ratio_class'])
        ls_df_training_class = [eval(i) for i in lis]

        lis = list(df_validation_class['ratio_class'])
        ls_df_validation_class = [eval(i) for i in lis]

        # unittest
        self.assertEqual(ls_df_training_class, list(data_train[1]))
        self.assertEqual(ls_df_validation_class, list(data_val[1]))
        self.assertEqual(len(df_training_class), 1311)
        self.assertEqual(len(df_validation_class), 171)


if __name__ == '__main__':
    unittest.defaultTestLoader.sortTestMethodsUsing = None
    unittest.main(testLoader=unittest.defaultTestLoader)
