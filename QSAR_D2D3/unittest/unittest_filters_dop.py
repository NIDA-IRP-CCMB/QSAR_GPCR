import unittest

## define enviroment
import sys, os
from pathlib import Path
home = str(Path.home())
core_dir = home+'/repositories/ai-x/core'
conf_dir = core_dir+"/keywords"
sys.path.insert(0, core_dir)
sys.path.insert(0, conf_dir)

from filters import *
from filters_dop import *

# Initialize global variables
buffer = None
unittest_data_dir = 'unittest_data/data4filters'
chembl_tsv_file = unittest_data_dir+'/chembl31_D2.tsv'
chembl_tsv_file2 = unittest_data_dir+'/chembl31_D3.tsv'

# 20 numbers, from starting to after deduplication pass
# D2 antagonist Ki
ref = [29576, 17973, 13369, 8935, 8879, 8051, 6356, 6286, 5451, 5346, 5213, 5210, 5200, 5052, 5052, 5052, 3507, 2920, 2825, 95]
# D3 antagonist Ki
ref2 = [12404, 8061, 6599, 4875, 4865, 4593, 3445, 3445, 3445, 3323, 3192, 3192, 3188, 3138, 3138, 3138, 2304, 2093, 2054, 39]


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.target = "D2"
        self.assaydefinition = "antagonist"

    def test_01_read_data(self):
        global buffer
        buffer = read_data(chembl_tsv_file)
        self.assertEqual(len(buffer), ref[0])

    def test_02_confidence_filter(self):
        global buffer
        buffer = filter_confidence(buffer)
        self.assertEqual(len(buffer), ref[1])

    def test_03_assay_filter(self):
        global buffer
        buffer = filter_assay_type(buffer, self.target, self.assaydefinition)
        self.assertEqual(len(buffer), ref[2])

    def test_04_affinity_filter(self):
        global buffer
        buffer = filter_affinity(buffer, keepIC50=False, keepKi=True)
        self.assertEqual(len(buffer), ref[3])

    def test_05_units_filter(self):
        global buffer
        buffer = filter_units(buffer)
        self.assertEqual(len(buffer), ref[4])

    def test_06_exact_filter(self):
        global buffer
        buffer = filter_exact(buffer)
        self.assertEqual(len(buffer), ref[5])

    def test_07_assay_definition_filter(self):
        global buffer
        buffer, unused = filter_assaydefinition(buffer, self.target, self.assaydefinition)
        self.assertEqual(len(buffer), ref[6])

    def test_08_year_filter(self):
        global buffer
        buffer = filter_year(buffer, self.target, year=1990)
        self.assertEqual(len(buffer), ref[7])

    def test_09_bao_format_filter(self):
        global buffer
        buffer = filter_bao_format(buffer, self.target , self.assaydefinition)
        self.assertEqual(len(buffer), ref[8])

    def test_10_selected_filter(self):
        global buffer
        buffer = filter_selected(buffer, self.target, self.assaydefinition)
        self.assertEqual(len(buffer), ref[9])

    def test_11_small_set_filter(self):
        global buffer
        buffer = filter_small_sets(buffer, threshold=4)
        self.assertEqual(len(buffer), ref[10])

    def test_12_salts_filter(self):
        global buffer
        buffer = filter_salts(buffer, conf_dir)
        self.assertEqual(len(buffer), ref[11])

    def test_13_elements_filter(self):
        global buffer
        buffer = filter_elements(buffer)
        self.assertEqual(len(buffer), ref[12])

    def test_14_size_filter(self):
        global buffer
        buffer = filter_size(buffer)
        self.assertEqual(len(buffer), ref[13])

    def test_15_pchembl_filter(self):
        global buffer
        buffer = filter_pchembl_values(buffer, replace=True)
        self.assertEqual(len(buffer), ref[14])

    def test_16_weirdos_filter(self):
        global buffer
        buffer = filter_weirdos(buffer)
        self.assertEqual(len(buffer), ref[15])

    def test_17_deduplicate_filter(self):
        global buffer
        buffer = deduplicate_mols(buffer)
        self.assertEqual(len(buffer), ref[16])


class TestFilters2(unittest.TestCase):
    def setUp(self):
        self.target = "D3"
        self.assaydefinition = "antagonist"

    def test_01_read_data(self):
        global buffer
        buffer = read_data(chembl_tsv_file2)
        self.assertEqual(len(buffer), ref2[0])

    def test_02_confidence_filter(self):
        global buffer
        buffer = filter_confidence(buffer)
        self.assertEqual(len(buffer), ref2[1])

    def test_03_assay_filter(self):
        global buffer
        buffer = filter_assay_type(buffer, self.target, self.assaydefinition)
        self.assertEqual(len(buffer), ref2[2])

    def test_04_affinity_filter(self):
        global buffer
        buffer = filter_affinity(buffer, keepIC50=False, keepKi=True)
        self.assertEqual(len(buffer), ref2[3])

    def test_05_units_filter(self):
        global buffer
        buffer = filter_units(buffer)
        self.assertEqual(len(buffer), ref2[4])

    def test_06_exact_filter(self):
        global buffer
        buffer = filter_exact(buffer)
        self.assertEqual(len(buffer), ref2[5])

    def test_07_assay_definition_filter(self):
        global buffer
        buffer, unused = filter_assaydefinition(buffer, self.target, self.assaydefinition)
        self.assertEqual(len(buffer), ref2[6])

    def test_08_year_filter(self):
        global buffer
        buffer = filter_year(buffer, self.target, year=1990)
        self.assertEqual(len(buffer), ref2[7])

    def test_09_bao_format_filter(self):
        global buffer
        buffer = filter_bao_format(buffer, self.target, self.assaydefinition)
        self.assertEqual(len(buffer), ref2[8])

    def test_10_selected_filter(self):
        global buffer
        buffer = filter_selected(buffer, self.target, self.assaydefinition)
        self.assertEqual(len(buffer), ref2[9])

    def test_11_small_set_filter(self):
        global buffer
        buffer = filter_small_sets(buffer, threshold=4)
        self.assertEqual(len(buffer), ref2[10])

    def test_12_salts_filter(self):
        global buffer
        buffer = filter_salts(buffer, conf_dir)
        self.assertEqual(len(buffer), ref2[11])

    def test_13_elements_filter(self):
        global buffer
        buffer = filter_elements(buffer)
        self.assertEqual(len(buffer), ref2[12])

    def test_14_size_filter(self):
        global buffer
        buffer = filter_size(buffer)
        self.assertEqual(len(buffer), ref2[13])

    def test_15_pchembl_filter(self):
        global buffer
        buffer = filter_pchembl_values(buffer, replace=True)
        self.assertEqual(len(buffer), ref2[14])

    def test_16_weirdos_filter(self):
        global buffer
        buffer = filter_weirdos(buffer)
        self.assertEqual(len(buffer), ref2[15])

    def test_17_deduplicate_filter(self):
        global buffer
        buffer = deduplicate_mols(buffer)
        self.assertEqual(len(buffer), ref2[16])


if __name__ == '__main__':
    unittest.defaultTestLoader.sortTestMethodsUsing = None
    unittest.main(testLoader=unittest.defaultTestLoader)
