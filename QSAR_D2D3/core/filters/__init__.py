# __init__.py

import pandas as pd

# Import functions from individual modules
from .read_data import read_data
from .filter_confidence import filter_confidence
from .filter_assay_type import filter_assay_type
from .filter_affinity import filter_affinity
from .filter_units import filter_units
from .filter_exact import filter_exact
from .filter_assaydefinition import filter_assaydefinition
from .filter_secondary_test_set import filter_secondary_test_set
from .filter_bao_format import filter_bao_format
from .filter_selected import filter_selected
from .filter_small_sets import filter_small_sets
from .filter_salts import filter_salts
from .filter_elements import filter_elements
from .filter_size import filter_size
from .filter_pchembl_values import filter_pchembl_values
from .filter_weirdos import filter_weirdos
from .pairwise_comparison import fingerprints
from .pairwise_comparison import pairwise_comparison
from .deduplicate_mols import deduplicate_mols
from .deduplicate_mols_by_chemblid import deduplicate_mols_by_chemblid
from .deduplicate_mols_by_similarity import deduplicate_mols_by_similarity
from .calc_pscale import calc_pscale
from .add_doc_cmpd_count import add_doc_cmpd_count
from .write_smi_act_reg import write_smi_act_reg
from .write_smi_act_class import write_smi_act_class
from .data_to_pickle import data_to_pickle

# the following modules are added by Precious
from .filter_negative_pchembl_value import filter_negative_pchembl_value
from .filter_multiple_entries import filter_multiple_entries
from .update_calc_pchembl_values import update_calc_pchembl_values
from .filter_low_pchembl_values import filter_low_pchembl_values


# Message to indicate successful import of the package
print("filters package has been imported!")
