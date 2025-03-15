## Filters package


The filters.py file is now modularized into the filters python package. As a consequence, there are changes on how 
it is used, e.g. passing of conf_dir as argument to individual modules that require it. This package has also been
generalized for the current protein targets. The option to save individual filter results (dfs before and after applying
the filter) was also added.


1. Contents

         filters
         ├── __init__.py
         ├── add_doc_cmpd_count.py
         ├── calc_pscale.py
         ├── data_to_pickle.py
         ├── deduplicate_mols_by_chemblid.py
         ├── deduplicate_mols_by_similarity.py
         ├── deduplicate_mols.py
         ├── filter_affinity.py
         ├── filter_assaydefinition.py
         ├── filter_assay_type.py
         ├── filter_bao_format.py
         ├── filter_confidence.py
         ├── filter_elements.py
         ├── filter_exact.py
         ├── filter_pchembl_values.py
         ├── filter_salts.py
         ├── filter_secondary_test_set.py
         ├── filter_selected.py
         ├── filter_size.py
         ├── filter_small_sets.py
         ├── filter_units.py
         ├── filter_weirdos.py
         ├── pairwise_comparison.py
         ├── read_data.py
         ├── write_smi_act_reg.py
         ├── write_smi_act_class.py
         └── README.md

2. Usage: The package can be loaded as the full package using
      
         from filters import *

   or as individual modules, e.g.

         from filters import pairwise_comparison

   The run_filters.py in ai-x/core has already been configured for the filters package and has passed unittest.


3. Precious updates these filters

         filters
         ├── filter_low_pchembl_values.py
         ├── filter_multiple_entries.py
         ├── filter_negative_pchembl_value.py
         └── update_calc_pchembl_values.py


