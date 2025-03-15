import sys
script_dir = "../scripts"
sys.path.insert(0, script_dir)
from filters_dop import *

print('Starting')
targets = ["D3"]
chembls = ["31"]
base_name = 'pubdata'
standard_types = ["Ki"]
assaydefinitions = ["antagonist"]
for chembl in chembls:
    df = pd.DataFrame()
    for target in targets:
        chembl_tsv_file = f"pgsql/all_pgsql/chembl{chembl}_{target}.tsv"
        for assaydefinition in assaydefinitions:
            for standard_type in standard_types:
                output_dir = f"new_datasets/C{chembl}/dataset_{target}_{assaydefinition}_{standard_type}"
                output = filters_d2d3(chembl_tsv_file, standard_type, target, assaydefinition, output_dir, base_name)
                ## if you want to save this to excel, we can make a df
                # df = get_dataframe(output, df, target, standard_type, assaydefinition)
                print(f"Dataset created: chembl = {chembl}, target = {target}, assaydefinition = {assaydefinition},"
                      f" standard_type = {standard_type}")
                print(output)
    # save_to_excel(df, chembl, xlsx_dir)

print('Finished')