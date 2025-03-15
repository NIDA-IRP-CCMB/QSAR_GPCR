import pandas as pd

def read_data(file_name, Verbose=False):
    '''
    labels = {0:'pref_name', 1:'organism', 2:'assay_id', 3:'assay_type',
                   4:'relationship_type', 5:'relationship_desc', 6:'confidence_score',
                   7'curated_by', 8:'description', 9:'activity_id', 10:'relation',
                   11:'value', 12:'units', 13:'type', 14:'standard_relation', 15:'standard_value',
                   16:'standard_units', 17:'standard_flag', 18:'standard_type', 19:'pchembl_value',
                   20:'activity_comment', 21:'data_validity_comment', 22:'potential_duplicate',
                   23:'text_value', 24:'standard_text_value', 25:'molregno', 26:'chembl_id',
                   27:'canonical_smiles', 28:'pref_name', 29:'parent_molregno', 30:'active_molregno',
                   31:'doc_id', 32:'pubmed_id', 33:'doi', 34:'journal', 35:'year', 36:'volume',
                   37:'first_page', 38:'src_short_name', 39:'bao_format'}
    '''
    labels = ['pref_name_target', 'organism', 'assay_id', 'assay_type', \
               'relationship_type', 'relationship_desc', 'confidence_score', \
               'curated_by', 'description', 'activity_id', 'relation', \
               'value', 'units', 'type', 'standard_relation', 'standard_value', \
               'standard_units', 'standard_flag', 'standard_type', 'pchembl_value', \
               'activity_comment', 'data_validity_comment', 'potential_duplicate', \
               'text_value', 'standard_text_value', 'molregno', 'chembl_id', \
               'canonical_smiles', 'pref_name', 'parent_molregno', 'active_molregno', \
               'doc_id', 'pubmed_id', 'doi', 'journal', 'year', 'volume', 'first_page', \
               'src_short_name', 'bao_format']
    raw_data = pd.read_csv(file_name, names=labels, header=None, sep="\t")

    if Verbose:
        print(f'Number of pharmacological activity at starting: ', len(raw_data))

    return raw_data