import pandas as pd

def add_doc_cmpd_count(in_lines, Verbose=False):
    # add field to end of data to represent how many compounds
    # are in a given publication (this makes field 39)
    #
    # This should be the first filter run, since it adds a field that other filters might depend upon
    # we don't use this function in run_filters.py. This is from Andy Fant version.

    count = []

    # reset index before iterating/looping
    in_lines = in_lines.reset_index()
    mol_per_doc_counts = {}

    for i in range(len(in_lines)):
        if in_lines['doc_id'][i] not in mol_per_doc_counts.keys():
            mol_per_doc_counts[in_lines['doc_id'][i]] = 1
        else:
            mol_per_doc_counts[in_lines['doc_id'][i]] = mol_per_doc_counts[in_lines['doc_id'][i]] + 1

    for i in range(len(in_lines)):
        count.append(mol_per_doc_counts[in_lines['doc_id'][i]])

    in_lines['doc_cmpd_count'] = count

    if Verbose:
        print(f'Number of pharmacological activity at adding mols_per_doc count: ', len(in_lines))

    return in_lines.reset_index(drop=True)