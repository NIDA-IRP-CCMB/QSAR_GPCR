import pandas as pd
def filter_multiple_entries(in_lines, Verbose=False, stats=False):

    '''
    Purpose:
    :param in_lines: dataset being used (df))
    :param stats: contains percentage of multiple entries
    :return:
    '''
    id_column_name = 'chembl_id'
    doi_column_name = 'doi'
    processed_ids = []
    multiple_entries = 0
    unique_entries = 0
    multiple_entries_df = pd.DataFrame()
    unique_entries_df = pd.DataFrame()

    # Track the indices of processed rows
    processed_indices = []

    for i in range(len(in_lines)):
        name = in_lines[id_column_name][i]
        doi = in_lines[doi_column_name][i]
        name_doi = f"{name}_{doi}"
        if name_doi not in processed_ids:
            # Add the name_doi to the list of processed ID
            processed_ids.append(name_doi)

            if pd.isna(doi):
                _sel_rows = in_lines[(in_lines[id_column_name] == name) & (pd.isna(in_lines[doi_column_name]))]

            elif pd.isna(name):
                _sel_rows = in_lines[pd.isna(in_lines[id_column_name] == name) & (in_lines[doi_column_name])]

            elif pd.isna(name) and pd.isna(doi):
                _sel_rows = in_lines[pd.isna(in_lines[id_column_name] == name) & (pd.isna(in_lines[doi_column_name]))]

            else:
                _sel_rows = in_lines[(in_lines[id_column_name] == name) & (in_lines[doi_column_name] == doi)]

            if len(_sel_rows) >= 2:
                multiple_entries += len(_sel_rows)
                multiple_entries_df = pd.concat([multiple_entries_df, _sel_rows], ignore_index=True)
                processed_indices.extend(_sel_rows.index.tolist())

            else:
                unique_entries += 1
                unique_entries_df = pd.concat([unique_entries_df, _sel_rows], ignore_index=True)
                processed_indices.extend(_sel_rows.index.tolist())

    all_processed = set(processed_indices) == set(range(len(in_lines)))

    if stats:
        print(f"All rows processed: {all_processed}")
        print('##statistics##')
        print('number of multiple entries for same doi and same ID:', multiple_entries)
        print('number of unique entries for each DOI and ID.:', unique_entries)
        print(f"multiple entries + unique doi entries: {multiple_entries + unique_entries}")
        percent_dupes = (multiple_entries / len(in_lines)) * 100
        print('percent of duplicates:', round(percent_dupes, 2))
    if not all_processed:
        missing_rows = set(range(len(in_lines))) - set(processed_indices)
        print(f"Missing rows indices: {missing_rows}")

    if Verbose:
        print("Number of pharmacological activity after removal of multiple entries from a the same doi:",
              len(unique_entries_df))

    return multiple_entries_df, unique_entries_df
