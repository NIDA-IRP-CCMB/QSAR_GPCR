import os
import inspect
import pandas as pd

def data_to_pickle(filtered_df, output_dir=None, save_removed=False, original_df=None):
    '''
    This function saves the buffer df into a tsv file. This can be called from any filter function but should be
    done after filtering. This also have the option to save the removed data on another dataframe. For this option
    to work, save the starting in_lines df into a variable 'raw' before doing any filtering. Then call this function
    after the filtering is done with the latest in_line df as the first argument, save_removed=True, and original_df=raw.
    '''

    # Get the name of the calling function
    calling_function = inspect.stack()[1].function

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(output_dir, 'filter_data')
    os.makedirs(output_dir, exist_ok=True)

    # Save the filtered DataFrame to _filtered.tsv
    filtered_df.to_pickle(os.path.join(output_dir, f'{calling_function}_filtered_{len(filtered_df)}.dat'))

    # Optionally, save the removed data into _removed.tsv
    if save_removed and original_df is not None:
        removed_indices = original_df.index[~original_df.index.isin(filtered_df.index)]
        removed_df = original_df.loc[removed_indices]
        removed_df.to_pickle(os.path.join(output_dir, f'{calling_function}_removed_{len(removed_df)}.dat'))