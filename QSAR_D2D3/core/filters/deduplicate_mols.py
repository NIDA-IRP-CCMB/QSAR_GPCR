import pandas as pd
import math
from .calc_pscale import calc_pscale
from .pairwise_comparison import pairwise_comparison
from .data_to_pickle import data_to_pickle


def deduplicate_mols(in_lines, output_dir=None, save=False, Verbose=False, useIsomer=False):
    # verify that we only have one instance of a molecule in the final set
    # this needs to be the last filter run
    if save:
        raw = in_lines

    working_copy = pd.DataFrame()
    mol_duplicates = []
    holding = []
    seen = []

    mol_dup_pairs = pairwise_comparison(in_lines, thresh=0.999, useIsomer=useIsomer)

    for dup_pair in mol_dup_pairs:
        if dup_pair[0] not in seen and dup_pair[1] not in seen and dup_pair[0] not in holding:
            holding.append(dup_pair[0])
            holding.sort()
            seen.append(dup_pair[0])
            seen.append(dup_pair[1])
            seen.sort()
        if dup_pair[1] not in seen:
            seen.append(dup_pair[1])
            seen.sort()

    for item in holding:
        mol_duplicates.append([item])

    for dup_pair in mol_dup_pairs:
        for i in range(len(mol_duplicates)):
            if dup_pair[0] in mol_duplicates[i] and dup_pair[1] not in mol_duplicates[i]:
                mol_duplicates[i].append(dup_pair[1])

    # select lowest affinity value out of all values per compound (except 1)

    for dup_group in mol_duplicates:

        all_acts = []
        sub_lines = []

        # make certain that we have a pchembl value for each entry and make a list of them

        for line_no in dup_group:
            sub_lines.append(in_lines.iloc[[line_no]])
            if math.isnan(in_lines['pchembl_value'][line_no]):
                all_acts.append(calc_pscale(in_lines['standard_value'][line_no],
                                            in_lines['standard_units'][line_no]))
            else:
                all_acts.append(in_lines['pchembl_value'][line_no])

        # if the values we have are all the same, take the first one and move on

        if min(all_acts) == max(all_acts):
            # if redundant data has the same activity data, just use that value
            working_copy = pd.concat([working_copy, sub_lines[0]])

        else:
            # Deal with boundaries:
            # 1. Remove the upper boundary condition (<11) to focus on high affinity binders
            # 2. Retain the existing lower boundary (>1) to incorporate low affinity binders
            #
            # Determine the approach for selecting values from duplicate data:
            # 1. Using the median value when there are many data points (>3 datapoints)
            # 2. Using the maximum value when there are for two redundant data points (<=3 datapoints)

            order = sorted(range(len(all_acts)), key=lambda k: all_acts[k])
            # print(order)
            # print(all_acts)
            if min(all_acts) > 1:
                if len(all_acts) <= 3:
                    max_index = all_acts.index(max(all_acts))
                    sel_index = max_index
                else:
                    mid_index = order[int(len(order) / 2)]
                    sel_index = mid_index
            else:
                order = [num for num in order if 1 < all_acts[num]]
                if len(order) <= 3:
                    max_index = order[-1]
                    sel_index = max_index
                else:
                    mid_index = order[int(len(order) / 2)]
                    sel_index = mid_index
            working_copy = pd.concat([working_copy, sub_lines[sel_index]])
            # print(all_acts[sel_index])

    in_lines = in_lines.drop(list(set(seen)))
    in_lines = pd.concat([in_lines, working_copy])

    if Verbose:
        print(f'Number of pharmacological activity after deduplication pass: ', len(in_lines))

    if save:
        data_to_pickle(in_lines, output_dir, save_removed=True, original_df=raw)

    return in_lines.reset_index(drop=True)