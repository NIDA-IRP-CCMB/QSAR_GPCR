def filter_low_pchembl_values(buffer, cutoff = 4, Verbose = False):
    buffer = buffer[buffer['pchembl_value']>=cutoff].reset_index(drop = True)
    
    if Verbose:
        print(f'Number of pharmacological activity after removing low pchembl values: ', len(buffer))
        
    return buffer