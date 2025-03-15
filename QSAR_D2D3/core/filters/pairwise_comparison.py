from rdkit.Chem import AllChem as Chem
from rdkit import DataStructs

def fingerprints(in_lines, useIsomer=False):
    '''
    Take fingerprints of all molecules
    '''

    fingerprints_df = []

    for i in range(len(in_lines)):
        mol_in = Chem.MolFromSmiles(in_lines['canonical_smiles'][i])
        if mol_in is not None:
            fp = Chem.GetMorganFingerprintAsBitVect(mol_in, 2, useChirality=useIsomer, nBits=2048)  # set nbits equal to what you will use in model?
            fingerprints_df.append(fp)
        elif mol_in is None:
            mol_in = Chem.MolFromSmiles("")
            fp = Chem.GetMorganFingerprintAsBitVect(mol_in, 2, useChirality=useIsomer, nBits=2048)
            fingerprints_df.append(fp)

    return fingerprints_df

def pairwise_comparison(in_lines, thresh=0.999, useIsomer=False):
    '''
    Determine similar compounds based on TanimotoSimilarity.
    Pairwise comparison of all fingerprints (matching pairs go into similars)
    '''

    fingerprints_df = fingerprints(in_lines, useIsomer)
    similars = []

    for i in range(0, ((len(fingerprints_df)) - 1), 1):
        for j in range((i + 1), (len(fingerprints_df))):
            if DataStructs.TanimotoSimilarity(fingerprints_df[i], fingerprints_df[j]) > thresh:
                similars.append((i, j))

    return similars