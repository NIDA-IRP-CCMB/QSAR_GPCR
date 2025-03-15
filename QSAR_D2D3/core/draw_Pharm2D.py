# Draw_Pharm2D.py
'''
    WIP: Module for mapping an dvisulaizing tje  Gobbi-Pharm2D fingerprints.
    So far, this module works through passing the mol object, a compound identifier (for labelling), and the target bit identifier:
    e.g., Draw_Pharm2D(mol, 'CHEMBLID' ,f'Ph2D_{bit}')
    #TODO (BRBV): use CHEMBLID instead of generating mol object first.,
'''
import sys, os
import io
import copy

from rdkit import Chem
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem.Pharm2D import Utils, SigFactory
from rdkit.Chem import Draw
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
from rdkit.Chem.rdDepictor import Compute2DCoords
from rdkit.Chem.rdMolTransforms import TransformConformer
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem import MolFromSmiles
from IPython.display import SVG
from collections import defaultdict


import numpy as np
import pandas as pd

import IPython.display as disp


def get_description(feature):
    if feature.startswith('Ph2D_'):
        bit_number = int(feature.split('_')[1])
        return Gobbi_Pharm2D.factory.GetBitDescription(bit_number)
    else:
        descriptor = getattr(Chem.Descriptors, feature)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        help(descriptor)
        descriptor_description = sys.stdout.getvalue()
        sys.stdout.close()
        sys.stdout = old_stdout

    # Extract and store the last non-empty line with characters
        lines = descriptor_description.split('\n')
        last_line = next((line for line in reversed(lines) if line.strip()), None)

        return last_line


def convert_id_to_mol(CHEMBLID, ls_filenames):
    """Using ChEMBL ID to get chemical mol
    :param chembl_id: numbers only, chembl structure from ls_filenames
    :param ls_filenames: list of chembl training dataset
    :return: chemical structure, mol
    """

    if isinstance(CHEMBLID, int):
        CHEMBLID = "CHEMBL" + str(int(CHEMBLID))

    dfs = []
    for filename in ls_filenames:
        df1 = pd.read_csv(filename + ".act", sep="\t", header=None)
        df1.columns = ["chembl", "pKi"]
        df2 = pd.read_csv(filename + ".smi", sep="\t", header=None)
        df2.columns = ["smi", "chembl"]
        df = pd.merge(df1, df2, on="chembl", how="outer")
        dfs.append(df)

    df = pd.concat(dfs)
    df = df.drop_duplicates(subset="chembl")
    df = df.reset_index(drop=True)

    dict_chembl = df.set_index("chembl")["smi"].to_dict()
    smi = dict_chembl.get(CHEMBLID)

    if smi is None:
        print("Chembl ID not found.")
        return None

    mol = Chem.MolFromSmiles(smi)
    return mol


def convert_drugbank_to_smi(drugbank_compound, ls_filenames):
    """
    Using drugbank compound name from ls_filenames to get chemical structure, mol
    :param drugbank_compound: string, compound name in ls_filenames, e.g., Haloperidol
    :param ls_filenames: list of .csv files from drugbank
    :return: chemical structure, mol
    """
    dfs = []
    for filename in filenames:
        df = pd.read_csv(filename)
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)
    dict_drugbank = df.set_index("compound")["Processed_SMILES"].to_dict()
    smi = dict_drugbank[drugbank_compound]
    mol = Chem.MolFromSmiles(smi)
    return mol


def GetMolFeats(mol):
    featFactory = ChemicalFeatures.BuildFeatureFactoryFromString(Gobbi_Pharm2D.fdef)
    factory = SigFactory(featFactory, minPointCount=2, maxPointCount=3)
    factory.SetBins(defaultBins)
    featFamilies = Gobbi_Pharm2D.factory.GetFeatFamilies()
    featMatches = {fam: [feat.GetAtomIds() for feat in featFactory.GetFeaturesForMol(mol, includeOnly=fam)] for fam in featFamilies}

    return featMatches


def GetAtomsMatchingBit(sigFactory, bitIdx, mol, dMat=None, justOne=False, 
                        matchingAtoms=None, verbose=False):
    assert sigFactory.shortestPathsOnly, 'not implemented for non-shortest path signatures'
    nPts, featCombo, scaffold = sigFactory.GetBitInfo(bitIdx)

    fams = sigFactory.GetFeatFamilies()

    if matchingAtoms is None:
        matchingAtoms = sigFactory.GetMolFeats(mol)

    choices = []

    for featIdx in featCombo:
        tmp = matchingAtoms[featIdx]
        if tmp:
            choices.append(tmp)
        else:
            return []

    if dMat is None:
        dMat = Chem.GetDistanceMatrix(mol, sigFactory.includeBondOrder)

    distsToCheck = Utils.nPointDistDict[nPts]
    protoPharmacophores = Utils.GetAllCombinations(choices, noDups=True)
    res = []

    for protoPharm in protoPharmacophores:
        for i in range(len(distsToCheck)):
            dLow, dHigh = sigFactory.GetBins()[scaffold[i]]
            a1, a2 = distsToCheck[i]
            idx1, idx2 = protoPharm[a1][0], protoPharm[a2][0]
            dist = dMat[idx1][idx2]
            if dist < dLow or dist >= dHigh:
                break
        else:
            if protoPharm not in res:
                res.append(protoPharm)
                if justOne:
                    break

    if not res:
        return []

    else:                
        result_list = []

        for res_set in res:
            current_dict = defaultdict(list)
            for i, feat in enumerate(featCombo):
                current_dict[fams[feat]].append(res_set[i][0])
            current_dict = dict(current_dict)
            result_list.append(current_dict)

        if verbose:
            print(f"Gobbi_Pharm2D bit description: {Gobbi_Pharm2D.factory.GetBitDescription(bitIdx)}")
#             print(f'nPts: {nPts}; featCombo: {featCombo}; scaffold: {scaffold}')
            print(f'Note: distance (number of bonds) greater than 8 are still assigned an 8!')
            if len(featCombo) == 2:
                print(f'Bond Distance Constraints:',
                      f'{fams[featCombo[0]]}-{fams[featCombo[1]]} =',
                      f'{Gobbi_Pharm2D.factory.GetBins()[scaffold[0]][0]}')
            else:
                print(f'Bond Distance Constraints:',
                      f'{fams[featCombo[0]]}-{fams[featCombo[1]]} =',
                      f'{Gobbi_Pharm2D.factory.GetBins()[scaffold[0]][0]} ;',
                      f'{fams[featCombo[0]]}-{fams[featCombo[2]]} =',
                      f'{Gobbi_Pharm2D.factory.GetBins()[scaffold[1]][0]} ;',
                      f'{fams[featCombo[1]]}-{fams[featCombo[2]]} =',
                      f'{Gobbi_Pharm2D.factory.GetBins()[scaffold[2]][0]}')
   
            print(f'There are {len(result_list)} Matching Scaffolds: {result_list}')
            
        return result_list


def get_svg(mol, CHEMBLID, bit_number, fp_centers, concat_highlight,
            save=False, output_dir = None, labels=False):
    
    d2d2 = Draw.MolDraw2DSVG(800,400)
    dopts2 = d2d2.drawOptions()
    dopts2.highlightRadius = .4
    dopts2.highlightBondWidthMultiplier = 16
    dopts2.fixedBondLength = 25
    if labels:
        dopts2.addAtomIndices = True
        d2d2.DrawMolecule(mol,legend=" | ".join([f"{key}: {value}" for key, value in fp_centers.items()]),
                          highlightAtoms=concat_highlight)
    else:
        d2d2.DrawMolecule(mol, highlightAtoms=concat_highlight)
        
    d2d2.FinishDrawing()

    svg = d2d2.GetDrawingText()
    if save:
        if output_dir is None:
            print('Specify output_dir for svg files')
        else:
            os.makedirs(output_dir, exist_ok=True)
            with open(
                    f'{output_dir}/{CHEMBLID}_Ph2D_{bit_number}_'
                    f'{"_".join(f"{value}" for key, value in fp_centers.items())}.svg',
                    'w'
            ) as f:
                f.write(svg)
    
    disp.display(disp.SVG(svg))


def transform_matrix(rot_x, rot_y, rot_z):
    """
    Generate transformation matrix for required molecule 
    rotations.
    """
    rx = np.radians(rot_x)
    ry = np.radians(rot_y)
    rz = np.radians(rot_z)
    
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rx), -np.sin(rx), 0],
        [0, np.sin(rx), np.cos(rx), 0],
        [0, 0, 0, 1]
    ])
    
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry), 0],
        [0, 1, 0, 0],
        [-np.sin(ry), 0, np.cos(ry), 0],
        [0, 0, 0, 1]
    ])
    
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0, 0],
        [np.sin(rz), np.cos(rz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    return R


def moltosvg(mol, filename="", rot_x = 0, rot_y = 0, rot_z = 0):
    Compute2DCoords(mol, canonOrient=True)
    transmat = transform_matrix(rot_x, rot_y, rot_z)
    TransformConformer(mol.GetConformer(0), transmat)

    canvas_width_pixels = 500
    canvas_height_pixels = 500

    mol2 = rdMolDraw2D.PrepareMolForDrawing(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(canvas_width_pixels, canvas_height_pixels)
    drawer.DrawMolecule(mol2)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace('svg:',
                                          '')  # Fixing a common issue with RDKit SVG output in certain environments

    if filename:
        with open(filename + '.svg', 'w') as f:
            f.write(svg)
    else:
        return svg
    disp.display(disp.SVG(svg))


def Draw_Pharm2D(mol, CHEMBLID, feat2D_name, output_dir = None, show_fig = True,
                 verbose=False, save=False, rot_x = 0, rot_y = 0, rot_z = 0, labels=False):

    if not feat2D_name.startswith('Ph2D_'):
        print(f"The feature {feat2D_name} is not from Gobbi_Pharm2D.")
    else:
        bit_number = int(feat2D_name.split('_')[1])

    highlights = GetAtomsMatchingBit(Gobbi_Pharm2D.factory,bit_number,mol,verbose=verbose)

    if not highlights:
        if verbose:
            print(f"{feat2D_name} is not present in {CHEMBLID}.")
        # return 0
        Compute2DCoords(mol, canonOrient=True)
        transmat = transform_matrix(rot_x, rot_y, rot_z)
        TransformConformer(mol.GetConformer(0), transmat)

        canvas_width_pixels = 500
        canvas_height_pixels = 500

        mol2 = rdMolDraw2D.PrepareMolForDrawing(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(canvas_width_pixels, canvas_height_pixels)
        drawer.DrawMolecule(mol2)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText().replace('svg:',
                                              '')  # Fixing a common issue with RDKit SVG output in certain environments
        if save:
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/{CHEMBLID}_Ph2D.svg", 'w') as f:
                f.write(svg)
        else:
            return svg
        disp.display(disp.SVG(svg))

    else:
        if verbose:
            print(f'Atoms Matching Bit: {highlights}')
        if show_fig:
            Compute2DCoords(mol,canonOrient=True) 
            transmat = transform_matrix(rot_x, rot_y, rot_z)
            TransformConformer(mol.GetConformer(0),transmat)

            for highlight in highlights:
                fp_centers = copy.copy(highlight)

                # Extend the atom selections #TODO: BRBV Update for other possible families as needed
                for feat in ['AR', 'RR']:
                    if feat in highlight:
                        ri = mol.GetRingInfo()
                        unique_atoms = set()
                        update_value = []
                        for i, atom_id in enumerate(highlight[feat]):
                            for ring in ri.AtomRings():
                                if atom_id in ring:
                                    if feat == 'AR':
                                        if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                                            ring_atoms = ring
                                            break
                                    else:
                                        ring_atoms = ring
                                        break
                            unique_atoms.add(atom_id)
                            unique_atoms.update(ring_atoms)
                            update_value += list(unique_atoms)
                        highlight[feat] = update_value

                for feat in ['LH']:
                    if feat in highlight:
                        unique_atoms = set()
                        update_value = []
                        for i, atom_id in enumerate(highlight[feat]):
                            central_atom = mol.GetAtomWithIdx(atom_id)
                            C_neigh_idx = []
                            for neigh in central_atom.GetNeighbors():
                                if neigh.GetAtomicNum() == 6:
                                    C_neigh_idx.append(neigh.GetIdx())
                            unique_atoms.add(atom_id)
                            unique_atoms.update(C_neigh_idx)
                            update_value += list(unique_atoms)
                        highlight[feat] = update_value

                if verbose:
                    print(f"\nExpanded scaffold atoms: {highlight}")

                concat_highlight = []

                for value in highlight.values():
                    if isinstance(value, list):
                        concat_highlight.extend(value)
                    elif isinstance(value, int):
                        concat_highlight.append(value)

                get_svg(mol, CHEMBLID, bit_number, fp_centers, concat_highlight, 
                        save=save, output_dir=output_dir, labels=labels)
