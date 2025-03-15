target1 = "D2"
assaydefinition1 = "antagonist"
measurement1 = "Ki"

target2 = "D3"
assaydefinition2 = "antagonist"
measurement2 = "Ki"

datadir = "/home/sjwon3789/repositories/ai-DR/datasets/chembl_datasets/C33"
target1_source = f"{datadir}/dataset_{target1}_{assaydefinition1}_{measurement1}/pubdata"
target2_source = f"{datadir}/dataset_{target2}_{assaydefinition2}_{measurement2}/pubdata"

path1 = target1_source
path2 = target2_source

print('local variables are imported')
