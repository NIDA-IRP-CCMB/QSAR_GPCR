# Introduction

C31 and C32 are old ChEMBL versions, datasets created with the filter updates (using version 3 for D2 and version 2 for D3)


For C33: 

* C33 		------> assaydefinition + hand select + binding B changes
* C33_2		------> same as above, but year(1990) and bao format (tissue,assay) changes made. 
* C33_6         ------> last stage of filtering process (version 2 for D3)

Make sure the following files are in each dataset folders. Use the following command:
$ bash ../scripts/prep_datasets.sh

# Annotations

Please see step 5 in the folder datasets. These are the manual curations made corresponding to the descriptions listed there.

* Include "CHEMBL198174" in to_remove.txt.

* Additionally, we looked at:


       D2 - CHEMBL8514 

       D3 - CHEMBL564709, CHEMBL198174

* We removed CHEMBL198174. The raw data is "12000+/-0", which does not make sense, as it cannot be "+/-0". We concluded that this was a strong sign that the study was not able to be measured accurately or properly.

* The CHEMBL564709 was correct and accurate.

* Based on our observation, if CHEMBL8514 is butaclamol, the annotation of 0.04 nM in Table 3 appeared to be correct.  However, from two other papers listed for butaclamol, it should not be this high of an affinity.

 * https://www.ebi.ac.uk/chembl/g/#browse/activities/filter/molecule_chembl_id%3A(%22CHEMBL8514%22%20OR%20%22CHEMBL1256753%22)%20AND%20standard_type%3A(%22Ki%22)
0.27 nM from CHEMBL4000185

* In a separate paper that appears not to be in chembl, it is 0.44 nM. See table 2 of the following
https://statics.teams.cdn.office.net/evergreen-assets/safelinks/1/atp-safelinks.html
