Previously... we have added the TIDs manually into the .sql files. Please see ../README.md or ../target_ids

# Creating tsv files
1) Please make sure TIDs are updated correctly. 

2) Using .sql file, we can create .tsv files. Make sure 'chembl_Classifier.sql' is in the designated folder, e.g. inside D2_pgsql

		$ ssh shilab5

		$ bash ../../scripts/do_pgsql.sh


3) Verify numbers (maybe unittest in future?)

	D2 
		29979 chembl_Classifier_c33.tsv
		29977 chembl_Classifier_c33_clean.tsv
	D3
		12673 chembl_Classifier_c33.tsv
		12671 chembl_Classifier_c33_clean.tsv
