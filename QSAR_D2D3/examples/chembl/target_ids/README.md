
# Necessary Files

	get_target_dict.sql			--> for generating target_dict.tsv
 	chembl_Classifier.sql 			--> modify accordingly with the retrieved TIDs
 	search_tids.ipynb			--> for searching descriptions and TIDs of D2, D3
 	search_tids_cleaned_d1d4d5.ipynb	--> for searching descriptions and TIDs of D1, D4, D5

# Instructions to retrieving target ids for each protein targets

These target IDs are used in the .sql files.
1) Using get_target_dict.sql (must have the file), generate target_dict.tsv. 



		$ ssh shilab5 # where psql is located

         $ psql -h 127.0.0.1 chembl_33 chembl --field-separator=$'\t'  --no-align -f get_target_dict.sql -o target_dict.tsv

2) Using search_tids.ipynb, find the associate ID numbers. You must experiment with diffferent keyterms to try and capture all the data points associated with each target ids, e.g. for D2, we tested "Dopamine 2", "Dopamine2", "D2" and much more (not written in the code). "D2" is the only keyterm needed to capture all the information we need. Please refer to search_tids_cleaned_d1d4d5.ipynb for searching D1, D4, and D5.

3) Organize the IDs and pick the correct, accurate ones based on the description labeling. Double check to see if some compounds with certain descriptions are appropriate for the final data set or not. This is where certain manual curation may be needed.

4) Once you have the final list (as you can see in the markdown annotation in the notebook), manually input these numbers in ../pgsql/D2_pgsql/chembl_Classifier.sql and ../pgsql/D3_pgsql/chembl_Classifier.sql

Your sql files are now ready to be used to generate tsv files. Please go back to ../pgsql (one tab back) for next steps.



