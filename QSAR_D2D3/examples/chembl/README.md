## Datasets

To create dopamine D2 and D3 receptor antagonist/agonist datasets, follow the following instructions:

1. First, prepare your tid numbers. See target_ids/README.md. These numbers are manually added to the chembl_Classifier.sql files in each protein target folder.

2. Add these updated chembl_Classifier.sql files in pgsql/D2_pgsql and pgsql/D3_pgsql.

3. Generate .tsv files using .sql files.

		$ cd pgsql

		$ bash ../scripts/do_pgsql.sh

You should have tsv files such as chembl_Classifier_c32_clean.tsv


4. We need to get our **keyterms** for antagonists and agonists and add them to conf/assaydefinition_DR_antagonist.txt and conf/assaydefinition_DR_agonist.txt . To do this, please see notebooks/ligand_keyterms.ipynb and update the keywords as you work through the notebook. This notebook uses the functions in scripts/filters_query.py

		You may need to soft link the conf folder

		$ ln -sf ../conf 

5. Now that we have our keywords, we have everything in place. We can generate our final datasets for D2_antagonist, D3_antagonist, D2_agonist, and D3_agonist using our current infrastructures. scripts/run_filters.py uses scripts/filters_dop.py. Please note that filters_dop.py is a modified version of ~/repositories/ai-x/core/filters.py, some of which are specific to dopamine receptor itself. Therefore, there are specific changes included that you may have to modify and tailor towards your protein target. These additional filter information can be seen in the section below. 
		
		$ module load python/anaconda3-2020.02-py3.7.6-ai  # use the correct python version
		
		$ python scripts/run_filters.py
		
6. You have your final datasets now called new_datasets. Rename this to chembl_datasets or whatever you prefer. You can use this to build your models now.
	

## Changes in the functions in filters_dop.py

The original filters.py can be found in core/filters. This version has been adapted by incorporating additional filters for the D2 and D3.

The following functions have been modified:
		
		read_data - included "bao_format' in the list of labels
		
		filter_confidence - included the choice of broad vs not-broad
		
		filter_assaydefinition - robust searches done now, using "natural language processing"
		
		filter_assay_type - DR antagonists use assay_type B only
		
		filter_year - NEW function, omit the data before year 1990 for D2 dataset
		
		filter_bao_format - NEW function, remove tissue-based and assay studies 
		
		filter_selected - NEW function, specifically searched compounds with "antagonists" as the keyterm for antagonists, then removed those specific ones. Also removed PATENT studies and Literature Reviews. 

For more information on specific columns in the tables we used, please refer [here](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/schema_documentation.html).

## Specific Changes in Detail

For more specific information about the changes made,

1. AssayDefinition: We updated the assay_definition to capture more compounds. Even though the keyterms should have matched, some were missed due to mis-annotations. These included mistakes with additional characters, e.g. dashes, spaces, and lowercases. 

2. asssay_type: We decided to exclude those bound through 'F' and make the list exclusive to 'B'.

3. Year: DR3 was discovered in fall 1990. Therefore, we decided to exclude DR2 compounds from 1990 and before, since these may have been mixed with unknown DR3 compounds.

4. BAO_FORMAT: This has information of how the study was conducted.  We want to exclude tissue-based studies. We want to include cell-based. To be determined: single protein, cell membrane, microsome, assay. Please see the excel files in notebooks/output_dir_papers.

5. Selected: 

	a) We noticed that some papers are PATENT labeled under 'src_name' column. We removed these. 

	b) We investigated the compounds with "antagonist" and "agonist" keyterms more closely. We hand-selected these papers and further filtered them out through their 'doc_id.' The annotations can be seen below. Additionally, please see the excel files in notebooks/output_dir_papers. For additional information on the specific reasons or numbers for each curations, please see datasets/chembl_datasets/README.md. 


	D2 antagonist: 
	
		48827 - citing ref11 therein; therefore, repeating information.
		
		71409 - review paper, therefore repeating information.
		
		81789 - review paper
		
		119044 - review paper
		
	D2 agonist: 
		
		71409 (same as above)
		
		77073 - review paper
		
		
	D3 antagonist: 
	
		good (?)
	
	D3 agonist: 
	
		77073 (same as above)


c) We searched through the keyterm "GTP" to filter through GTPgamma S assay. If these appear in description, they are in agonistic mode, not antagonistic.

	D2:
	
		48827 (does not contain antagonist or agonist in description)

	D3:
	
		48827 (does not contain antagonist or agonist in description)
		
		98345 (contains antagonist in description)
		
		98610 (contains antagonist in description)


Other future todos: GTP assay S makes antagonistic into agonist. These are not included in agonist yet. However, since our main study did not include agonists, this is not relevant for this study.


