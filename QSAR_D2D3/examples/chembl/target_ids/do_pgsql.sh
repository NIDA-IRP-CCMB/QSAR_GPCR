
psql -h 127.0.0.1 chembl_33 chembl  --field-separator=$'\t'  --no-align -f chembl_Classifier.sql -o chembl_Classifier_c${ichembl}.tsv
wc -l chembl_Classifier_c${ichembl}.tsv | grep -v clean

sed '1d;$d' chembl_Classifier_c${ichembl}.tsv > chembl_Classifier_c${ichembl}_clean.tsv
wc -l chembl_Classifier_c${ichembl}_clean.tsv
