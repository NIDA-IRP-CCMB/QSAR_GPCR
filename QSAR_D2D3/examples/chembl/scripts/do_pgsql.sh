current_dir=$PWD
#mkdir all_pgsql
#lsdir=("D2" "D3")
lschembl=("33")

#for idir in "${lsdir[@]}"
#do
#cd ${current_dir}/${idir}_pgsql

for ichembl in "${lschembl[@]}"
do
echo ${idir}
psql -h 127.0.0.1 chembl_${ichembl} chembl  --field-separator=$'\t'  --no-align -f chembl_Classifier.sql -o chembl_Classifier_c${ichembl}.tsv
wc -l chembl_Classifier_c${ichembl}.tsv | grep -v clean

sed '1d;$d' chembl_Classifier_c${ichembl}.tsv > chembl_Classifier_c${ichembl}_clean.tsv
wc -l chembl_Classifier_c${ichembl}_clean.tsv

cp chembl_Classifier_c${ichembl}_clean.tsv ../all_pgsql/chembl${ichembl}_${idir}.tsv
wc -l ../all_pgsql/chembl${ichembl}_${idir}.tsv
done

#cd ..
#done

