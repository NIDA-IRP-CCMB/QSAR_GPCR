#!/bin/bash
for i in {0..9}
do
echo $i
cd model_$i
bash do_buildmodel.sh
cd ..
done
