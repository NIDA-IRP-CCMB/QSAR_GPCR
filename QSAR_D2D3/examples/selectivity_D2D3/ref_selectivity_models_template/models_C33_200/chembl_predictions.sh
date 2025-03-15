#!/bin/bash
cd models_D2
bash all_predictions.sh
cd ..
cd models_D3
bash all_predictions.sh
cd ..
cd models_D2_overlap
bash all_predictions.sh
cd ..
cd models_D3_overlap
bash all_predictions.sh
cd ..
cd models__ratio_D2D3
bash all_predictions.sh
cd ..
