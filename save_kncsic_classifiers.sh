#!/bin/bash

START_K="${1:-1}"
END_K="${2:-20}"
N_JOBS="${3:-4}"

# Loop through numbers from START_K to END_K using a C-style loop
for ((k=START_K; k<=END_K; k++)); do
  echo "\n Training Classifiers for k= $k \n ##"
  python train_svm_classifiers.py --feature_type kncsic --k "$k" --n_jobs "$N_JOBS"  --multi_class_mode CommonC
done