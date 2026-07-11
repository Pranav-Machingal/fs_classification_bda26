#!/bin/bash

START_K="${1:-1}"
END_K="${2:-10}"
N_JOBS1="${3:-8}"
N_JOBS2="${3:-4}"

# Loop through numbers from START_K to END_K using a C-style loop
for ((k=START_K; k<=END_K; k++)); do
    echo "Training binary classifiers for k= $k"
    python train_svm_classifiers.py --feature_type kncsic --k "$k" --n_jobs "$N_JOBS1" --tasks global_binary input_dep_binary
done

for ((k=START_K; k<=END_K; k++)); do
    echo "Training 3 class classifiers for k= $k"
    python train_svm_classifiers.py --feature_type kncsic --k "$k" --n_jobs "$N_JOBS2" --tasks global_3class input_dep_3class --multi_class_mode CommonC
done