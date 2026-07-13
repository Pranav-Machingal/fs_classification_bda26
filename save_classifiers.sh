#!/bin/bash

START_K="${1:-1}"
END_K="${2:-12}"

PLM_FEATURES=("fusionprot" "saprot" "protssn" "prosst")
KMER_FEATURES=("kncsic" "kndrf")

# # Train binary classifiers with PLM features
# echo -e "\n ## Training binary classifiers with PLM features ##\n"
# for feature in "${PLM_FEATURES[@]}"; do
#   echo -e " ---> Feature: $feature"
#   python train_svm_classifiers.py --feature_type "$feature" --tasks global_binary input_dep_binary 
# done

# # Train binary classifiers for k-mer features
# for ((k=START_K; k<=END_K; k++)); do
#   echo -e "\n ## Training binary classifiers for k = $k ##\n"
#   for feature in "${KMER_FEATURES[@]}"; do
#     echo -e " ---> Feature: $feature"
#     python train_svm_classifiers.py --feature_type "$feature" --k "$k" --tasks global_binary input_dep_binary
#   done
# done

# Train 3 class classifiers with PLM features
echo -e "\n ## Training 3 class classifiers with PLM features ##\n"
for feature in "${PLM_FEATURES[@]}"; do
  echo -e " ---> Feature: $feature"
  python train_svm_classifiers.py --feature_type "$feature" --tasks global_3class input_dep_3class 
done

# Train 3 class classifiers for k-mer features
for ((k=START_K; k<=END_K; k++)); do
  echo -e "\n ## Training 3 class classifiers for k = $k ##\n"
  for feature in "${KMER_FEATURES[@]}"; do
    echo -e " ---> Feature: $feature"
    python train_svm_classifiers.py --feature_type "$feature" --k "$k" --tasks global_3class input_dep_3class
  done
done