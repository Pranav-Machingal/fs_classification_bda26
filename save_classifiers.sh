#!/bin/bash

START_K="${1:-1}"
END_K="${2:-15}"

PLM_FEATURES=("fusionprot" "saprot" "protssn" "prosst")
# KMER_FEATURES=("kncsic" "kndrf")
KMER_FEATURES=("kncsic")

# Train binary classifiers with PLM features
echo -e "\n ## Training binary classifiers with PLM features ##\n"
for feature in "${PLM_FEATURES[@]}"; do
  if [[ "$feature" == "fusionprot" ]]; then
    MAX_K=3
  else
    MAX_K=6
  fi
  echo -e " ---> Feature: $feature (Max Kernels: $MAX_K)"
  python train_svm_classifiers.py --feature_type "$feature" --tasks global_binary input_dep_binary --max_parallel_kernels $MAX_K
done

# Train binary classifiers for k-mer features
for ((k=START_K; k<=END_K; k++)); do
  if [ "$k" -le 4 ]; then
    MAX_K=6
  elif [ "$k" -le 8 ]; then
    MAX_K=5
  elif [ "$k" -le 12 ]; then
    MAX_K=4
  else
    MAX_K=3
  fi
  echo -e "\n ## Training binary classifiers for k = $k (Max Kernels: $MAX_K) ##\n"
  for feature in "${KMER_FEATURES[@]}"; do
    echo -e " ---> Feature: $feature"
    python train_svm_classifiers.py --feature_type "$feature" --k "$k" --tasks global_binary input_dep_binary --max_parallel_kernels $MAX_K
  done
done

# Train 3 class classifiers with PLM features
echo -e "\n ## Training 3 class classifiers with PLM features ##\n"
for feature in "${PLM_FEATURES[@]}"; do
  if [[ "$feature" == "fusionprot" ]]; then
    MAX_K=3
  else
    MAX_K=6
  fi
  echo -e " ---> Feature: $feature (Max Kernels: $MAX_K)"
  python train_svm_classifiers.py --feature_type "$feature" --tasks global_3class input_dep_3class --max_parallel_kernels $MAX_K
done

# Train 3 class classifiers for k-mer features
for ((k=START_K; k<=END_K; k++)); do
  if [ "$k" -le 4 ]; then
    MAX_K=8
  elif [ "$k" -le 8 ]; then
    MAX_K=6
  elif [ "$k" -le 12 ]; then
    MAX_K=6
  else
    MAX_K=6
  fi
  echo -e "\n ## Training 3 class classifiers for k = $k (Max Kernels: $MAX_K) ##\n"
  for feature in "${KMER_FEATURES[@]}"; do
    echo -e " ---> Feature: $feature"
    python train_svm_classifiers.py --feature_type "$feature" --k "$k" --tasks global_3class input_dep_3class --max_parallel_kernels $MAX_K
  done
done