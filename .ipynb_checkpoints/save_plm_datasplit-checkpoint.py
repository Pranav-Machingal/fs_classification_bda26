import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm
from pathlib import Path
import sys
import argparse

parser = argparse.ArgumentParser(description="Save train/val/test split features")
parser.add_argument('--feature_type', type=str, default='saprot', choices=['saprot', 'fusionprot', 'protssn', 'prosst'], help="Feature type for creating splits")
args = parser.parse_args()
FEATURE_TYPE = args.feature_type

labels_df = pd.read_csv('dataset/data_split.csv')
features_dict = pickle.load(open(f'saved_data/feature_extraction/{FEATURE_TYPE}_features.pkl', 'rb'))

switch_labels = [f'{s1}/{s2}' for s1 in ['a', 'b', 'c'] for s2 in ['a', 'b', 'c']]

def save_feature_data_splits():
  X_train_list = []
  y_train_list = []
  X_val_list = []
  y_val_list = []
  X_test_list = []
  y_test_list = []
  
  for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    if row.get('Switch_Label', 'x') != 'x':
      chain_id = row['Raw_Input_ID']
      center_res_id = row['Residue_ID']
      key = (chain_id, center_res_id)
      
      label = f"{row['SS_Label']}/{row['Switch_Label']}"
      label_idx = switch_labels.index(label)
      feature = features_dict[key]
      split = row['split']
      if split == 'train':
        X_train_list.append(feature)
        y_train_list.append(label_idx)
      elif split == 'val':
        X_val_list.append(feature)
        y_val_list.append(label_idx)
      elif split == 'test':
        X_test_list.append(feature)
        y_test_list.append(label_idx)
  
  X_train = np.array(X_train_list)
  y_train = np.array(y_train_list)
  X_val = np.array(X_val_list)
  y_val = np.array(y_val_list)
  X_test = np.array(X_test_list)
  y_test = np.array(y_test_list)
  
  # Ensure output directory exists before saving
  save_dir = Path('saved_data/feature_extraction')
  save_dir.mkdir(parents=True, exist_ok=True)
  
  save_file_path = save_dir / f'{FEATURE_TYPE}.npz'
  
  np.savez(save_file_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
  save_feature_data_splits()