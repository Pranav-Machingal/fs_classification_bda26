import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm
from pathlib import Path
import sys

labels_df = pd.read_csv('dataset/data_split.csv')
d103_features_dict = pickle.load(open('saved_data/feature_extraction/kncsic_103d_features.pkl', 'rb'))

switch_labels = [f'{s1}/{s2}' for s1 in ['a', 'b', 'c'] for s2 in ['a', 'b', 'c']]

def save_kncsic_feature_data_splits(k):
  X_train_list = []
  y_train_list = []
  X_val_list = []
  y_val_list = []
  X_test_list = []
  y_test_list = []
  
  for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Building k={k} Data Split"):
    if row.get('Switch_Label', 'x') != 'x':
      chain_id = row['Raw_Input_ID']
      center_res_id = row['Residue_ID']
      key = (chain_id, center_res_id)
      
      label = f"{row['SS_Label']}/{row['Switch_Label']}"
      label_idx = switch_labels.index(label)
      
      # Construct the (2k+1) * 103 feature vector
      k_mer_feature = []
      for offset in range(-k, k + 1):
        neighbor_key = (chain_id, center_res_id + offset)
        
        # If neighbor exists, append its 103D vector
        if neighbor_key in d103_features_dict:
          k_mer_feature.append(d103_features_dict[neighbor_key])
        else:
          # If neighbor is missing (e.g., at chain terminus), pad with zeros
          k_mer_feature.append(np.zeros(103))
          
      # Flatten the list of arrays into a single 1D array
      k_mer_feature = np.concatenate(k_mer_feature)
  
      split = row['split']
      if split == 'train':
        X_train_list.append(k_mer_feature)
        y_train_list.append(label_idx)
      elif split == 'val':
        X_val_list.append(k_mer_feature)
        y_val_list.append(label_idx)
      elif split == 'test':
        X_test_list.append(k_mer_feature)
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
  
  save_file_path = save_dir / f'k{k}ncsic.npz'
  
  np.savez(save_file_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)
  print(f"Saved {save_file_path}")

if __name__ == "__main__":
  for k in range(1, 25):
    save_kncsic_feature_data_splits(k)
