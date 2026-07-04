import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm
from pathlib import Path
import sys


labels_df = pd.read_csv('residue_switch_labels.csv')
train_val_dict = pickle.load(open('train_val_split0.pkl', 'rb'))
train_ids = train_val_dict['train']
val_ids = train_val_dict['val']
d29_features_dict = pickle.load(open('saved_data/feature_extraction/29d_features.pkl', 'rb'))

switch_labels = [f'{s1}/{s2}' for s1 in ['a', 'b', 'c'] for s2 in ['a', 'b', 'c']]

def save_kndr_feature_data_splits(k):
  X_train_list = []
  y_train_list = []
  X_val_list = []
  y_val_list = []
  X_test_list = []
  y_test_list = []
  
  for _, row in tqdm(labels_df.iterrows(), total=len(labels_df)):
    if row['switch_SS_label'] != 'x':
      chain_id = row['Raw_Input_ID']
      center_res_id = row['Residue_ID']
      key = (chain_id, center_res_id)
      
      label = f"{row['SS_Label']}/{row['switch_SS_label']}"
      label_idx = switch_labels.index(label)
      
      # Construct the (2k+1) * 29 feature vector
      k_mer_feature = []
      for offset in range(-k, k + 1):
        neighbor_key = (chain_id, center_res_id + offset)
        
        # If neighbor exists, append its 29D vector
        if neighbor_key in d29_features_dict:
          k_mer_feature.append(d29_features_dict[neighbor_key])
        else:
          # If neighbor is missing (e.g., at chain terminus), pad with zeros
          k_mer_feature.append(np.zeros(29))
          
      # Flatten the list of arrays into a single 1D array
      k_mer_feature = np.concatenate(k_mer_feature)
  
      if key in train_ids:
        X_train_list.append(k_mer_feature)
        y_train_list.append(label_idx)
      elif key in val_ids:
        X_val_list.append(k_mer_feature)
        y_val_list.append(label_idx)
      else:
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
  
  save_file_path = save_dir / f'k{k}ndrf.npz'
  
  np.savez(save_file_path, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

if __name__ == "__main__":
  for k in range(21,31):
    save_kndr_feature_data_splits(k)