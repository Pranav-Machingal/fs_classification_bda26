import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import networkx as nx
from tqdm.auto import tqdm

def format_id(raw):
  if len(raw) >= 5:
    return raw[:4].upper() + "." + raw[4:]
  return raw.upper()

def get_category_idx(ss, sw):
  mapping = {'a': 0, 'b': 1, 'c': 2}
  return mapping[ss] * 3 + mapping[sw]

def main():
  rng = np.random.default_rng(42)
  # 1. Load data
  labels_df = pd.read_csv("dataset/residue_switch_labels.csv")
  meta_df = pd.read_csv("dataset/filtered_metamorphic_pairs.csv")
  
  # 2. Group PDBs using connected components (to ensure pairs stay together)
  G = nx.Graph()
  for pdb in labels_df['Raw_Input_ID'].unique():
    G.add_node(pdb)
    
  for _, row in meta_df.iterrows():
    p1 = format_id(row['PDB1'])
    p2 = format_id(row['PDB2'])
    if p1 in G and p2 in G:
      G.add_edge(p1, p2)
      
  groups = list(nx.connected_components(G))
  
  # Precompute fold-switch counts for each group
  # There are 9 categories: (a,a), (a,b), (a,c), (b,a), (b,b), (b,c), (c,a), (c,b), (c,c)
  group_counts = np.zeros((len(groups), 9))
  
  # Compute counts for each PDB
  pdb_counts = {}
  for pdb, df in labels_df.groupby('Raw_Input_ID'):
    counts = np.zeros(9)
    for _, row in df.iterrows():
      counts[get_category_idx(row['SS_Label'], row['Switch_Label'])] += 1
    pdb_counts[pdb] = counts
    
  for i, group in enumerate(groups):
    for pdb in group:
      group_counts[i] += pdb_counts[pdb]
      
  # 3. Find the best split
  best_js = float('inf')
  best_split = None
  
  n_groups = len(groups)
  total_structures = len(G.nodes)
  target_trainval_size = int(0.8 * total_structures)
  target_val_size = int(0.1 * target_trainval_size)
  
  print("Sampling 10000 random splits to minimize JS divergence across Train, Val, and Test...")
  for _ in tqdm(range(10000)):
    indices = rng.permutation(n_groups)
    
    train_indices = []
    val_indices = []
    test_indices = []
    current_trainval = 0
    current_val = 0
    
    for idx in indices:
      group_size = len(groups[idx])
      if current_trainval < target_trainval_size:
        if current_val < target_val_size:
          val_indices.append(idx)
          current_val += group_size
        else:
          train_indices.append(idx)
        current_trainval += group_size
      else:
        test_indices.append(idx)
        
    train_counts = np.sum(group_counts[train_indices], axis=0) if train_indices else np.zeros(9)
    val_counts = np.sum(group_counts[val_indices], axis=0) if val_indices else np.zeros(9)
    test_counts = np.sum(group_counts[test_indices], axis=0) if test_indices else np.zeros(9)
    
    train_dist = train_counts / np.sum(train_counts) if np.sum(train_counts) > 0 else np.ones(9)/9
    val_dist = val_counts / np.sum(val_counts) if np.sum(val_counts) > 0 else np.ones(9)/9
    test_dist = test_counts / np.sum(test_counts) if np.sum(test_counts) > 0 else np.ones(9)/9
    
    js = jensenshannon(train_dist, test_dist) + jensenshannon(train_dist, val_dist)
    
    if js < best_js:
      best_js = js
      best_split = (train_indices, val_indices, test_indices)
      
  print(f"Best JS Divergence (Train-Test + Train-Val): {best_js:.4f}")
  
  train_group_indices, val_group_indices, test_group_indices = best_split
  
  train_pdbs = set()
  for idx in train_group_indices:
    train_pdbs.update(groups[idx])
    
  val_pdbs = set()
  for idx in val_group_indices:
    val_pdbs.update(groups[idx])
    
  test_pdbs = set()
  for idx in test_group_indices:
    test_pdbs.update(groups[idx])
    
  # 4. Assign split labels
  split_col = np.empty(len(labels_df), dtype=object)
  
  test_mask = labels_df['Raw_Input_ID'].isin(test_pdbs)
  val_mask = labels_df['Raw_Input_ID'].isin(val_pdbs)
  
  for i in range(len(labels_df)):
    if test_mask[i]:
      split_col[i] = 'test'
    elif val_mask[i]:
      split_col[i] = 'val'
    else:
      split_col[i] = 'train'
      
  labels_df['split'] = split_col
  
  labels_df.to_csv("dataset/data_split.csv", index=False)
  print(f"Train residues: {np.sum(labels_df['split'] == 'train')}")
  print(f"Val residues: {np.sum(labels_df['split'] == 'val')}")
  print(f"Test residues: {np.sum(labels_df['split'] == 'test')}")
  print("Saved data split to dataset/data_split.csv")

if __name__ == "__main__":
  main()
