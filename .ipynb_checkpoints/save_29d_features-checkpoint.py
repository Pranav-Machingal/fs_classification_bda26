import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm

def compute_29d_features(csv_file_path):
  """
  Reads a CSV of protein structural features and computes a 29-dimensional 
  vector for each residue position.
  """
  # Standard definitions for one-hot encoding
  aa_list = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
  ]
  ss_list = ['a', 'b', 'c'] # helix, sheet, coil/chain
  
  # Load the dataset
  df = pd.read_csv(csv_file_path)
  
  # Define the 6 dihedral columns
  dihedral_cols = ['phi_min', 'phi', 'phi_max', 'psi_min', 'psi', 'psi_max']
  
  # Handle missing values (e.g., extreme N/C terminus missing phi or psi)
  # Fills missing values with 0.0 to maintain vector dimensions
  df[dihedral_cols] = df[dihedral_cols].fillna(0.0)
  
  # Dictionary to store the parsed 29-D vectors
  # Format -> {(Raw_Input_ID, Residue_ID): numpy array of shape (29,)}
  feature_dict = {}
  
  for _, row in tqdm(df.iterrows(), total=len(df)):
    # 1. Encode Amino Acid (20 dimensions)
    aa_vec = np.zeros(20)
    if row['Residue_Name'] in aa_list:
      aa_vec[aa_list.index(row['Residue_Name'])] = 1.0
      
    # 2. Encode Dihedral Rotational Freedom (6 dimensions)
    dihedral_vec = row[dihedral_cols].values.astype(float)
    
    # 3. Encode Secondary Structure Element (3 dimensions)
    ss_vec = np.zeros(3)
    if row['SS_Label'] in ss_list:
      ss_vec[ss_list.index(row['SS_Label'])] = 1.0
      
    # 4. Concatenate to form the 29-dimensional vector
    vector_29d = np.concatenate([aa_vec, dihedral_vec/180, ss_vec])
    
    # Store the result using the unique identifiers as the key
    key = (row['Raw_Input_ID'], row['Residue_ID'])
    feature_dict[key] = vector_29d
    
  return feature_dict


if __name__ == "__main__":
  data_file_path = 'residue_dihedral_rotational_freedom.csv'
  
  feature_dict = compute_29d_features(data_file_path)

  save_file_path = 'saved_data/feature_extraction/29d_features.pkl'
  with open(save_file_path, 'wb') as f:
    pickle.dump(feature_dict, f)