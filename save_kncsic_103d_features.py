import pandas as pd
import numpy as np
import pickle
import os
from tqdm.auto import tqdm
import biotite.structure as struc
import biotite.structure.io as strucio
from scipy.spatial.distance import cdist

def compute_103d_features(csv_file_path, cif_dir):
  """
  Computes the 103-dimensional kNCSIC feature vector for each residue.
  - 20D: One-hot encoded amino acid
  - 80D: Contact amino acid frequencies in 4 sequence-separation ranges
  - 3D: One-hot encoded secondary structure element (SSE)
  """
  aa_list = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
  ]
  ss_list = ['a', 'b', 'c']
  
  df = pd.read_csv(csv_file_path)
  
  feature_dict = {}
  grouped = df.groupby('Raw_Input_ID')
  
  for raw_id, group in tqdm(grouped, desc="Computing 103D Features", total=len(grouped)):
    cif_path = os.path.join(cif_dir, f"{raw_id}.cif")
    try:
      array = strucio.load_structure(cif_path)
      array = array[struc.filter_amino_acids(array)]
      if "altloc_id" in array.get_annotation_categories():
        array = array[(array.altloc_id == '') | (array.altloc_id == 'A') | (array.altloc_id == '1')]
      
      df_atoms = pd.DataFrame({
        'res_id': array.res_id,
        'atom_name': array.atom_name,
        'element': array.element,
        'x': array.coord[:,0],
        'y': array.coord[:,1],
        'z': array.coord[:,2]
      })
    except Exception as e:
      df_atoms = pd.DataFrame(columns=['res_id', 'atom_name', 'element', 'x', 'y', 'z'])
      
    n_res = len(group)
    seq_ids = group['Residue_ID'].values
    res_names = group['Residue_Name'].values
    ss_labels = group['SS_Label'].values
    
    # 1. Extract centre of mass coordinates for each residue
    com_coords = np.full((n_res, 3), np.nan)
    
    # Standard atomic masses
    mass_map = {'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.065, 'H': 1.008}
    
    for i, res_id in enumerate(seq_ids):
      atoms = df_atoms[df_atoms['res_id'] == res_id]
      if len(atoms) > 0:
        coords = atoms[['x', 'y', 'z']].values
        elements = atoms['element'].values
        
        # Determine masses for each atom, defaulting to 12.0 (Carbon) if unknown
        masses = np.array([mass_map.get(el.upper(), 12.0) for el in elements]).reshape(-1, 1)
        
        # Calculate the centre of mass
        com = np.sum(coords * masses, axis=0) / np.sum(masses)
        com_coords[i] = com
        
    # 2. Compute pairwise distances (fill NaNs with inf to avoid warnings and false positives)
    com_clean = np.where(np.isnan(com_coords), np.inf, com_coords)
    
    dist_com = cdist(com_clean, com_clean)
    
    # Contact map: distance between centres of mass < 8A
    contact_map = (dist_com < 8.0)
    
    # 3. Precompute AA and SS one-hot encodings
    aa_one_hot = np.zeros((n_res, 20))
    for i, name in enumerate(res_names):
      if name in aa_list:
        aa_one_hot[i, aa_list.index(name)] = 1.0
        
    ss_one_hot = np.zeros((n_res, 3))
    for i, ss in enumerate(ss_labels):
      if ss in ss_list:
        ss_one_hot[i, ss_list.index(ss)] = 1.0
        
    # 4. Compute the 103D vector for each residue
    for i in range(n_res):
      # Sequence separation based on index position in the sequence array
      sep = np.abs(np.arange(n_res) - i)
      
      mask_cat1 = (sep >= 2) & (sep <= 5)
      mask_cat2 = (sep >= 6) & (sep <= 11)
      mask_cat3 = (sep >= 12) & (sep <= 23)
      mask_cat4 = (sep >= 24)
      
      c1 = contact_map[i] & mask_cat1
      c2 = contact_map[i] & mask_cat2
      c3 = contact_map[i] & mask_cat3
      c4 = contact_map[i] & mask_cat4
      
      counts1 = np.sum(aa_one_hot[c1], axis=0) if np.any(c1) else np.zeros(20)
      counts2 = np.sum(aa_one_hot[c2], axis=0) if np.any(c2) else np.zeros(20)
      counts3 = np.sum(aa_one_hot[c3], axis=0) if np.any(c3) else np.zeros(20)
      counts4 = np.sum(aa_one_hot[c4], axis=0) if np.any(c4) else np.zeros(20)
      
      # Concatenate: AA (20) + Contacts (80) + SSE (3) = 103D
      vec_103d = np.concatenate([aa_one_hot[i], counts1, counts2, counts3, counts4, ss_one_hot[i]])
      
      key = (raw_id, seq_ids[i])
      feature_dict[key] = vec_103d
      
  return feature_dict

if __name__ == "__main__":
  data_file_path = 'residue_dihedral_rotational_freedom.csv'
  cif_dir = 'fixed_cif_files'
  
  feature_dict = compute_103d_features(data_file_path, cif_dir)
  
  save_dir = 'saved_data/feature_extraction'
  os.makedirs(save_dir, exist_ok=True)
  
  save_file_path = os.path.join(save_dir, 'kncsic_103d_features.pkl')
  with open(save_file_path, 'wb') as f:
    pickle.dump(feature_dict, f)
  
  print(f"\nSuccessfully saved {len(feature_dict)} 103D features to {save_file_path}")
