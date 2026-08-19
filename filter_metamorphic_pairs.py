import os
import pandas as pd
import numpy as np
import biotite.sequence as seq
import biotite.sequence.align as align
import biotite.structure.io as strucio
from tqdm.auto import tqdm

def format_id(raw):
  # '1ceeB' -> '1CEE.B'
  if len(raw) >= 5:
    return raw[:4].upper() + "." + raw[4:]
  return raw.upper()

def get_structure_ca_coords(pdb_id_chain, res_indices):
  # Returns the CA coordinates for the given 0-based residue indices
  pdb_id, chain_id = pdb_id_chain.split('.')
  cif_path = f"dataset/fixed_cif_files/{pdb_id_chain}.cif"
  if not os.path.exists(cif_path):
    cif_path = f"dataset/cif_files/{pdb_id}.cif"
  if not os.path.exists(cif_path):
    return None
  
  try:
    array = strucio.load_structure(cif_path, model=1)
    # Filter to target chain and CA atoms
    chain_array = array[(array.chain_id == chain_id) & (array.atom_name == "CA")]
    
    # We need to map 0-based sequence indices to the structure.
    # Since the sequence in pdb_secondary_structures comes from the structure,
    # we assume the CA atoms are in the same order as the sequence.
    coords = []
    for idx in res_indices:
      if idx < len(chain_array):
        coords.append(chain_array.coord[idx])
    return np.array(coords)
  except Exception as e:
    print(f"Error loading {cif_path}: {e}")
    return None

def main():
  pairs_df = pd.read_csv("dataset/metamorphic_pairs.csv")
  ss_df = pd.read_csv("dataset/pdb_secondary_structures.csv")
  ss_dict = {row['Raw_Input_ID']: row['Secondary_Structure'] for _, row in ss_df.iterrows()}
  seq_dict = {row['Raw_Input_ID']: row['Sequence'] for _, row in ss_df.iterrows()}
  
  matrix = align.SubstitutionMatrix.std_protein_matrix()
  
  valid_pairs = []
  output_rows = []
  
  for idx, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
    p1 = format_id(row['PDB1'])
    p2 = format_id(row['PDB2'])
    
    if p1 not in seq_dict or p2 not in seq_dict:
      print(f"Skipping {p1}-{p2}: Sequence not found.")
      continue
      
    s1 = seq.ProteinSequence(seq_dict[p1])
    s2 = seq.ProteinSequence(seq_dict[p2])
    ss1 = ss_dict[p1]
    ss2 = ss_dict[p2]
    
    # 1. Align
    alignments = align.align_optimal(s1, s2, matrix, local=True)
    if not alignments:
      continue
    best_aln = alignments[0]
    
    seq_id = align.get_sequence_identity(best_aln)
    if seq_id < 0.8:
      continue
      
    # 2. Residue switch labels
    trace = best_aln.trace
    aligned_pos1 = trace[ (trace[:, 0] != -1) & (trace[:, 1] != -1), 0 ]
    aligned_pos2 = trace[ (trace[:, 0] != -1) & (trace[:, 1] != -1), 1 ]
    
    fs_pos1 = []
    fs_pos2 = []
    
    for p1_idx, p2_idx in zip(aligned_pos1, aligned_pos2):
      if ss1[p1_idx] != ss2[p2_idx]:
        fs_pos1.append(p1_idx)
        fs_pos2.append(p2_idx)
        
    # 3. Minimum contiguous stretch of at least 4
    def has_contiguous_stretch(positions, min_len=4):
      if not positions:
        return False
      positions = sorted(positions)
      stretch = 1
      max_stretch = 1
      for i in range(1, len(positions)):
        if positions[i] == positions[i-1] + 1:
          stretch += 1
          max_stretch = max(max_stretch, stretch)
        else:
          stretch = 1
      return max_stretch >= min_len
      
    if not has_contiguous_stretch(fs_pos1, 4):
      continue
      
    # 4. Unaligned parts and contacts
    def check_unaligned_contacts(p_id, seq_len, aligned_pos, fs_pos):
      unaligned_len = seq_len - len(aligned_pos)
      if unaligned_len / seq_len > 0.25:
        unaligned_indices = set(range(seq_len)) - set(aligned_pos)
        unaligned_coords = get_structure_ca_coords(p_id, list(unaligned_indices))
        fs_coords = get_structure_ca_coords(p_id, fs_pos)
        
        if unaligned_coords is not None and fs_coords is not None and len(unaligned_coords) > 0 and len(fs_coords) > 0:
          # Compute pairwise distances
          from scipy.spatial.distance import cdist
          dists = cdist(unaligned_coords, fs_coords)
          if np.min(dists) < 8.0:
            return True # Has contact
      return False
      
    if check_unaligned_contacts(p1, len(s1), aligned_pos1, fs_pos1):
      continue
    if check_unaligned_contacts(p2, len(s2), aligned_pos2, fs_pos2):
      continue
      
    # Passed all filters
    valid_pairs.append((row['PDB1'], row['PDB2']))
    
    # Save detailed position-wise data
    for p1_idx, p2_idx in zip(aligned_pos1, aligned_pos2):
      output_rows.append({
        'PDB1_Formatted': p1,
        'PDB2_Formatted': p2,
        'Residue_ID_1': p1_idx + 1,  # 1-indexed
        'Residue_ID_2': p2_idx + 1,
        'SS_Label_1': ss1[p1_idx],
        'SS_Label_2': ss2[p2_idx],
        'Switch_Label_1': ss2[p2_idx],
        'Switch_Label_2': ss1[p1_idx]
      })
      
  print(f"Total valid pairs: {len(valid_pairs)} / {len(pairs_df)}")
  
  # Save the resulting data
  out_df = pd.DataFrame(output_rows)
  out_df.to_csv("dataset/filtered_metamorphic_aligned_positions.csv", index=False)
  
  # Also save the filtered pairs
  valid_pairs_df = pd.DataFrame(valid_pairs, columns=['PDB1', 'PDB2'])
  valid_pairs_df.to_csv("dataset/filtered_metamorphic_pairs.csv", index=False)
  print("Saved to dataset/filtered_metamorphic_aligned_positions.csv and dataset/filtered_metamorphic_pairs.csv")

if __name__ == "__main__":
  main()
