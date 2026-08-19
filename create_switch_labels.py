import pandas as pd
from tqdm.auto import tqdm

def format_id(raw):
  if len(raw) >= 5:
    return raw[:4].upper() + "." + raw[4:]
  return raw.upper()

def main():
  # 1. Determine which PDB IDs are metamorphic
  meta_df = pd.read_csv("dataset/metamorphic_pairs.csv")
  meta_ids = set()
  for _, row in meta_df.iterrows():
    meta_ids.add(format_id(row['PDB1']))
    meta_ids.add(format_id(row['PDB2']))
    
  # 2. Load aligned positions for metamorphic structures
  aligned_df = pd.read_csv("dataset/filtered_metamorphic_aligned_positions.csv")
  switch_dict = {}
  for _, row in aligned_df.iterrows():
    p1 = row['PDB1_Formatted']
    p2 = row['PDB2_Formatted']
    r1 = int(row['Residue_ID_1'])
    r2 = int(row['Residue_ID_2'])
    sl1 = row['Switch_Label_1']
    sl2 = row['Switch_Label_2']
    
    if p1 not in switch_dict:
      switch_dict[p1] = {}
    if p2 not in switch_dict:
      switch_dict[p2] = {}
      
    switch_dict[p1][r1] = sl1
    switch_dict[p2][r2] = sl2
    
  # 3. Load all residue secondary structures
  res_ss_df = pd.read_csv("dataset/residue_secondary_structures.csv")
  
  # 4. Load training data to get the universe of PDB IDs we care about
  train_df = pd.read_csv("dataset/training_data.csv")
  train_ids = set(train_df['PDB_ID'])
  
  output_rows = []
  
  # Group the residue SS data by PDB_ID for faster lookup
  grouped = res_ss_df.groupby('Raw_Input_ID')
  
  print("Generating residue switch labels...")
  for pdb_id in tqdm(train_ids):
    if pdb_id not in grouped.groups:
      print(f"Warning: {pdb_id} not found in residue_secondary_structures.csv")
      continue
      
    pdb_res_df = grouped.get_group(pdb_id)
    is_meta = pdb_id in meta_ids
    
    for _, row in pdb_res_df.iterrows():
      res_id = int(row['Residue_ID'])
      
      if is_meta:
        # For metamorphic, only include if it was aligned (and successfully passed filters)
        if pdb_id in switch_dict and res_id in switch_dict[pdb_id]:
          output_rows.append({
            'Raw_Input_ID': pdb_id,
            'Residue_ID': res_id,
            'Residue_Name': row['Residue_Name'],
            'SS_Label': row['SS_Label'],
            'Switch_Label': switch_dict[pdb_id][res_id]
          })
      else:
        # For monomorphic, include all positions, and Switch_Label == SS_Label
        output_rows.append({
          'Raw_Input_ID': pdb_id,
          'Residue_ID': res_id,
          'Residue_Name': row['Residue_Name'],
          'SS_Label': row['SS_Label'],
          'Switch_Label': row['SS_Label']
        })
        
  out_df = pd.DataFrame(output_rows)
  out_df.to_csv("dataset/residue_switch_labels.csv", index=False)
  print(f"Saved {len(out_df)} residue entries to dataset/residue_switch_labels.csv")

if __name__ == "__main__":
  main()
