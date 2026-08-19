import os
import re
import numpy as np
import pandas as pd
import biotite.database.rcsb as rcsb
import biotite.structure as struc
import biotite.structure.io as strucio
from tqdm.auto import tqdm

def _get_dssp_3state(array, use_dssp=True):
  """
  Calculates secondary structure using the official DSSP executable
  and reduces the 8-state classification down to a 3-state output (a, b, c).
  """
  if use_dssp:

    from biotite.application.dssp import DsspApp
    import sys
    import shutil
    
    # 1. Search for the executable name
    dssp_bin = shutil.which("mkdssp") or shutil.which("dssp")
    
    # 2. Gentle Conda hunt if standard PATH fails (e.g. inside Jupyter)
    if not dssp_bin:
      conda_bin_dir = os.path.dirname(sys.executable)
      if os.path.exists(os.path.join(conda_bin_dir, "mkdssp")):
        dssp_bin = os.path.join(conda_bin_dir, "mkdssp")
      elif os.path.exists(os.path.join(conda_bin_dir, "dssp")):
        dssp_bin = os.path.join(conda_bin_dir, "dssp")
        
    if not dssp_bin:
      raise FileNotFoundError("Neither 'mkdssp' nor 'dssp' found in PATH.")

    # Initialize and run DSSP application using the resolved binary
    app = DsspApp(array, bin_path=dssp_bin)
    app.start()
    app.join()
    
    # Check for execution errors (Allow both FINISHED and JOINED states)
    state = app.get_app_state()
    if "FINISHED" not in str(state) and "JOINED" not in str(state):
       raise RuntimeError(f"DSSP execution failed with state: {state}")
       
    sse_8state = app.get_sse()
    
    # Map 8-state to 3-state
    sse_3state = []
    for state in sse_8state:
      if state in ['H', 'G', 'I']:
        sse_3state.append('a')
      elif state in ['E', 'B']:
        sse_3state.append('b')
      else:
        sse_3state.append('c')
        
    return np.array(sse_3state)
  else:
    return struc.annotate_sse(array)

def save_secondary_structures(input_csv, cif_dir="dataset/fixed_cif_files", output_csv="dataset/residue_secondary_structures.csv"):
  # Read the training data
  df = pd.read_csv(input_csv)
  raw_ids = df['PDB_ID'].unique()
  results = []
  
  print(f"Processing {len(raw_ids)} unique structures...")
  
  for raw_id in raw_ids:
      
    # Local File Search: prioritize the 'cif_files' directory
    local_path = os.path.join(cif_dir, f"{raw_id}.cif")
    if os.path.exists(local_path):
      source = local_path
    else:
      # Fallback: Fetch BinaryCIF from RCSB
      tqdm.write('#'*50+f"\n{raw_id} is missing in {cif_dir}\n"+'#'*50)
      continue
    
    # Load structure with Entity ID metadata (extra_fields is required for _ IDs)
    array = strucio.load_structure(source)
        
    # Filter for protein atoms only
    array = array[struc.filter_amino_acids(array)]

    # Filter out insertion codes (keep only the standard empty string "")
    if "ins_code" in array.get_annotation_categories():
      array = array[array.ins_code == ""]
    
    if len(array) == 0:
      continue
      
    # Calculate Secondary Structure (a=alpha, b=beta, c=coil)
    sse = _get_dssp_3state(array, use_dssp=True)
    
    # Extract residue-level information
    starts = struc.get_residue_starts(array)
    res_ids = array.res_id[starts]
    res_names = array.res_name[starts]
    seen = set()
    for r_id, r_name, ss in zip(res_ids, res_names, sse):
      if r_id in seen:
        print(f"{raw_id} has mutiple residues at {r_id}.")
        continue
      seen.add(r_id)
      results.append({
        "Raw_Input_ID": raw_id,
        "Residue_ID": r_id,
        "Residue_Name": r_name,
        "SS_Label": ss
      })
        
    # print(f"Failed to process {raw_id}: {e}")
      
  # Export final results
  out_df = pd.DataFrame(results)
  out_df.to_csv(output_csv, index=False)
  print(f"\nFinished! Results saved to '{output_csv}'")


# Standard mapping for amino acid codes
AA_MAP = {
  'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
  'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
  'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
  'TYR': 'Y', 'VAL': 'V'
}

def transform_to_sequence_format(input_file="dataset/residue_secondary_structures.csv", output_file="dataset/pdb_secondary_structures.csv"):
  # Load the residue-level data
  df = pd.read_csv(input_file)
  
  # Convert 3-letter codes to 1-letter symbols
  # Ensure the name is a string before calling .upper() to avoid float errors
  df['AA_1'] = df['Residue_Name'].apply(lambda x: AA_MAP.get(str(x).upper(), 'X'))
  
  # Ensure the data is sorted by Residue_ID to maintain sequence order
  df = df.sort_values(by=['Raw_Input_ID', 'Residue_ID'])
  
  # Helper function to join values while ignoring NaNs and forcing string type
  def join_labels(series):
    # Filter out NaNs, convert to string, and join
    return ''.join(series.dropna().astype(str))

  # Aggregate the residues into strings
  transformed = df.groupby('Raw_Input_ID').agg({
    'AA_1': join_labels,
    'SS_Label': join_labels,
  }).reset_index()
  
  # Rename for clarity
  transformed = transformed.rename(columns={
    'AA_1': 'Sequence',
    'SS_Label': 'Secondary_Structure'
  })
  
  # Reorder columns
  cols = ['Raw_Input_ID', 'Sequence', 'Secondary_Structure']
  transformed = transformed[cols]
  
  transformed.to_csv(output_file, index=False)
  print(f"Transformation complete. Data saved to '{output_file}'")



if __name__ == "__main__":
  save_secondary_structures("dataset/training_data.csv")
  transform_to_sequence_format()