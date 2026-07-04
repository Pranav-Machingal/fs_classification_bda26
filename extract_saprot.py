import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
from transformers import EsmTokenizer, EsmModel

# Standard mapping for 3-letter to 1-letter amino acid codes
AA_MAP = {
  'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
  'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
  'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
  'TYR': 'Y', 'VAL': 'V'
}


# Change this to "./local_saprot_model" if you ran the download script
SAPROT_MODEL_PATH = "saved_data/saprot_model/SaProt_650M_AF2" 
SEQUENCE_LENGTH_LIMIT = 1300 # should be even
def _extract_sequence_and_ids(raw_id, cif_dir, required_res_ids):
  """
  Reads the CIF file, extracts the specific chain, resolves the exact 
  residue IDs, safely truncates if >SEQUENCE_LENGTH_LIMIT residues, and converts to a 1D sequence.
  """
  if "_" in raw_id:
    pdb_id, entity_id = raw_id.split("_", 1)
    chain_id = None
  elif "." in raw_id:
    pdb_id, chain_id = raw_id.split(".", 1)
    entity_id = None
  else:
    pdb_id = raw_id
    chain_id = None
    entity_id = None

  # 1. load structure file
  source = os.path.join(cif_dir, f"{raw_id}.cif")
  
  try:
    array = strucio.load_structure(source)
  except Exception as e:
    print(f"\nError reading {source}: {e}")
    return None, None

  array = array[struc.filter_amino_acids(array)]
  
  # 2. Remove Alternate Locations
  if "altloc_id" in array.get_annotation_categories():
    array = array[(array.altloc_id == '') | (array.altloc_id == 'A') | (array.altloc_id == '1')]
    
  if len(array) == 0:
    return None, None

  # 3. TRUNCATE: Prevent ESM Tensor Size Crashes (>SEQUENCE_LENGTH_LIMIT residues)
  starts = struc.get_residue_starts(array)
  if len(starts) > SEQUENCE_LENGTH_LIMIT:
    real_res_ids = array.res_id[starts]
    req_set = set(required_res_ids)
    req_indices = [i for i, r in enumerate(real_res_ids) if r in req_set]
    
    if req_indices:
      min_idx, max_idx = min(req_indices), max(req_indices)
      if max_idx - min_idx >= SEQUENCE_LENGTH_LIMIT:
        max_idx = min_idx + SEQUENCE_LENGTH_LIMIT-1  # Clamp if targets are unbelievably far apart
        
      center_idx = (min_idx + max_idx) // 2
      start_idx = max(0, center_idx - SEQUENCE_LENGTH_LIMIT/2)
      end_idx = min(len(starts), start_idx + SEQUENCE_LENGTH_LIMIT)
      
      if end_idx - start_idx < SEQUENCE_LENGTH_LIMIT and start_idx > 0:
        start_idx = max(0, end_idx - SEQUENCE_LENGTH_LIMIT)
        
      # Slice the array perfectly to preserve continuity
      first_atom_idx = starts[start_idx]
      last_atom_idx = starts[end_idx] if end_idx < len(starts) else len(array)
      array = array[first_atom_idx:last_atom_idx]
      starts = struc.get_residue_starts(array)

  # 4. Extract 1-letter Sequence and exact Residue IDs
  real_res_ids = array.res_id[starts]
  res_names = array.res_name[starts]
  
  sequence = "".join([AA_MAP.get(name, 'X') for name in res_names])
  return sequence, real_res_ids


def _get_saprot_embeddings(sequence, model, tokenizer, device):
  """
  Formats the sequence with structure masks ('#'), tokenizes it, 
  and extracts the hidden states from the SaProt model.
  """
  # Format sequence to SaProt's structure-agnostic vocabulary (e.g., M -> M#)
  formatted_seq = "".join([aa + "#" for aa in sequence])
  
  inputs = tokenizer(formatted_seq, return_tensors="pt", add_special_tokens=True).to(device)
  
  with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # Extract the last hidden state 
    hidden_states = outputs.last_hidden_state
    
  # Strip the <cls> token at index 0 and <eos> token at the end
  # The output shape becomes (sequence_length, embedding_dim)
  embeddings = hidden_states[0, 1:len(sequence)+1, :].cpu().numpy()
  
  return embeddings


def save_saprot_embeddings():
  labels_df = pd.read_csv('residue_switch_labels.csv')
  cif_dir = 'fixed_cif_files'
  grouped = labels_df.groupby('Raw_Input_ID')
  features_dict = {}
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  print(f"Loading SaProt Model from {SAPROT_MODEL_PATH}...")
  try:
    tokenizer = EsmTokenizer.from_pretrained(SAPROT_MODEL_PATH)
    model = EsmModel.from_pretrained(SAPROT_MODEL_PATH).to(device)
    model.eval()
  except Exception as e:
    print(f"Error loading model: {e}")
    print("If you haven't downloaded it yet, run download_saprot.py or ensure internet access.")
    return

  for raw_input_id, group in tqdm(grouped, desc="Extracting SaProt Embeddings"):
    required_res_ids = group['Residue_ID'].unique()
    sequence, real_res_ids = _extract_sequence_and_ids(raw_input_id, cif_dir, required_res_ids)
    
    if sequence is None or len(sequence) == 0:
      print(f"\nWarning: Valid sequence not found for {raw_input_id}. Skipping.")
      continue
      
    try:
      # Generate structure-agnostic embeddings
      embeddings = _get_saprot_embeddings(sequence, model, tokenizer, device)
    except Exception as e:
      print(f"\nError processing {raw_input_id}: {e}")
      continue 
    
    # Map residue IDs to their corresponding embedding vector
    res_to_embed = {res_id: emb for res_id, emb in zip(real_res_ids, embeddings)}
    
    for res_id in required_res_ids:
      if res_id in res_to_embed:
        features_dict[(raw_input_id, res_id)] = res_to_embed[res_id]
      else:
        print(f"\nWarning: Residue ID {res_id} not found in parsed embeddings for {raw_input_id}")

  # Save the Dictionary
  save_dir = Path('saved_data/feature_extraction')
  save_dir.mkdir(parents=True, exist_ok=True)
  
  save_file_path = save_dir / 'saprot_features.pkl'
  with open(save_file_path, 'wb') as f:
    pickle.dump(features_dict, f)
    
  print(f"\nSuccessfully saved {len(features_dict)} embeddings to {save_file_path}")


if __name__ == "__main__":
  save_saprot_embeddings()