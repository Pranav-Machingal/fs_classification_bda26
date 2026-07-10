import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
import tempfile
from tqdm.auto import tqdm
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
from transformers import EsmTokenizer, EsmModel

sys.path.append("SaProt")
from utils.foldseek_util import get_struc_seq

FOLDSEEK_BIN = "foldseek/bin/foldseek"
SAPROT_MODEL_PATH = "saved_data/saprot_model/SaProt_650M_AF2" 
SEQUENCE_LENGTH_LIMIT = 1300 # should be even

def _extract_combined_seq_and_ids(raw_id, cif_dir, required_res_ids):
  """
  Reads the CIF file, extracts the specific chain, truncates if needed, 
  saves to a temp PDB, and extracts the structure-aware combined sequence using Foldseek.
  """
  source = os.path.join(cif_dir, f"{raw_id}.cif")
  try:
    array = strucio.load_structure(source)
  except Exception as e:
    print(f"\nError reading {source}: {e}")
    return None, None

  array = array[struc.filter_amino_acids(array)]
  if "altloc_id" in array.get_annotation_categories():
    array = array[(array.altloc_id == '') | (array.altloc_id == 'A') | (array.altloc_id == '1')]
    
  if len(array) == 0:
    return None, None

  starts = struc.get_residue_starts(array)
  if len(starts) > SEQUENCE_LENGTH_LIMIT:
    real_res_ids = array.res_id[starts]
    req_set = set(required_res_ids)
    req_indices = [i for i, r in enumerate(real_res_ids) if r in req_set]
    
    if req_indices:
      min_idx, max_idx = min(req_indices), max(req_indices)
      if max_idx - min_idx >= SEQUENCE_LENGTH_LIMIT:
        max_idx = min_idx + SEQUENCE_LENGTH_LIMIT-1
        
      center_idx = (min_idx + max_idx) // 2
      start_idx = max(0, center_idx - int(SEQUENCE_LENGTH_LIMIT/2))
      end_idx = min(len(starts), start_idx + SEQUENCE_LENGTH_LIMIT)
      
      if end_idx - start_idx < SEQUENCE_LENGTH_LIMIT and start_idx > 0:
        start_idx = max(0, end_idx - SEQUENCE_LENGTH_LIMIT)
        
      first_atom_idx = starts[start_idx]
      last_atom_idx = starts[end_idx] if end_idx < len(starts) else len(array)
      array = array[first_atom_idx:last_atom_idx]
      starts = struc.get_residue_starts(array)

  real_res_ids = array.res_id[starts]
  array.bonds = None 

  fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
  os.close(fd)
  try:
    strucio.save_structure(tmp_path, array)
    seq_dict = get_struc_seq(FOLDSEEK_BIN, tmp_path, plddt_mask=False)
    if not seq_dict:
      return None, None
    # Usually only one chain in the dictionary because of our isolation/truncation
    chain_id = list(seq_dict.keys())[0]
    _, _, combined_seq = seq_dict[chain_id]
  finally:
    if os.path.exists(tmp_path):
      os.remove(tmp_path)
      
  return combined_seq, real_res_ids


def _get_saprot_embeddings(formatted_seq, model, tokenizer, device):
  """
  Tokenizes the structure-aware combined sequence and extracts the hidden states.
  """
  inputs = tokenizer(formatted_seq, return_tensors="pt", add_special_tokens=True).to(device)
  
  with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.last_hidden_state
    
  # Strip the <cls> token at index 0 and <eos> token at the end
  embeddings = hidden_states[0, 1:-1, :].cpu().numpy()
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
    combined_seq, real_res_ids = _extract_combined_seq_and_ids(raw_input_id, cif_dir, required_res_ids)
    
    if combined_seq is None or len(combined_seq) == 0:
      print(f"\nWarning: Valid structure/sequence not found for {raw_input_id}. Skipping.")
      continue
      
    try:
      # Generate structure-aware embeddings using combined sequence
      embeddings = _get_saprot_embeddings(combined_seq, model, tokenizer, device)
    except Exception as e:
      print(f"\nError processing {raw_input_id}: {e}")
      continue 
    
    res_to_embed = {res_id: emb for res_id, emb in zip(real_res_ids, embeddings)}
    
    for res_id in required_res_ids:
      if res_id in res_to_embed:
        features_dict[(raw_input_id, res_id)] = res_to_embed[res_id]
      else:
        print(f"\nWarning: Residue ID {res_id} not found in parsed embeddings for {raw_input_id}")

  save_dir = Path('saved_data/feature_extraction')
  save_dir.mkdir(parents=True, exist_ok=True)
  
  save_file_path = save_dir / 'saprot_features.pkl'
  with open(save_file_path, 'wb') as f:
        pickle.dump(features_dict, f)
    
  print(f"\nSuccessfully saved {len(features_dict)} embeddings to {save_file_path}")

if __name__ == "__main__":
  save_saprot_embeddings()