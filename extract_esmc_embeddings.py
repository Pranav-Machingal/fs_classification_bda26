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

# Standard mapping for 3-letter to 1-letter amino acid codes
AA_MAP = {
  'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q',
  'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
  'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W',
  'TYR': 'Y', 'VAL': 'V'
}

def _extract_sequence_and_ids(raw_id, cif_dir, required_res_ids):
  """
  Reads the CIF file, extracts the specific chain, resolves the exact 
  residue IDs, and converts it to a 1D amino acid sequence for ESM-C.
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

  # 1. Robust Case-Insensitive File Resolution
  source = os.path.join(cif_dir, f"{pdb_id}.cif")
  if not os.path.exists(source):
    source = os.path.join(cif_dir, f"{pdb_id.lower()}.cif")
    if not os.path.exists(source):
      source = os.path.join(cif_dir, f"{pdb_id.upper()}.cif")
      if not os.path.exists(source):
        return None, None

  extra = ["label_entity_id", "auth_asym_id"]
  try:
    array = strucio.load_structure(source, model=1, extra_fields=extra)
  except Exception as e:
    print(f"\nError reading {source}: {e}")
    return None, None
  
  if entity_id:
    if "label_entity_id" in array.get_annotation_categories():
      entity_mask = (array.label_entity_id == str(entity_id))
      if not np.any(entity_mask):
        return None, None
      chain_id_to_keep = np.unique(array.chain_id[entity_mask])[0]
      array = array[array.chain_id == chain_id_to_keep]
    else:
      return None, None
      
  elif chain_id:
    if "auth_asym_id" in array.get_annotation_categories():
      chain_mask = (array.auth_asym_id == chain_id) | (array.chain_id == chain_id)
    else:
      chain_mask = (array.chain_id == chain_id)
    array = array[chain_mask]

  array = array[struc.filter_amino_acids(array)]
  
  # 2. Remove Alternate Locations to prevent duplicate residue mappings
  if "altloc_id" in array.get_annotation_categories():
    array = array[(array.altloc_id == '') | (array.altloc_id == 'A') | (array.altloc_id == '1')]
    
  if len(array) == 0:
    return None, None

  # 3. Extract 1-letter Sequence and exact Residue IDs
  starts = struc.get_residue_starts(array)
  real_res_ids = array.res_id[starts]
  res_names = array.res_name[starts]
  
  # Map 3-letter to 1-letter, defaulting to 'X' for unknown/non-standard AAs
  sequence = "".join([AA_MAP.get(name, 'X') for name in res_names])
  
  return sequence, real_res_ids


def _get_esmc_embeddings(sequence, model, tokenizer, device):
  """
  Helper function to handle ESM-C inference. 
  Isolating this makes it easy to swap syntax if using HuggingFace vs Native ESMC.
  """
  # Truncate sequence if it exceeds standard LM limits (e.g., 1024 or 2048)
  # Adjust this if your ESM-C model supports longer contexts natively
  max_len = 1022 
  if len(sequence) > max_len:
    sequence = sequence[:max_len]
    
  # Standard HuggingFace / ESM tokenization logic
  inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(device)
  
  with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True)
    # Extract the last hidden state 
    # Shape is usually (batch_size, sequence_length + special_tokens, embedding_dim)
    hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs.hidden_states[-1]
    
  # Strip the <cls> token at index 0 and <eos> token at the end
  # This aligns the tensor exactly with the original amino acid sequence
  embeddings = hidden_states[0, 1:len(sequence)+1, :].cpu().numpy()
  
  return embeddings


def save_esmc_embeddings():
  labels_df = pd.read_csv('residue_switch_labels.csv')
  cif_dir = 'fixed_cif_files'
  grouped = labels_df.groupby('Raw_Input_ID')
  features_dict = {}
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # --- 1. Initialize ESM-C Model ---
  print("Loading ESM-C (600M) Model...")
  
  try:
    # Attempt to load using standard HuggingFace transformers API
    from transformers import AutoTokenizer, AutoModel
    model_name = "EvolutionaryScale/esmc-600m" # Update this path if stored locally
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
  except ImportError:
    # Fallback to native esmc package if installed that way
    import esmc
    model = esmc.ESMC.from_pretrained("esmc-600m").to(device)
    model.eval()
    tokenizer = model.tokenizer
  
  # --- 2. Run ESM-C Inference ---
  for raw_input_id, group in tqdm(grouped, desc="Extracting ESM-C Embeddings"):
    required_res_ids = group['Residue_ID'].unique()
    sequence, real_res_ids = _extract_sequence_and_ids(raw_input_id, cif_dir, required_res_ids)
    
    if sequence is None or len(sequence) == 0:
      print(f"\nWarning: Valid sequence not found for {raw_input_id}. Skipping.")
      continue
      
    try:
      # Generate embeddings for the sequence
      embeddings = _get_esmc_embeddings(sequence, model, tokenizer, device)
    except Exception as e:
      print(f"\nError processing {raw_input_id}: {e}")
      continue 
    
    # Map residue IDs to their corresponding embedding vector.
    # We slice real_res_ids in case the sequence was truncated due to max_len limits
    res_to_embed = {res_id: emb for res_id, emb in zip(real_res_ids[:len(embeddings)], embeddings)}
    
    # --- 3. Save only the requested Residue IDs ---
    for res_id in required_res_ids:
      if res_id in res_to_embed:
        features_dict[(raw_input_id, res_id)] = res_to_embed[res_id]
      else:
        print(f"\nWarning: Residue ID {res_id} not found in parsed embeddings for {raw_input_id}")

  # --- 4. Save the Dictionary ---
  save_dir = Path('saved_data/feature_extraction')
  save_dir.mkdir(parents=True, exist_ok=True)
  
  save_file_path = save_dir / 'esmc_features.pkl'
  with open(save_file_path, 'wb') as f:
    pickle.dump(features_dict, f)
    
  print(f"\nSuccessfully saved {len(features_dict)} embeddings to {save_file_path}")


if __name__ == "__main__":
  save_esmc_embeddings()