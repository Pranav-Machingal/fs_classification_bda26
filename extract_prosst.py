import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pickle
import torch
import shutil
from tqdm.auto import tqdm
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio

PROSST_DIR = 'ProSST'
sys.path.append(PROSST_DIR)

from transformers import AutoTokenizer, AutoModelForMaskedLM
from prosst.structure.get_sst_seq import SSTPredictor

def _load_and_extract_chain_to_pdb(raw_id, cif_dir, required_res_ids, target_path):
  """
  Reads the CIF file, extracts the specific chain, applies memory-safe 
  truncation (max 1022 residues), and writes it to a specific target path.
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
      
  array = array[array.element != "H"]
  array = array[np.isin(array.atom_name, ["N", "CA", "C", "O", "CB"])]
  
  df_atoms = pd.DataFrame({
    'res_id': array.res_id,
    'ins_code': array.ins_code,
    'atom_name': array.atom_name,
    'idx': np.arange(len(array))
  })
  keep_idx = df_atoms.drop_duplicates(subset=['res_id', 'ins_code', 'atom_name'])['idx'].values
  array = array[keep_idx]
  
  if len(array) == 0:
    return None, None

  starts = struc.get_residue_starts(array)
  if len(starts) > 1022:
    real_res_ids = array.res_id[starts]
    req_set = set(required_res_ids)
    req_indices = [i for i, r in enumerate(real_res_ids) if r in req_set]
    
    if req_indices:
      min_idx, max_idx = min(req_indices), max(req_indices)
      if max_idx - min_idx >= 1022:
        max_idx = min_idx + 1021
          
      center_idx = (min_idx + max_idx) // 2
      start_idx = max(0, center_idx - 511)
      end_idx = min(len(starts), start_idx + 1022)
      
      if end_idx - start_idx < 1022 and start_idx > 0:
        start_idx = max(0, end_idx - 1022)
          
      first_atom_idx = starts[start_idx]
      last_atom_idx = starts[end_idx] if end_idx < len(starts) else len(array)
      array = array[first_atom_idx:last_atom_idx]
      starts = struc.get_residue_starts(array)

  real_res_ids = array.res_id[starts]
  array.bonds = None 

  strucio.save_structure(target_path, array)
  return target_path, real_res_ids

def tokenize_structure_sequence(structure_sequence):
  shift_structure_sequence = [i + 3 for i in structure_sequence]
  shift_structure_sequence = [1, *shift_structure_sequence, 2]
  return torch.tensor([shift_structure_sequence], dtype=torch.long)

def save_prosst_embeddings():
  labels_df = pd.read_csv('residue_switch_labels.csv')
  cif_dir = 'fixed_cif_files'
  grouped = labels_df.groupby('Raw_Input_ID')
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  print("Loading ProSST Quantizer and Model...")
  # Suppress annoying multiprocessing warnings from pathos by setting num_processes=1
  predictor = SSTPredictor(structure_vocab_size=2048, device=device, num_processes=1, num_threads=1)
  
  # Load the locally saved model
  model_path = "saved_data/ProSST-2048"
  model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(device)
  tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
  model.eval()

  features_dict = {}
  
  for raw_input_id, group in tqdm(grouped, desc="Extracting ProSST Embeddings"):
    required_res_ids = group['Residue_ID'].unique()
    
    fd, tmp_pdb_path = tempfile.mkstemp(suffix=".pdb")
    os.close(fd)
    
    tmp_path, real_res_ids = _load_and_extract_chain_to_pdb(raw_input_id, cif_dir, required_res_ids, tmp_pdb_path)
    
    if tmp_path is None:
      print(f"Warning: Valid structure not found for {raw_input_id}. Skipping.")
      if os.path.exists(tmp_pdb_path):
        os.remove(tmp_pdb_path)
      continue
    
    try:
      # Predict Structure Tokens
      pred_result = predictor.predict_from_pdb(tmp_pdb_path)
      if not pred_result or len(pred_result) == 0:
        continue
        
      aa_seq = pred_result[0]["aa_seq"]
      sst_seq = pred_result[0]["2048_sst_seq"]
      
      ss_input_ids = tokenize_structure_sequence(sst_seq).to(device)
      tokenized_results = tokenizer([aa_seq], return_tensors="pt")
      input_ids = tokenized_results["input_ids"].to(device)
      attention_mask = tokenized_results["attention_mask"].to(device)
      
      with torch.no_grad():
        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask,
          ss_input_ids=ss_input_ids,
          output_hidden_states=True
        )
        
        # Huggingface typically returns a tuple of hidden states
        # Shape: (1, seq_len + 2, hidden_dim)
        hidden_states = outputs.hidden_states[-1]
        embeddings = hidden_states[0, 1:-1, :].cpu().numpy()
        
      res_to_embed = {res_id: emb for res_id, emb in zip(real_res_ids[:len(embeddings)], embeddings)}
      
      for res_id in required_res_ids:
        if res_id in res_to_embed:
          features_dict[(raw_input_id, res_id)] = res_to_embed[res_id]
        else:
          print(f"Warning: Residue ID {res_id} not found in embeddings for {raw_input_id}")
          
    except Exception as e:
      print(f"\nError processing {raw_input_id}: {e}")
    finally:
      if os.path.exists(tmp_pdb_path):
        os.remove(tmp_pdb_path)

  save_dir = Path('saved_data/feature_extraction')
  save_dir.mkdir(parents=True, exist_ok=True)
  save_file_path = save_dir / 'prosst_features.pkl'
  with open(save_file_path, 'wb') as f:
    pickle.dump(features_dict, f)
      
  print(f"\nSuccessfully saved {len(features_dict)} ProSST embeddings to {save_file_path}")

if __name__ == "__main__":
  save_prosst_embeddings()
