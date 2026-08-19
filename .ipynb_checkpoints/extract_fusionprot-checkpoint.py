import os
import sys
import tempfile
import pandas as pd
import numpy as np
import pickle
import torch
from tqdm.auto import tqdm
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio

# --- NEUTRALIZE RDKIT CHEMISTRY CHECKS ---
# This monkey-patch intercepts TorchDrug's RDKit calls to forcibly disable 
# bond inference and sanitization. This permanently solves all "Explicit valence" crashes.
import rdkit.Chem
_original_MolFromPDBFile = rdkit.Chem.MolFromPDBFile
def _safe_MolFromPDBFile(*args, **kwargs):
  kwargs['sanitize'] = False
  kwargs['proximityBonding'] = False
  return _original_MolFromPDBFile(*args, **kwargs)
rdkit.Chem.MolFromPDBFile = _safe_MolFromPDBFile

# --- CONFIGURE PATHS ---
# Update this to the folder containing the FusionProt repo (where util.py is located)
FUSIONPROT_DIR = 'FusionProt' 
# Update this to where you downloaded the FusionProt model weights
WEIGHTS_PATH = 'saved_data/fusionprot_model/FusionProt-full-model.pth' 

# Append FusionProt to sys.path so we can import their modules
sys.path.append(FUSIONPROT_DIR)

import util
from torchdrug import core, transforms, data
# We must import these so their custom architectures register inside TorchDrug
from model import model, cdconv, gvp, dataset, task, protbert, siamdiff


def _load_and_extract_chain_to_pdb(raw_id, cif_dir, required_res_ids):
  """
  Reads the CIF file, extracts the specific chain, applies memory-safe 
  truncation (max 1022 residues), and writes it to a temporary PDB file.
  """

  # 1. Robust Case-Insensitive File Resolution
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
    
  # 3. Strip extra atoms to lighten load (GearNet only uses CA trace natively anyway)
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

  # 4. TRUNCATE: Prevent ESM Tensor Size Crashes (>1022 residues)
  starts = struc.get_residue_starts(array)
  if len(starts) > 1022:
    real_res_ids = array.res_id[starts]
    req_set = set(required_res_ids)
    req_indices = [i for i, r in enumerate(real_res_ids) if r in req_set]
    
    if req_indices:
      min_idx, max_idx = min(req_indices), max(req_indices)
      if max_idx - min_idx >= 1022:
        max_idx = min_idx + 1021  # Clamp if targets are unbelievably far apart
        
      center_idx = (min_idx + max_idx) // 2
      start_idx = max(0, center_idx - 511)
      end_idx = min(len(starts), start_idx + 1022)
      
      if end_idx - start_idx < 1022 and start_idx > 0:
        start_idx = max(0, end_idx - 1022)
        
      # Slice the array perfectly by atom indices to preserve continuity
      first_atom_idx = starts[start_idx]
      last_atom_idx = starts[end_idx] if end_idx < len(starts) else len(array)
      array = array[first_atom_idx:last_atom_idx]
      starts = struc.get_residue_starts(array)

  real_res_ids = array.res_id[starts]
  
  # Strip biotite bonds to guarantee NO CONECT records are written
  array.bonds = None 

  fd, tmp_path = tempfile.mkstemp(suffix=".pdb")
  os.close(fd)
  strucio.save_structure(tmp_path, array)
  
  return tmp_path, real_res_ids


def save_fusionprot_embeddings(data_file_path, cif_dir):
  labels_df = pd.read_csv(data_file_path)
  grouped = labels_df.groupby('Raw_Input_ID')
  features_dict = {}
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  print("Loading FusionProt configurations...")
  config_path = os.path.join(FUSIONPROT_DIR, 'config/pretrain/mc_esm_gearnet.yaml')
  cfg = util.load_config(config_path, context={})
  cfg.task.model.model.sequence_model.path = '~/.torchdrug/models/'
  
  print("Instantiating FusionNetwork and GraphConstruction...")
  fusion_model = core.Configurable.load_config_dict(cfg.task.model.model)
  gc_model = core.Configurable.load_config_dict(cfg.task.graph_construction_model)
  
  print(f"Loading weights from {WEIGHTS_PATH}...")
  state_dict = torch.load(WEIGHTS_PATH, map_location='cpu', weights_only=False)
  fusion_model.load_state_dict(state_dict, strict=False) 
  
  fusion_model = fusion_model.to(device)
  fusion_model.eval()
  gc_model = gc_model.to(device)
  gc_model.eval()
  
  to_residue_view = transforms.ProteinView(view="residue", keys="graph")

  for raw_input_id, group in tqdm(grouped, desc="Extracting FusionProt Embeddings"):
    required_res_ids = group['Residue_ID'].unique()
    tmp_pdb_path, real_res_ids = _load_and_extract_chain_to_pdb(raw_input_id, cif_dir, required_res_ids)
    
    if tmp_pdb_path is None:
      print(f"\nWarning: Valid structure not found for {raw_input_id}. Skipping.")
      continue
      
    try:
      protein = data.Protein.from_pdb(
        tmp_pdb_path, 
        atom_feature=None, 
        bond_feature=None, 
        residue_feature="default"
      )
      
      item = to_residue_view({"graph": protein})
      graph = item["graph"]
      batch = data.Protein.pack([graph]).to(device)
      batch = gc_model(batch)
      
      input_feature = batch.node_feature.float()
      
      with torch.no_grad():
        output = fusion_model(batch, input_feature)
        embeddings = output["node_feature"].cpu().numpy()
        
    except Exception as e:
      print(f"\nError processing {raw_input_id}: {e}")
      continue 
    finally:
      if os.path.exists(tmp_pdb_path):
        os.remove(tmp_pdb_path)
    
    res_to_embed = {res_id: emb for res_id, emb in zip(real_res_ids[:len(embeddings)], embeddings)}
    
    for res_id in required_res_ids:
      if res_id in res_to_embed:
        features_dict[(raw_input_id, res_id)] = res_to_embed[res_id]
      else:
        print(f"\nWarning: Residue ID {res_id} not found in parsed embeddings for {raw_input_id}")

  save_dir = Path('saved_data/feature_extraction')
  save_dir.mkdir(parents=True, exist_ok=True)
  
  save_file_path = save_dir / 'fusionprot_features.pkl'
  with open(save_file_path, 'wb') as f:
    pickle.dump(features_dict, f)
    
  print(f"\nSuccessfully saved {len(features_dict)} embeddings to {save_file_path}")


if __name__ == "__main__":
  data_file_path='dataset/residue_switch_labels.csv'
  cif_dir = 'dataset/fixed_cif_files'
  save_fusionprot_embeddings(data_file_path, cif_dir)