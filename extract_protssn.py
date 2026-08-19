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

# --- CONFIGURE PATHS ---
PROTSSN_DIR = 'ProtSSN'
WEIGHTS_PATH = 'saved_data/protssn_model/protssn_k20_h512.pt'

sys.path.append(PROTSSN_DIR)

import yaml
from torch_geometric.loader import DataLoader
from src.models import PLM_model, GNN_model
from src.data import build_mutant_dataset
from src.utils.utils import param_num

def _load_and_extract_chain_to_pdb(raw_id, cif_dir, target_dir):
  """
  Reads the CIF file, extracts the specific chain, applies memory-safe 
  sliding window chunking (max 1022 residues), and writes each chunk to the target dataset directory.
  Returns a list of (chunk_id, real_res_ids) tuples.
  """
  source = os.path.join(cif_dir, f"{raw_id}.cif")
  try:
    array = strucio.load_structure(source)
  except Exception as e:
    print(f"\nError reading {source}: {e}")
    return []

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
    return []

  array.bonds = None 

  starts = struc.get_residue_starts(array)
  
  max_len = 1022
  overlap = 511
  stride = max_len - overlap

  chunk_ranges = []
  if len(starts) <= max_len:
    chunk_ranges.append((0, len(starts)))
  else:
    idx = 0
    while idx < len(starts):
      start_idx = idx
      end_idx = min(len(starts), idx + max_len)
      chunk_ranges.append((start_idx, end_idx))
      if end_idx == len(starts):
        break
      idx += stride

  chunks_info = []
  for i, (start_idx, end_idx) in enumerate(chunk_ranges):
    first_atom_idx = starts[start_idx]
    last_atom_idx = starts[end_idx] if end_idx < len(starts) else len(array)
    
    chunk_array = array[first_atom_idx:last_atom_idx]
    chunk_starts = struc.get_residue_starts(chunk_array)
    chunk_res_ids = chunk_array.res_id[chunk_starts]
    
    chunk_id = f"{raw_id}_chunk{i}"
    protein_folder = os.path.join(target_dir, chunk_id)
    os.makedirs(protein_folder, exist_ok=True)
    target_path = os.path.join(protein_folder, f"{chunk_id}.pdb")
    
    strucio.save_structure(target_path, chunk_array)
    chunks_info.append((chunk_id, chunk_res_ids))

  return chunks_info


class DummyArgs:
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      setattr(self, k, v)


def save_protssn_embeddings(data_file_path, cif_dir):
  labels_df = pd.read_csv(data_file_path)
  grouped = labels_df.groupby('Raw_Input_ID')
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  # We will build a temporary directory formatted exactly how ProtSSN expects:
  # tmp_dataset_dir/DATASET/protein_id/protein_id.pdb
  tmp_dataset_dir = tempfile.mkdtemp()
  raw_dataset_dir = os.path.join(tmp_dataset_dir, "DATASET")
  os.makedirs(raw_dataset_dir, exist_ok=True)
  
  protein_id_to_real_res = {}
  chunk_to_protein = {}
  
  print(f"Constructing ProtSSN dataset structure in {tmp_dataset_dir}...")
  for raw_input_id, group in tqdm(grouped, desc="Preparing PDBs"):
    chunks_info = _load_and_extract_chain_to_pdb(raw_input_id, cif_dir, raw_dataset_dir)
    
    if not chunks_info:
      print(f"Warning: Valid structure not found for {raw_input_id}. Skipping.")
    else:
      for chunk_id, real_res_ids in chunks_info:
        protein_id_to_real_res[chunk_id] = real_res_ids
        chunk_to_protein[chunk_id] = raw_input_id

  # Prepare model args matching ProtSSN k20_h512
  k = 20
  h = 512
  args = DummyArgs(
    gnn="egnn",
    gnn_config=os.path.join(PROTSSN_DIR, "src/config/egnn.yaml"),
    gnn_model_dir=os.path.dirname(WEIGHTS_PATH),
    plm="facebook/esm2_t33_650M_UR50D",
    mutant_dataset_dir=tmp_dataset_dir,
    mutant_name="extracted_k20",
    c_alpha_max_neighbors=k
  )
  
  args.gnn_config = yaml.load(open(args.gnn_config), Loader=yaml.FullLoader)[args.gnn]
  args.gnn_config["hidden_channels"] = h
  
  print("Loading Models...")
  plm_model = PLM_model(args)
  args.plm_hidden_size = plm_model.model.config.hidden_size
  gnn_model = GNN_model(args)
  gnn_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location='cpu'))
  
  plm_model = plm_model.to(device)
  plm_model.eval()
  gnn_model = gnn_model.to(device)
  gnn_model.eval()
  
  # This will trigger processing of all PDBs in tmp_dataset_dir/DATASET
  print("Building ProtSSN PyG Dataset...")
  # NOTE: run inside ProtSSN directory so relative paths like "norm/cath_..." work
  os.chdir(PROTSSN_DIR)
  # Ensure norm files are accessed relatively correctly
  args.mutant_dataset_dir = f"../{tmp_dataset_dir}" if not tmp_dataset_dir.startswith('/') else tmp_dataset_dir
  mutant_dataset = build_mutant_dataset(args)
  mutant_loader = DataLoader(mutant_dataset, batch_size=1, shuffle=False)
  os.chdir("..")

  print(f"Extracting embeddings for {len(mutant_dataset)} valid chunks...")
  features_dict = {}
  raw_res_to_embeds = {}
  
  with torch.no_grad():
    for data in tqdm(mutant_loader, desc="ProtSSN Inference"):
      chunk_id = data.protein_name[0]
      data = data.to(device)
      
      graph_data = plm_model(data)
      out, gnn_embed = gnn_model(graph_data)
      gnn_embed = gnn_embed.cpu().numpy()
      
      real_res_ids = protein_id_to_real_res[chunk_id]
      raw_id = chunk_to_protein[chunk_id]
      
      if raw_id not in raw_res_to_embeds:
        raw_res_to_embeds[raw_id] = {}
        
      for res_id, emb in zip(real_res_ids[:len(gnn_embed)], gnn_embed):
        if res_id not in raw_res_to_embeds[raw_id]:
          raw_res_to_embeds[raw_id][res_id] = []
        raw_res_to_embeds[raw_id][res_id].append(emb)
        
  for raw_input_id, group in grouped:
    required_res_ids = group['Residue_ID'].unique()
    if raw_input_id in raw_res_to_embeds:
      for res_id in required_res_ids:
        if res_id in raw_res_to_embeds[raw_input_id]:
          features_dict[(raw_input_id, res_id)] = np.mean(raw_res_to_embeds[raw_input_id][res_id], axis=0)
        else:
          print(f"Warning: Residue ID {res_id} not found in embeddings for {raw_input_id}")

  # Cleanup temp directory
  shutil.rmtree(tmp_dataset_dir)
  
  save_dir = Path('saved_data/feature_extraction')
  save_dir.mkdir(parents=True, exist_ok=True)
  save_file_path = save_dir / 'protssn_features.pkl'
  with open(save_file_path, 'wb') as f:
    pickle.dump(features_dict, f)
      
  print(f"\nSuccessfully saved {len(features_dict)} ProtSSN embeddings to {save_file_path}")

if __name__ == "__main__":
  data_file_path = 'dataset/residue_secondary_structures.csv'
  cif_dir = 'dataset/fixed_cif_files'
  save_protssn_embeddings(data_file_path, cif_dir)
