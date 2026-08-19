import os
import glob
import argparse
from pdbfixer import PDBFixer
import biotite.structure.io as strucio
from openmm.app import PDBxFile
from tqdm.auto import tqdm
import numpy as np

import pandas as pd

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Fix and isolate PDB chains.")
  parser.add_argument("--redo", action="store_true", help="Redo fixing, overwriting existing files.")
  args = parser.parse_args()

  os.makedirs("dataset/fixed_cif_files", exist_ok=True)

  data_df = pd.read_csv("dataset/training_data.csv")
  for raw_id in tqdm(data_df['PDB_ID'], total=len(data_df)):
    """
    Uses PDBFixer to isolate a specific chain or entity, adds missing atoms, 
    and removes all other chains (including duplicate instances) from the structure.
    """
    output_cif = f"dataset/fixed_cif_files/{raw_id}.cif"
    if not args.redo and os.path.exists(output_cif):
      # tqdm.write(f"\nSkipping {raw_id}, already exists.") # Optional, could be noisy
      continue

    tqdm.write(f"\nProcessing {raw_id}...")
    try:
    # 1. Resolve the Target Chain ID
    if "." in raw_id:
      pdb_id, target_chain = raw_id.split(".", 1)
      tqdm.write(f"  -> Resolved PDB ID {pdb_id} to Chain {target_chain}")
      input_cif = f"dataset/cif_files/{pdb_id}.cif"
    elif "_" in raw_id:
      pdb_id, entity_id = raw_id.split("_", 1)
      input_cif = f"dataset/cif_files/{pdb_id}.cif"
      
      # PDBFixer doesn't read entity IDs, so we use Biotite to find the mapping
      array = strucio.load_structure(input_cif, model=1, extra_fields=["label_entity_id"])
      entity_mask = (array.label_entity_id == str(entity_id))
      
      if not np.any(entity_mask):
        raise ValueError(f"Entity ID {entity_id} not found in {input_cif}")
        
      # Grab the first chain associated with this entity
      target_chain = np.unique(array.chain_id[entity_mask])[0]
      tqdm.write(f"  -> Resolved Entity {entity_id} to Chain {target_chain}")
      
    else:
      raise ValueError(f"Unrecognized ID format: {raw_id}")
  
    # 2. Load the structure into PDBFixer
    fixer = PDBFixer(filename=input_cif)
  
    # 3. Find indices of all chains EXCEPT the first instance of the target chain
    chains_to_remove = []
    kept_primary_chain = False
    
    # fixer.topology.chains() yields OpenMM Chain objects
    for i, chain in enumerate(fixer.topology.chains()):
      if chain.id == target_chain and not kept_primary_chain:
        # We found the first instance of our target chain. Keep it.
        kept_primary_chain = True
      else:
        # Delete everything else (other chains, and any duplicate/broken target chains)
        chains_to_remove.append(i)
  
    # Check if we accidentally marked everything for deletion
    if not kept_primary_chain:
      raise ValueError(f"Chain {target_chain} was not found in the PDBFixer topology!")
      
    tqdm.write(f"  -> Removing {len(chains_to_remove)} other chain segment(s)...")
    fixer.removeChains(chains_to_remove)
  
    # 4. Fix the single, isolated chain
    tqdm.write("  -> Adding missing atoms...")
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
  
    # 5. Save the perfectly clean, single-chain structure
    with open(output_cif, 'w') as f:
      PDBxFile.writeFile(fixer.topology, fixer.positions, f)
      
    tqdm.write(f"  -> Success! Saved to {output_cif}\n")
  except Exception as e:
    print(f"\nFailed to fix {raw_id}: {e}\n")