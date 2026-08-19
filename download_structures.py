# Download the pdb structures
import os
import re
import pandas as pd
import biotite.database.rcsb as rcsb
import glob
import biotite.structure.io as strucio

def download_pdb_biotite(pdb_ids, output_dir="dataset/pdb_files", redo=False):
  
  # Create output directory
  os.makedirs(output_dir, exist_ok=True)
  
  if not redo:
    pdb_ids_to_fetch = []
    for pid in pdb_ids:
      if not (os.path.exists(os.path.join(output_dir, f"{pid.lower()}.pdb")) or 
              os.path.exists(os.path.join(output_dir, f"{pid.upper()}.pdb"))):
        pdb_ids_to_fetch.append(pid)
    
    if not pdb_ids_to_fetch:
      print("All PDB structures already downloaded. Skipping download.")
      return
    
    pdb_ids = pdb_ids_to_fetch

  print(f"Found {len(pdb_ids)} unique PDB IDs to fetch. Starting download...")
  
  # Biotite fetches the entire list efficiently
  downloaded_files = rcsb.fetch(
    pdb_ids, 
    target_path=output_dir, 
    format="pdb", 
    overwrite=redo
  )
  
  print(f"Successfully downloaded structures to '{output_dir}'.")


def convert_pdb_to_cif(input_dir="dataset/pdb_files", output_dir="dataset/cif_files", redo=False):
  # Create the output directory if it doesn't exist
  os.makedirs(output_dir, exist_ok=True)
  
  # Find all .pdb files in the input directory
  pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))
  
  if not pdb_files:
    print(f"No .pdb files found in '{input_dir}'.")
    return
    
  files_to_convert = []
  if not redo:
    for pdb_path in pdb_files:
      base_name = os.path.splitext(os.path.basename(pdb_path))[0]
      cif_path = os.path.join(output_dir, f"{base_name}.cif")
      if not os.path.exists(cif_path):
        files_to_convert.append(pdb_path)
  else:
    files_to_convert = pdb_files

  if not files_to_convert:
    print("All CIF files already exist. Skipping conversion.")
    return
    
  print(f"Found {len(files_to_convert)} PDB files to convert. Starting conversion...")
  
  for pdb_path in files_to_convert:
    # Extract the base filename (e.g., '2N0A' from '2N0A.pdb')
    filename = os.path.basename(pdb_path)
    base_name = os.path.splitext(filename)[0]
    cif_path = os.path.join(output_dir, f"{base_name}.cif")
    
    try:
      # Read the legacy PDB format
      array = strucio.load_structure(pdb_path, model=1)
      
      # Biotite infers the mmCIF format automatically from the .cif extension
      strucio.save_structure(cif_path, array)
      
      print(f"Converted: {filename} -> {base_name}.cif")
      
    except Exception as e:
      print(f"Failed to convert {filename}: {e}")
      
  print(f"\nConversion complete. Files saved to '{output_dir}'.")


if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description="Download and convert PDB structures.")
  parser.add_argument("--redo", action="store_true", help="Redo download and conversion, overwriting existing files.")
  args = parser.parse_args()

  csv_path = "dataset/training_data.csv"
  
  # Read the CSV file
  df = pd.read_csv(csv_path)
  
  # Extract PDB ID handling both '.' and '_' delimiters
  # e.g., '1LMB_3' -> '1LMB', '2BMO.B' -> '2BMO'
  pdb_ids = df['PDB_ID'].apply(lambda x: re.split(r'[._]', str(x))[0]).unique().tolist()
  
  download_pdb_biotite(pdb_ids, redo=args.redo)
  convert_pdb_to_cif(redo=args.redo)