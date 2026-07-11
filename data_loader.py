import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from config import SS_CLASSES
from metrics import decode_labels

def load_and_preprocess_data(feature_type, k_val, models_dir):
  """Loads data, standardizes it, and returns the splits along with decoded labels."""
  print(f"Loading {feature_type} data...")
  
  if feature_type == 'fusionprot':
    data = np.load('saved_data/feature_extraction/fusionprot.npz') 
  elif feature_type == 'saprot':
    data = np.load('saved_data/feature_extraction/saprot.npz')
  elif feature_type == 'protssn':
    data = np.load('saved_data/feature_extraction/protssn.npz')
  elif feature_type == 'prosst':
    data = np.load('saved_data/feature_extraction/prosst.npz')
  elif feature_type == 'kncsic':
    data = np.load(f'saved_data/feature_extraction/k{k_val}ncsic.npz')
  elif feature_type == 'kndrf':
    data = np.load(f'saved_data/feature_extraction/k{k_val}ndrf.npz') 
  else:
    raise ValueError(f"Unknown feature type: {feature_type}")
  
  X_train_raw = np.float32(data['X_train'])
  y_train_raw = data['y_train']
  X_val_raw = np.float32(data['X_val'])
  y_val_raw = data['y_val']
  X_test_raw = np.float32(data['X_test'])
  y_test_raw = data['y_test']
  
  print(f"Standardizing {feature_type.capitalize()} features (Fitting on Train)...")
  scaler = StandardScaler()
  
  X_train_raw = np.nan_to_num(X_train_raw)
  X_val_raw = np.nan_to_num(X_val_raw)
  X_test_raw = np.nan_to_num(X_test_raw)

  X_train_raw = scaler.fit_transform(X_train_raw).astype(np.float32)
  X_val_raw = scaler.transform(X_val_raw).astype(np.float32)
  X_test_raw = scaler.transform(X_test_raw).astype(np.float32)
  
  X_train_raw = np.nan_to_num(X_train_raw)
  X_val_raw = np.nan_to_num(X_val_raw)
  X_test_raw = np.nan_to_num(X_test_raw)

  scaler_path = models_dir / f'{feature_type}_scaler.joblib'
  joblib.dump(scaler, scaler_path)
  print(f"Saved fitted StandardScaler to {scaler_path}")

  # Decode Labels
  train_orig, y_train_bin, y_train_3c = decode_labels(y_train_raw)
  val_orig,   y_val_bin,   y_val_3c   = decode_labels(y_val_raw)
  test_orig,  y_test_bin,  y_test_3c  = decode_labels(y_test_raw)

  test_states_idx = np.array([SS_CLASSES.index(char) for char in test_orig])

  return {
    'X_train': X_train_raw, 'y_train_raw': y_train_raw, 'y_train_bin': y_train_bin, 'y_train_3c': y_train_3c, 'train_orig': train_orig,
    'X_val': X_val_raw, 'y_val_raw': y_val_raw, 'y_val_bin': y_val_bin, 'y_val_3c': y_val_3c, 'val_orig': val_orig,
    'X_test': X_test_raw, 'y_test_raw': y_test_raw, 'y_test_bin': y_test_bin, 'y_test_3c': y_test_3c, 'test_orig': test_orig,
    'test_states_idx': test_states_idx
  }
