import numpy as np
import pandas as pd
import pickle
import joblib
import sys
import argparse
import optuna
from optuna.samplers import TPESampler
from pathlib import Path
from tqdm.auto import tqdm

# Enable Scikit-Learn Metadata Routing to allow sample_weights inside OvO/OvR wrappers
import sklearn
sklearn.set_config(enable_metadata_routing=True)

# Preprocessing for numerical stability
from sklearn.preprocessing import StandardScaler

# Hide Optuna's per-trial spam in the terminal
optuna.logging.set_verbosity(optuna.logging.WARNING)

# RAPIDS cuML for GPU Acceleration
from cuml.svm import SVC, LinearSVC

# Scikit-learn wrappers
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Import helper functions
from classifier_utils import (
  decode_labels, compute_test_metrics, SS_CLASSES, compute_switch_f1, compute_sample_weights,
  IndividuallyTunedOvR, IndividuallyTunedOvO, StateRoutedCombinedClassifier
)

# --- 1. Parse Command Line Arguments ---
parser = argparse.ArgumentParser(description="Train cuML SVM classifiers (Unified Pipeline).")
parser.add_argument('--feature_type', type=str, default='fusionprot', choices=['kndrf', 'fusionprot', 'saprot', 'protssn', 'prosst'], help="Feature set to use for training")
parser.add_argument('--k', type=int, default=3, help="k value (Used only if feature_type is 'kndrf')")
parser.add_argument('--n_jobs', type=int, default=2, help="Number of parallel Optuna trials")
parser.add_argument('--n_trials', type=int, default=30, help="Total number of hyperparameter combinations to test")
parser.add_argument('--weight_strategies', nargs='+', default=['all'], choices=['all', 'binary', 'type'], help="Sample weighting strategies to evaluate")
parser.add_argument('--tasks', nargs='+', default=['all'], 
                    choices=['all', 'global_binary', 'global_3class', 'input_dep_binary', 'input_dep_3class'],
                    help="Which classifier suites to train")
parser.add_argument('--multi_class_mode', type=str, default='both', 
                    choices=['CommonC', 'Indiv', 'both'], 
                    help="Which 3-class classifier strategy to train")
args = parser.parse_args()

FEATURE_TYPE = args.feature_type
K_VAL = args.k
N_JOBS = args.n_jobs
N_TRIALS = args.n_trials
MULTI_CLASS_MODE = args.multi_class_mode
tasks_to_run = args.tasks

if 'all' in tasks_to_run:
  tasks_to_run = ['global_binary', 'global_3class', 'input_dep_binary', 'input_dep_3class']
  
strategies_to_run = ['binary', 'type'] if 'all' in args.weight_strategies else args.weight_strategies

# Use threading backend for cuML to share the CUDA context safely.
BACKEND = "threading"

# --- 2. Setup Directories ---
if FEATURE_TYPE == 'fusionprot':
  BASE_DIR = Path('saved_data/svm_experiments_fusionprot')
elif FEATURE_TYPE == 'saprot':
  BASE_DIR = Path('saved_data/svm_experiments_saprot')
elif FEATURE_TYPE == 'protssn':
  BASE_DIR = Path('saved_data/svm_experiments_protssn')
elif FEATURE_TYPE == 'prosst':
  BASE_DIR = Path('saved_data/svm_experiments_prosst')
else:
  BASE_DIR = Path(f'saved_data/svm_experiments_k{K_VAL}')

MODELS_DIR = BASE_DIR / 'models'
METRICS_DIR = BASE_DIR / 'metrics'
TUNING_DIR = BASE_DIR / 'tuning'
CURVES_DIR = BASE_DIR / 'curves'

for d in [MODELS_DIR, METRICS_DIR, TUNING_DIR, CURVES_DIR]:
  d.mkdir(parents=True, exist_ok=True)

# --- 3. Core Tuning & Evaluation Engine ---
def save_evaluation(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, weight_strategy):
  y_test_pred = best_model.predict(X_test)
  
  if hasattr(best_model, "predict_proba"):
    y_score = best_model.predict_proba(X_test)
    if is_binary and y_score.ndim == 2:
      y_score = y_score[:, 1]
  else:
    y_score = best_model.decision_function(X_test)
  
  metrics_dict, cm, curves = compute_test_metrics(
    y_test, y_test_pred, y_score, is_binary, classes, test_orig, y_test_raw
  )

  prefix = f'{task_name}_{weight_strategy}_best_f1'
  
  joblib.dump(best_model, MODELS_DIR / f'{prefix}_model.joblib')
    
  with open(CURVES_DIR / f'{prefix}_curves.pkl', 'wb') as f:
    pickle.dump(curves, f)
  np.save(METRICS_DIR / f'{prefix}_confusion_matrix.npy', cm)
  
  return metrics_dict

def evaluate_and_summarize(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, weight_strategy, force_eval=False):
  summary_path = METRICS_DIR / f'{task_name}_{weight_strategy}_test_summary.pkl'
  if summary_path.exists() and not force_eval:
    print(f"  -> Skipping Evaluation for {task_name} ({weight_strategy}) (Already computed)")
    return

  print(f"  -> Testing Best Switch-F1 Model for {task_name} ({weight_strategy})")
  metrics = save_evaluation(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, weight_strategy)
  
  summary = {'best_f1_model_test_metrics': metrics}
  with open(summary_path, 'wb') as f:
    pickle.dump(summary, f)

def tune_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, task_name, train_orig, y_train_raw, val_orig, test_orig, y_test_raw, weight_strategy, multi_strategy=None):
  summary_path = METRICS_DIR / f'{task_name}_{weight_strategy}_test_summary.pkl'
  tuning_csv_path = TUNING_DIR / f'{task_name}_{weight_strategy}_tuning.csv'
  f1_model_path = MODELS_DIR / f'{task_name}_{weight_strategy}_best_f1_model.joblib'
  db_path = TUNING_DIR / f"{task_name}_{weight_strategy}.db"
  
  is_binary = (multi_strategy is None)
  classes = [0, 1] if is_binary else [0, 1, 2]
  
  if summary_path.exists() and f1_model_path.exists() and db_path.exists():
    study_check = optuna.load_study(study_name=f"{task_name}_{weight_strategy}", storage=f"sqlite:///{db_path.absolute()}")
    if len([t for t in study_check.trials if t.state == optuna.trial.TrialState.COMPLETE]) >= N_TRIALS:
      print(f"\n--- Skipping Tuning: {task_name} [{weight_strategy}] (Required trials already computed) ---")
      return joblib.load(f1_model_path)

  print(f"\n--- Tuning via Optuna: {task_name} [{weight_strategy}] ---")
  
  def _build_model(params):
    import sklearn
    sklearn.set_config(enable_metadata_routing=True)
    
    if params['kernel'] == 'linear':
      base_svm = LinearSVC(C=params['C'], max_iter=10000) 
    else:
      base_svm = SVC(
        C=params['C'], 
        kernel=params['kernel'], 
        degree=params.get('degree', 3), 
        gamma=params['gamma'],
        random_state=42,
        max_iter=10000,
        cache_size=2048
      )
      
    base_svm = CalibratedClassifierCV(estimator=base_svm, ensemble=False)
    
    if hasattr(base_svm, 'set_fit_request'):
      base_svm.set_fit_request(sample_weight=True)
    
    if multi_strategy == 'ovo':
      wrapper = OneVsOneClassifier(base_svm, n_jobs=1)
      if hasattr(wrapper, 'set_fit_request'):
        wrapper.set_fit_request(sample_weight=True)
      return wrapper
    elif multi_strategy == 'ovr':
      wrapper = OneVsRestClassifier(base_svm, n_jobs=1)
      if hasattr(wrapper, 'set_fit_request'):
        wrapper.set_fit_request(sample_weight=True)
      return wrapper
    else:
      return base_svm

  def objective(trial):
    import sklearn
    sklearn.set_config(enable_metadata_routing=True)
    
    params = {
      'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf']),
      'C': trial.suggest_float('C', 1e-4, 1e3, log=True)
    }
    
    if params['kernel'] != 'linear':
      if FEATURE_TYPE in ['fusionprot', 'saprot']:
        params['gamma'] = trial.suggest_float('gamma', 1e-6, 1e1, log=True)
      else:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e2, log=True)
        
      if params['kernel'] == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 3)
        
    clf = _build_model(params)
    
    sample_weights = compute_sample_weights(y_train_raw, strategy=weight_strategy)
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    
    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)
    
    train_f1 = compute_switch_f1(y_train, y_train_pred, train_orig, is_binary)
    val_f1 = compute_switch_f1(y_val, y_val_pred, val_orig, is_binary)
    
    trial.set_user_attr('train_f1', train_f1)
    return val_f1
  
  storage_name = f"sqlite:///{db_path.absolute()}"
  study = optuna.create_study(
    directions=["maximize"], 
    storage=storage_name, 
    study_name=f"{task_name}_{weight_strategy}", 
    load_if_exists=True,
    sampler=TPESampler(seed=42)
  )
  
  completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
  trials_to_run = max(0, N_TRIALS - completed)
  
  if trials_to_run > 0:
    print(f"  -> Executing {trials_to_run} new trials (out of {N_TRIALS} total target)...")
    # CRITICAL FIX: Set n_jobs=1 to prevent cuML "Working set already initialized" CUDA memory collisions.
    study.optimize(objective, n_trials=trials_to_run, n_jobs=1)
  else:
    print(f"  -> Target of {N_TRIALS} trials already met in database.")

  best_trial = study.best_trial
  print(f"\n  -> Champion Configuration (Switch-F1: {best_trial.value:.4f}): {best_trial.params}")

  best_model = _build_model(best_trial.params)
  sample_weights = compute_sample_weights(y_train_raw, strategy=weight_strategy)
  best_model.fit(X_train, y_train, sample_weight=sample_weights)

  df = study.trials_dataframe(attrs=('params', 'values', 'user_attrs'))
  rename_map = {
    'params_kernel': 'kernel',
    'params_degree': 'degree',
    'params_C': 'C',
    'params_gamma': 'gamma',
    'values_0': 'val_f1',
    'user_attrs_train_f1': 'train_f1'
  }
  df = df.rename(columns=rename_map)
  cols_to_keep = [c for c in rename_map.values() if c in df.columns]
  df[cols_to_keep].to_csv(tuning_csv_path, index=False)
  
  evaluate_and_summarize(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, weight_strategy, force_eval=True)
  
  return best_model

# --- 4. Multi-Class Execution Suite ---
def run_multi_class_suite(X_train, y_train_3c, X_val, y_val_3c, X_test, y_test_3c, base_name, train_orig, y_train_raw, val_orig, test_orig, y_test_raw, weight_strategy, mode='both'):
  mdls = {}
  
  if mode in ['CommonC', 'both']:
    mdls['CommonC_OvO'] = tune_and_evaluate(X_train, y_train_3c, X_val, y_val_3c, X_test, y_test_3c, f"{base_name}_CommonC_OvO", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, weight_strategy, multi_strategy='ovo')
    mdls['CommonC_OvR'] = tune_and_evaluate(X_train, y_train_3c, X_val, y_val_3c, X_test, y_test_3c, f"{base_name}_CommonC_OvR", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, weight_strategy, multi_strategy='ovr')
  
  if mode in ['Indiv', 'both']:
    print(f"\n--- Constructing Individually Tuned OvR for {base_name} [{weight_strategy}] ---")
    f1_ests_ovr = []
    for cls in [0, 1, 2]:
      yt_bin = (y_train_3c == cls).astype(int)
      yv_bin = (y_val_3c == cls).astype(int)
      yts_bin = (y_test_3c == cls).astype(int)
      
      f1_mdl = tune_and_evaluate(
        X_train, yt_bin, X_val, yv_bin, X_test, yts_bin, 
        f"{base_name}_IndivOvR_Class{cls}vsRest",
        train_orig, y_train_raw, val_orig, test_orig, y_test_raw, weight_strategy
      )
      f1_ests_ovr.append(f1_mdl)

    ovr_f1 = IndividuallyTunedOvR(f1_ests_ovr, [0, 1, 2])
    evaluate_and_summarize(ovr_f1, X_test, y_test_3c, False, [0, 1, 2], f"{base_name}_IndivOvR_Combined", test_orig, y_test_raw, weight_strategy, force_eval=True)
    mdls['IndivOvR'] = ovr_f1

    print(f"\n--- Constructing Individually Tuned OvO for {base_name} [{weight_strategy}] ---")
    f1_ests_ovo = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for (c1, c2) in pairs:
      m_tr = (y_train_3c == c1) | (y_train_3c == c2)
      m_v = (y_val_3c == c1) | (y_val_3c == c2)
      m_ts = (y_test_3c == c1) | (y_test_3c == c2)

      yt_bin = np.where(y_train_3c[m_tr] == c1, 1, 0)
      yv_bin = np.where(y_val_3c[m_v] == c1, 1, 0)
      yts_bin = np.where(y_test_3c[m_ts] == c1, 1, 0)

      f1_mdl = tune_and_evaluate(
        X_train[m_tr], yt_bin, X_val[m_v], yv_bin, X_test[m_ts], yts_bin, 
        f"{base_name}_IndivOvO_Pair{c1}vs{c2}",
        train_orig[m_tr], y_train_raw[m_tr], val_orig[m_v], test_orig[m_ts], y_test_raw[m_ts], weight_strategy
      )
      f1_ests_ovo.append(f1_mdl)

    ovo_f1 = IndividuallyTunedOvO(f1_ests_ovo, [0, 1, 2], pairs)
    evaluate_and_summarize(ovo_f1, X_test, y_test_3c, False, [0, 1, 2], f"{base_name}_IndivOvO_Combined", test_orig, y_test_raw, weight_strategy, force_eval=True)
    mdls['IndivOvO'] = ovo_f1

  return mdls

if __name__ == "__main__":
  # --- 5. Data Assembly Pipeline ---
  print(f"Loading {FEATURE_TYPE} data...")
  if FEATURE_TYPE == 'fusionprot':
    data = np.load('saved_data/feature_extraction/fusionprot.npz') 
  elif FEATURE_TYPE == 'saprot':
    data = np.load('saved_data/feature_extraction/saprot.npz')
  elif FEATURE_TYPE == 'protssn':
    data = np.load('saved_data/feature_extraction/protssn.npz')
  elif FEATURE_TYPE == 'prosst':
    data = np.load('saved_data/feature_extraction/prosst.npz')
  else:
    data = np.load(f'saved_data/feature_extraction/k{K_VAL}ndrf.npz') 
  
  X_train_raw = np.float32(data['X_train'])
  y_train_raw = data['y_train']
  X_val_raw = np.float32(data['X_val'])
  y_val_raw = data['y_val']
  X_test_raw = np.float32(data['X_test'])
  y_test_raw = data['y_test']
  
  # --- CRITICAL FIX: Standardize Embeddings (FusionProt & SaProt) ---
  if FEATURE_TYPE in ['fusionprot', 'saprot', 'protssn', 'prosst']:
    print(f"Standardizing {FEATURE_TYPE.capitalize()} embeddings (Fitting on Train)...")
    scaler = StandardScaler()
    
    # Fill any potential NaNs in the raw data with 0 before scaling
    X_train_raw = np.nan_to_num(X_train_raw)
    X_val_raw = np.nan_to_num(X_val_raw)
    X_test_raw = np.nan_to_num(X_test_raw)

    X_train_raw = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val_raw = scaler.transform(X_val_raw).astype(np.float32)
    X_test_raw = scaler.transform(X_test_raw).astype(np.float32)
    
    # Catch any NaNs introduced BY the scaler (e.g., zero-variance features divided by zero)
    X_train_raw = np.nan_to_num(X_train_raw)
    X_val_raw = np.nan_to_num(X_val_raw)
    X_test_raw = np.nan_to_num(X_test_raw)

    scaler_path = MODELS_DIR / f'{FEATURE_TYPE}_scaler.joblib'
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler to {scaler_path}")

  # --- 6. Decode Labels ---
  train_orig, y_train_bin, y_train_3c = decode_labels(y_train_raw)
  val_orig,   y_val_bin,   y_val_3c   = decode_labels(y_val_raw)
  test_orig,  y_test_bin,  y_test_3c  = decode_labels(y_test_raw)

  # Pre-align state indices for the custom wrapper
  test_states_idx = np.array([SS_CLASSES.index(char) for char in test_orig])

  # --- Outer Loop for Weight Strategies ---
  for ws in strategies_to_run:
    print(f"\n===================================================================")
    print(f" STARTING EXECUTION PIPELINE FOR WEIGHT STRATEGY: {ws.upper()}")
    print(f"===================================================================")
    
    if 'global_binary' in tasks_to_run:
      tune_and_evaluate(
        X_train_raw, y_train_bin, X_val_raw, y_val_bin, X_test_raw, y_test_bin, 
        "Global_Binary", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, ws
      )
    
    if 'global_3class' in tasks_to_run:
      run_multi_class_suite(
        X_train_raw, y_train_3c, X_val_raw, y_val_3c, X_test_raw, y_test_3c, 
        "Global_3Class", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, ws, mode=MULTI_CLASS_MODE
      )

    if 'input_dep_binary' in tasks_to_run or 'input_dep_3class' in tasks_to_run:
      collected_bin_models = {}
      collected_3c_models = {'CommonC_OvO': {}, 'CommonC_OvR': {}, 'IndivOvR': {}, 'IndivOvO': {}}

      for state_idx, state_char in enumerate(SS_CLASSES):
        m_train = (train_orig == state_char)
        m_val   = (val_orig == state_char)
        m_test  = (test_orig == state_char)
        
        if np.sum(m_train) == 0:
          print(f"\nSkipping Input-Dependent {state_char} (No training samples)")
          continue
            
        if 'input_dep_binary' in tasks_to_run:
          bin_mdl = tune_and_evaluate(
            X_train_raw[m_train], y_train_bin[m_train],
            X_val_raw[m_val],     y_val_bin[m_val],
            X_test_raw[m_test],   y_test_bin[m_test],
            f"InputDep_{state_char}_Binary",
            train_orig[m_train], y_train_raw[m_train], val_orig[m_val], test_orig[m_test], y_test_raw[m_test], ws
          )
          collected_bin_models[state_idx] = bin_mdl
        
        if 'input_dep_3class' in tasks_to_run:
          mc_mdls = run_multi_class_suite(
            X_train_raw[m_train], y_train_3c[m_train], 
            X_val_raw[m_val],     y_val_3c[m_val], 
            X_test_raw[m_test],   y_test_3c[m_test], 
            f"InputDep_{state_char}_3Class",
            train_orig[m_train], y_train_raw[m_train], val_orig[m_val], test_orig[m_test], y_test_raw[m_test], ws, mode=MULTI_CLASS_MODE
          )
          for variant, variant_models in mc_mdls.items():
            collected_3c_models[variant][state_idx] = variant_models

      if 'input_dep_binary' in tasks_to_run and collected_bin_models:
        print("\n--- Constructing Unified Input-Dependent Binary Model ---")
        combined_f1 = StateRoutedCombinedClassifier(collected_bin_models, test_states_idx)
        
        evaluate_and_summarize(
          combined_f1, X_test_raw, y_test_bin, True, [0, 1], 
          "Combined_InputDep_Binary", 
          test_orig=test_orig, y_test_raw=y_test_raw, weight_strategy=ws, force_eval=True
        )

      if 'input_dep_3class' in tasks_to_run:
        for variant, state_dict in collected_3c_models.items():
          if state_dict:
            print(f"\n--- Constructing Unified Input-Dependent 3-Class Model ({variant}) ---")
            combined_f1 = StateRoutedCombinedClassifier(state_dict, test_states_idx)
            
            evaluate_and_summarize(
              combined_f1, X_test_raw, y_test_3c, False, [0, 1, 2], 
              f"Combined_InputDep_3Class_{variant}", 
              test_orig=test_orig, y_test_raw=y_test_raw, weight_strategy=ws, force_eval=True
            )