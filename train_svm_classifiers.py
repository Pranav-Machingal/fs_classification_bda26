import numpy as np
import pandas as pd
import pickle
import joblib
from joblib import Parallel, delayed

import sklearn
sklearn.set_config(enable_metadata_routing=True)

from cuml.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.calibration import CalibratedClassifierCV

from config import get_args, setup_directories, SS_CLASSES
from data_loader import load_and_preprocess_data
from metrics import compute_test_metrics, compute_switch_f1, decode_labels
from tuning_utils import get_weights, get_log_grid, get_linear_grid
from models import StateRoutedCombinedClassifier

def save_evaluation(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, dirs):
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

  prefix = f'{task_name}_binary_best_f1'
  
  joblib.dump(best_model, dirs['models'] / f'{prefix}_model.joblib')
    
  with open(dirs['curves'] / f'{prefix}_curves.pkl', 'wb') as f:
    pickle.dump(curves, f)
  np.save(dirs['metrics'] / f'{prefix}_confusion_matrix.npy', cm)
  
  return metrics_dict

def evaluate_and_summarize(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, dirs, force_eval=False):
  summary_path = dirs['metrics'] / f'{task_name}_binary_test_summary.pkl'
  if summary_path.exists() and not force_eval:
    print(f"  -> Skipping Evaluation for {task_name} (Already computed)")
    return

  print(f"  -> Testing Best Switch-F1 Model for {task_name}")
  metrics = save_evaluation(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, dirs)
  
  summary = {'best_f1_model_test_metrics': metrics}
  with open(summary_path, 'wb') as f:
    pickle.dump(summary, f)

def _build_model(params, multi_strategy):
  import sklearn
  sklearn.set_config(enable_metadata_routing=True)
  from cuml.svm import SVC, LinearSVC
  from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
  from sklearn.calibration import CalibratedClassifierCV
  
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

def _evaluate_single_config(params, X_train, y_train, X_val, y_val, y_train_bin_raw, train_orig, val_orig, is_binary, multi_strategy):
  from metrics import compute_switch_f1
  from tuning_utils import get_weights
  
  model = _build_model(params, multi_strategy)
  sample_weights = get_weights(y_train_bin_raw, params['a'])
  
  try:
    model.fit(X_train, y_train, sample_weight=sample_weights)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_f1 = compute_switch_f1(y_train, y_train_pred, train_orig, is_binary)
    val_f1 = compute_switch_f1(y_val, y_val_pred, val_orig, is_binary)
  except Exception as e:
    train_f1 = 0.0
    val_f1 = 0.0
    print(f"Warning: model fit failed for params {params}: {e}")
    
  params['train_f1'] = train_f1
  params['val_f1'] = val_f1
  return params

def tune_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, task_name, train_orig, y_train_raw, val_orig, test_orig, y_test_raw, dirs, multi_strategy=None, n_jobs=1, R=4):
  summary_path = dirs['metrics'] / f'{task_name}_binary_test_summary.pkl'
  tuning_csv_path = dirs['tuning'] / f'{task_name}_binary_tuning.csv'
  f1_model_path = dirs['models'] / f'{task_name}_binary_best_f1_model.joblib'
  
  is_binary = (multi_strategy is None)
  classes = [0, 1] if is_binary else [0, 1, 2]
  
  if summary_path.exists() and f1_model_path.exists():
    print(f"\n--- Skipping Tuning: {task_name} (Already computed) ---")
    return joblib.load(f1_model_path)

  print(f"\n--- Tuning via Dynamic {R}-Stage Search: {task_name} (n_jobs={n_jobs}) ---")

  _, y_train_bin_raw, _ = decode_labels(y_train_raw)
  n_plus = np.sum(y_train_bin_raw == 1)
  n_total = len(y_train_bin_raw)
  a_bar = float(n_plus) / n_total if n_total > 0 else 0.5
  if a_bar == 0: a_bar = 0.01

  configs = [
    {'kernel': 'linear'},
    {'kernel': 'rbf'},
    {'kernel': 'poly', 'degree': 2},
    {'kernel': 'poly', 'degree': 3}
  ]
  
  history = []
  best_global_val_f1 = -1
  best_global_params = None
  
  evaluated_configs = {} # Cache to prevent redundant recomputations

  for config in configs:
    best_c_exp = 0.0
    best_g_exp = 0.0
    best_a = a_bar
    
    step_c = 2.0
    step_g = 2.0
    step_a = a_bar / 2.0
    
    for r in range(1, R + 1):
      c_grid = [best_c_exp - step_c, best_c_exp, best_c_exp + step_c]
      if config['kernel'] in ['rbf', 'poly']:
        g_grid = [best_g_exp - step_g, best_g_exp, best_g_exp + step_g]
      else:
        g_grid = [None]
        
      a_grid_raw = [best_a - step_a, best_a, best_a + step_a]
      a_grid = [a for a in a_grid_raw if 0 < a < 1]
      
      expansions = 0
      max_expansions = 10
      best_r_val_f1 = -1
      best_r_params = None
      
      while expansions < max_expansions:
        all_params_to_eval = []
        all_params_in_grid = []
        
        for c_exp in c_grid:
          for g_exp in g_grid:
            for a_val in a_grid:
              params = config.copy()
              params['C'] = 10 ** c_exp
              params['c_exp'] = c_exp
              params['a'] = a_val
              params['round'] = r
              if g_exp is not None:
                params['gamma'] = 10 ** g_exp
                params['g_exp'] = g_exp
                
              sig = (params['kernel'], params.get('degree', -1), round(c_exp, 5), round(g_exp if g_exp is not None else 0, 5), round(a_val, 6))
              params['sig'] = sig
              all_params_in_grid.append(params)
              
              if sig not in evaluated_configs:
                all_params_to_eval.append(params)
                
        # Execute the grid manually via joblib loky backend for context isolation (only for unseen configs)
        if all_params_to_eval:
          results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_evaluate_single_config)(
              p, X_train, y_train, X_val, y_val, y_train_bin_raw, train_orig, val_orig, is_binary, multi_strategy
            ) for p in all_params_to_eval
          )
          
          for res_params in results:
            sig = res_params.pop('sig')
            evaluated_configs[sig] = res_params
            history.append(res_params)
            
        # Get scores for the current grid
        scores = [evaluated_configs[p['sig']]['val_f1'] for p in all_params_in_grid]
        std_dev = np.std(scores) if len(scores) > 1 else 0.0
        
        # Find best in current grid
        best_p = max(all_params_in_grid, key=lambda p: evaluated_configs[p['sig']]['val_f1'])
        best_r_val_f1 = evaluated_configs[best_p['sig']]['val_f1']
        best_r_params = evaluated_configs[best_p['sig']]
        
        expanded = False
        
        # Condition 2: Plateau Escape
        if std_dev < 0.01:
          min_c = min(c_grid)
          max_c = max(c_grid)
          c_grid.extend([min_c - step_c, max_c + step_c])
          
          if config['kernel'] in ['rbf', 'poly']:
            min_g = min(g_grid)
            max_g = max(g_grid)
            g_grid.extend([min_g - step_g, max_g + step_g])
            
          min_a = min(a_grid)
          max_a = max(a_grid)
          if min_a - step_a > 0: a_grid.append(min_a - step_a)
          if max_a + step_a < 1: a_grid.append(max_a + step_a)
          
          expanded = True
        else:
          # Condition 3: Directional Boundary Expansion
          best_c = best_p['c_exp']
          best_g = best_p.get('g_exp', None)
          best_a_val = best_p['a']
          
          if best_c == min(c_grid):
            c_grid.append(best_c - step_c)
            expanded = True
          elif best_c == max(c_grid):
            c_grid.append(best_c + step_c)
            expanded = True
            
          if config['kernel'] in ['rbf', 'poly']:
            if best_g == min(g_grid):
              g_grid.append(best_g - step_g)
              expanded = True
            elif best_g == max(g_grid):
              g_grid.append(best_g + step_g)
              expanded = True
              
          if best_a_val == min(a_grid) and best_a_val - step_a > 0:
            a_grid.append(best_a_val - step_a)
            expanded = True
          elif best_a_val == max(a_grid) and best_a_val + step_a < 1:
            a_grid.append(best_a_val + step_a)
            expanded = True
            
        if expanded:
          c_grid = sorted(list(set(c_grid)))
          if g_grid[0] is not None: g_grid = sorted(list(set(g_grid)))
          a_grid = sorted(list(set(a_grid)))
          expansions += 1
        else:
          # Strictly internal best point, and variance is good. Round complete.
          break
          
      if best_r_params is not None:
        best_c_exp = best_r_params['c_exp']
        best_a = best_r_params['a']
        if 'g_exp' in best_r_params:
          best_g_exp = best_r_params['g_exp']
          
      # Refine step sizes for next round
      step_c /= 2.0
      step_g /= 2.0
      step_a /= 2.0
          
    if best_global_params is None or best_r_val_f1 > best_global_val_f1:
      best_global_val_f1 = best_r_val_f1
      best_global_params = best_r_params

  print(f"\n  -> Champion Configuration (Switch-F1: {best_global_val_f1:.4f}): {best_global_params}")
  
  df = pd.DataFrame(history)
  df.to_csv(tuning_csv_path, index=False)
  
  best_model = _build_model(best_global_params, multi_strategy)
  sample_weights = get_weights(y_train_bin_raw, best_global_params['a'])
  best_model.fit(X_train, y_train, sample_weight=sample_weights)
  
  evaluate_and_summarize(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, dirs, force_eval=True)
  
  return best_model

def run_multi_class_suite(X_train, y_train_3c, X_val, y_val_3c, X_test, y_test_3c, base_name, train_orig, y_train_raw, val_orig, test_orig, y_test_raw, dirs, n_jobs=1, R=4):
  mdls = {}
  mdls['CommonC_OvO'] = tune_and_evaluate(X_train, y_train_3c, X_val, y_val_3c, X_test, y_test_3c, f"{base_name}_CommonC_OvO", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, dirs, multi_strategy='ovo', n_jobs=n_jobs, R=R)
  mdls['CommonC_OvR'] = tune_and_evaluate(X_train, y_train_3c, X_val, y_val_3c, X_test, y_test_3c, f"{base_name}_CommonC_OvR", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, dirs, multi_strategy='ovr', n_jobs=n_jobs, R=R)
  return mdls

if __name__ == "__main__":
  args = get_args()
  dirs = setup_directories(args.feature_type, args.k)
  
  data = load_and_preprocess_data(args.feature_type, args.k, dirs['models'])
  
  X_train_raw, y_train_raw, y_train_bin, y_train_3c, train_orig = data['X_train'], data['y_train_raw'], data['y_train_bin'], data['y_train_3c'], data['train_orig']
  X_val_raw, y_val_raw, y_val_bin, y_val_3c, val_orig = data['X_val'], data['y_val_raw'], data['y_val_bin'], data['y_val_3c'], data['val_orig']
  X_test_raw, y_test_raw, y_test_bin, y_test_3c, test_orig, test_states_idx = data['X_test'], data['y_test_raw'], data['y_test_bin'], data['y_test_3c'], data['test_orig'], data['test_states_idx']

  print(f"\n===================================================================")
  print(f" STARTING EXECUTION PIPELINE FOR WEIGHT STRATEGY: BINARY")
  print(f"===================================================================")
  
  if 'global_binary' in args.tasks:
    tune_and_evaluate(
      X_train_raw, y_train_bin, X_val_raw, y_val_bin, X_test_raw, y_test_bin, 
      "Global_Binary", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, dirs,
      n_jobs=args.n_jobs, R=args.R
    )
  
  if 'global_3class' in args.tasks:
    run_multi_class_suite(
      X_train_raw, y_train_3c, X_val_raw, y_val_3c, X_test_raw, y_test_3c, 
      "Global_3Class", train_orig, y_train_raw, val_orig, test_orig, y_test_raw, dirs,
      n_jobs=args.n_jobs, R=args.R
    )

  if 'input_dep_binary' in args.tasks or 'input_dep_3class' in args.tasks:
    collected_bin_models = {}
    collected_3c_models = {'CommonC_OvO': {}, 'CommonC_OvR': {}}

    for state_idx, state_char in enumerate(SS_CLASSES):
      m_train = (train_orig == state_char)
      m_val   = (val_orig == state_char)
      m_test  = (test_orig == state_char)
      
      if np.sum(m_train) == 0:
        print(f"\nSkipping Input-Dependent {state_char} (No training samples)")
        continue
          
      if 'input_dep_binary' in args.tasks:
        bin_mdl = tune_and_evaluate(
          X_train_raw[m_train], y_train_bin[m_train],
          X_val_raw[m_val],     y_val_bin[m_val],
          X_test_raw[m_test],   y_test_bin[m_test],
          f"InputDep_{state_char}_Binary",
          train_orig[m_train], y_train_raw[m_train], val_orig[m_val], test_orig[m_test], y_test_raw[m_test], dirs,
          n_jobs=args.n_jobs, R=args.R
        )
        collected_bin_models[state_idx] = bin_mdl
      
      if 'input_dep_3class' in args.tasks:
        mc_mdls = run_multi_class_suite(
          X_train_raw[m_train], y_train_3c[m_train], 
          X_val_raw[m_val],     y_val_3c[m_val], 
          X_test_raw[m_test],   y_test_3c[m_test], 
          f"InputDep_{state_char}_3Class",
          train_orig[m_train], y_train_raw[m_train], val_orig[m_val], test_orig[m_test], y_test_raw[m_test], dirs,
          n_jobs=args.n_jobs, R=args.R
        )
        for variant, variant_models in mc_mdls.items():
          collected_3c_models[variant][state_idx] = variant_models

    if 'input_dep_binary' in args.tasks and collected_bin_models:
      print("\n--- Constructing Unified Input-Dependent Binary Model ---")
      combined_f1 = StateRoutedCombinedClassifier(collected_bin_models, test_states_idx)
      
      evaluate_and_summarize(
        combined_f1, X_test_raw, y_test_bin, True, [0, 1], 
        "Combined_InputDep_Binary", 
        test_orig=test_orig, y_test_raw=y_test_raw, dirs=dirs, force_eval=True
      )

    if 'input_dep_3class' in args.tasks:
      for variant, state_dict in collected_3c_models.items():
        if state_dict:
          print(f"\n--- Constructing Unified Input-Dependent 3-Class Model ({variant}) ---")
          combined_f1 = StateRoutedCombinedClassifier(state_dict, test_states_idx)
          
          evaluate_and_summarize(
            combined_f1, X_test_raw, y_test_3c, False, [0, 1, 2], 
            f"Combined_InputDep_3Class_{variant}", 
            test_orig=test_orig, y_test_raw=y_test_raw, dirs=dirs, force_eval=True
          )