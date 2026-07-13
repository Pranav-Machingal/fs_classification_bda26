"""
SVM Classifier Training and Tuning Module

This module implements the training, hyperparameter optimization, and evaluation of Support Vector Machine (SVM) 
models for fold-switch prediction. It supports both Binary and Multi-class (Granular) paradigms, as well as 
Global (Input-Agnostic) and Input-Dependent routing architectures. 

Hyperparameter tuning is powered by Optuna using the Tree-structured Parzen Estimator (TPE) algorithm. 
The tuning explicitly optimizes regularisation (C), kernel-specific parameters (gamma, degree, coef0), 
and a continuous class-imbalance weight parameter (a).

To bypass cuML's memory limitations (where multiple fit calls cause GPU context starvation), the Optuna objective
offloads model training and scoring to isolated child processes using `multiprocessing` with the 'fork' context.
"""
import sys
import numpy as np
import pandas as pd
import pickle
import joblib
import optuna
import multiprocessing as mp
import gc

import sklearn

sklearn.set_config(enable_metadata_routing=True)

from config import get_args, setup_directories, SS_CLASSES
from data_loader import load_and_preprocess_data
from metrics import compute_test_metrics, compute_switch_f1, decode_labels
from tuning_utils import get_weights
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from cuml.svm import SVC, LinearSVC

from models import StateRoutedCombinedClassifier


from sklearn.base import BaseEstimator, ClassifierMixin

# --- CuML Scikit-Learn Wrapper ---
# This wrapper enables compatibility between cuML SVM models and scikit-learn ensemble/multiclass wrappers
# by explicitly intercepting `predict_proba` and `decision_function` property calls, and explicitly
# propagating sample weights during fit.
class CuMLWrapper(BaseEstimator, ClassifierMixin):
  def __init__(self, estimator=None):
    self.estimator = estimator

  def fit(self, X, y, sample_weight=None):
    self.estimator.fit(X, y, sample_weight=sample_weight)
    if hasattr(self.estimator, 'classes_'):
      self.classes_ = self.estimator.classes_
    return self

  def predict(self, X):
    return self.estimator.predict(X)

  @property
  def predict_proba(self):
    if not hasattr(self.estimator, 'predict_proba'):
      raise AttributeError
    return self.estimator.predict_proba

  @property
  def decision_function(self):
    if not hasattr(self.estimator, 'decision_function'):
      raise AttributeError
    return self.estimator.decision_function


class LazyLoadedModel:

  def __init__(self, path):
    self.path = path

  def predict(self, X):
    model = joblib.load(self.path)
    res = model.predict(X)
    del model
    import gc
    gc.collect()
    return res

  def predict_proba(self, X):
    model = joblib.load(self.path)
    res = model.predict_proba(X)
    del model
    import gc
    gc.collect()
    return res

  def decision_function(self, X):
    model = joblib.load(self.path)
    res = model.decision_function(X)
    del model
    import gc
    gc.collect()
    return res


# --- GLOBAL DATA FOR FORK WORKERS ---
G_X_TRAIN = None
G_Y_TRAIN = None
G_X_VAL = None
G_Y_VAL = None
G_TRAIN_ORIG = None
G_VAL_ORIG = None
G_Y_TRAIN_BIN_RAW = None


def _get_cuml_model_for_tuning(params, multi_strategy):
  """
  Constructs the base cuML SVM and conditionally wraps it in OneVsOne or OneVsRest classifiers
  if a multi-class strategy is specified.
  """
  if params['kernel'] == 'linear':
    base_svm = LinearSVC(C=params['C'], max_iter=10000)
  else:
    base_svm = SVC(C=params['C'],
                   kernel=params['kernel'],
                   degree=params.get('degree', 3),
                   gamma=params.get('gamma', 'scale'),
                   coef0=params.get('coef0', 0.0),
                   max_iter=10000,
                   random_state=42,
                   cache_size=4096)

  if multi_strategy in ['ovo', 'ovr']:
    wrapped_base = CuMLWrapper(estimator=base_svm)
    if hasattr(wrapped_base, 'set_fit_request'):
      wrapped_base.set_fit_request(sample_weight=True)

    if multi_strategy == 'ovo':
      return OneVsOneClassifier(wrapped_base, n_jobs=1)
    else:
      return OneVsRestClassifier(wrapped_base, n_jobs=1)
  else:
    return base_svm


def worker_fit_and_score(params, multi_strategy, is_binary, r_plus, r_minus,
                         child_conn):
  """
  Isolated worker function executed in a dedicated forked process.
  Trains the SVM model on the global training arrays and computes the validation Switch-F1 score.
  Prevents cuML GPU context memory leaks during exhaustive Optuna tuning.
  """
  try:
    import sklearn
    sklearn.set_config(enable_metadata_routing=True)
    sample_weights = np.where(G_Y_TRAIN_BIN_RAW == 1, r_plus, r_minus)
    model = _get_cuml_model_for_tuning(params, multi_strategy)

    model.fit(G_X_TRAIN, G_Y_TRAIN, sample_weight=sample_weights)

    y_train_pred = model.predict(G_X_TRAIN)
    y_val_pred = model.predict(G_X_VAL)

    if hasattr(y_train_pred, 'get'):
      y_train_pred = y_train_pred.get()
    elif hasattr(y_train_pred, 'to_numpy'):
      y_train_pred = y_train_pred.to_numpy()
    if hasattr(y_val_pred, 'get'):
      y_val_pred = y_val_pred.get()
    elif hasattr(y_val_pred, 'to_numpy'):
      y_val_pred = y_val_pred.to_numpy()

    train_f1 = compute_switch_f1(G_Y_TRAIN, y_train_pred, G_TRAIN_ORIG,
                                 is_binary)
    val_f1 = compute_switch_f1(G_Y_VAL, y_val_pred, G_VAL_ORIG, is_binary)

    child_conn.send({'train_f1': float(train_f1), 'val_f1': float(val_f1)})
  except Exception as e:
    child_conn.send({'error': str(e)})
  finally:
    child_conn.close()


def objective(trial, kernel, fixed_degree, multi_strategy, is_binary):
  """
  Optuna objective function for tuning SVM hyperparameters.
  Samples C, gamma, coef0, and class weight parameter (a) from continuous probabilistic distributions.
  """
  C = trial.suggest_float("C", 1e-6, 1e3, log=True)
  params = {'kernel': kernel, 'C': C}

  if kernel == 'rbf':
    params['gamma'] = trial.suggest_float("gamma", 1e-6, 1e2, log=True)
  elif kernel == 'poly':
    params['gamma'] = trial.suggest_float("gamma", 1e-6, 1e-1, log=True)
    params['coef0'] = trial.suggest_float("coef0", 0.0, 1.0)
    params['degree'] = fixed_degree
  else:
    params['gamma'] = 'scale'

  a = trial.suggest_float("a", 0.0001, 0.9999)
  params['a'] = a

  r_plus = 1.0 / (2.0 * a)
  r_minus = 1.0 / (2.0 * (1.0 - a))

  parent_conn, child_conn = mp.Pipe()

  ctx = mp.get_context('fork')
  p = ctx.Process(target=worker_fit_and_score,
                  args=(params, multi_strategy, is_binary, r_plus, r_minus,
                        child_conn))
  p.start()

  child_conn.close()  # Close in parent to prevent fd leak

  p.join()

  if parent_conn.poll():
    return_dict = parent_conn.recv()
  else:
    return_dict = {'error': 'Process crashed or died out of memory.'}

  parent_conn.close()

  if 'error' in return_dict:
    label = f"{kernel}(d={fixed_degree})" if fixed_degree else kernel
    print(f"  Warning: {label} trial failed: {return_dict['error']}")
    return -1.0

  train_f1 = return_dict['train_f1']
  val_f1 = return_dict['val_f1']

  gap = train_f1 - val_f1

  trial.set_user_attr("train_f1", train_f1)
  trial.set_user_attr("val_f1", val_f1)
  trial.set_user_attr("gap", gap)

  # Return regular Switch-F1 instead of penalized score
  return val_f1


def tune_kernel_study(kernel_spec, multi_strategy, is_binary, n_trials,
                      db_path_str, study_name, n_jobs):
  if isinstance(kernel_spec, tuple):
    kernel, fixed_degree = kernel_spec
  else:
    kernel, fixed_degree = kernel_spec, None

  study = optuna.create_study(study_name=study_name,
                              storage=f"sqlite:///{db_path_str}",
                              load_if_exists=True,
                              direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))

  study.optimize(lambda trial: objective(trial, kernel, fixed_degree,
                                         multi_strategy, is_binary),
                 n_trials=n_trials,
                 n_jobs=n_jobs)

  label = f"{kernel}(d={fixed_degree})" if fixed_degree else kernel
  print(f"  -> {label}: Best Switch-F1 = {study.best_value:.4f}")
  return study


def _build_final_model(params, multi_strategy):
  if params['kernel'] == 'linear':
    base_svm = LinearSVC(C=params['C'], max_iter=10000)
  else:
    base_svm = SVC(C=params['C'],
                   kernel=params['kernel'],
                   degree=params.get('degree', 3),
                   gamma=params.get('gamma', 'scale'),
                   coef0=params.get('coef0', 0.0),
                   random_state=42,
                   max_iter=10000,
                   probability=True,
                   cache_size=4096)

  if multi_strategy in ['ovo', 'ovr']:
    wrapped_base = CuMLWrapper(estimator=base_svm)
    if hasattr(wrapped_base, 'set_fit_request'):
      wrapped_base.set_fit_request(sample_weight=True)

    if multi_strategy == 'ovo':
      return OneVsOneClassifier(wrapped_base, n_jobs=1)
    else:
      return OneVsRestClassifier(wrapped_base, n_jobs=1)
  else:
    return base_svm


def evaluate_and_summarize(best_model,
                           X_test,
                           y_test,
                           is_binary,
                           classes,
                           task_name,
                           test_orig,
                           y_test_raw,
                           dirs,
                           force_eval=False):
  summary_path = dirs['metrics'] / f'{task_name}_test_summary.pkl'
  if summary_path.exists() and not force_eval:
    print(f"  -> Skipping Evaluation for {task_name} (Already computed)")
    return

  print(f"  -> Testing Best Switch-F1 Model for {task_name}")

  y_test_pred = best_model.predict(X_test)

  y_score = None
  
  if hasattr(best_model, "predict_proba"):
    try:
      y_score = best_model.predict_proba(X_test)
      if is_binary and getattr(y_score, 'ndim', len(getattr(y_score, 'shape', []))) == 2:
        y_score = y_score[:, 1]
    except AttributeError:
      pass

  if y_score is None and hasattr(best_model, "decision_function"):
    try:
      y_score = best_model.decision_function(X_test)
    except AttributeError:
      pass
      
  if y_score is None:
    y_score = y_test_pred

  if hasattr(y_test_pred, 'get'):
    y_test_pred = y_test_pred.get()
  elif hasattr(y_test_pred, 'to_numpy'):
    y_test_pred = y_test_pred.to_numpy()
  if hasattr(y_score, 'get'):
    y_score = y_score.get()
  elif hasattr(y_score, 'to_numpy'):
    y_score = y_score.to_numpy()

  metrics, cm, curves = compute_test_metrics(y_test,
                                             y_test_pred,
                                             y_score,
                                             is_binary,
                                             classes,
                                             test_orig=test_orig,
                                             y_test_raw=y_test_raw)

  summary = {'best_f1_model_test_metrics': metrics}
  with open(summary_path, 'wb') as f:
    pickle.dump(summary, f)

  prefix = f'{task_name}_best_f1'
  with open(dirs['metrics'] / f'{prefix}_curves.pkl', 'wb') as f:
    pickle.dump(curves, f)
  np.save(dirs['metrics'] / f'{prefix}_confusion_matrix.npy', cm)


def _fit_and_eval_worker(best_params, multi_strategy, X_train, y_train, sample_weights, f1_model_path, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, dirs):
  import sklearn
  sklearn.set_config(enable_metadata_routing=True)
  best_model = _build_final_model(best_params, multi_strategy)
  best_model.fit(X_train, y_train, sample_weight=sample_weights)
  joblib.dump(best_model, f1_model_path)
  evaluate_and_summarize(best_model, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, dirs, force_eval=True)

# --- Main Tuning and Evaluation Pipeline ---

def tune_and_evaluate(X_train,
                      y_train,
                      X_val,
                      y_val,
                      X_test,
                      y_test,
                      task_name,
                      train_orig,
                      y_train_bin,
                      val_orig,
                      test_orig,
                      y_test_raw,
                      dirs,
                      multi_strategy=None,
                      n_trials=80,
                      max_parallel_kernels=4):
  summary_path = dirs['metrics'] / f'{task_name}_test_summary.pkl'
  tuning_csv_path = dirs['tuning'] / f'{task_name}_tuning.csv'
  f1_model_path = dirs['models'] / f'{task_name}_best_f1_model.joblib'

  is_binary = (multi_strategy is None)
  classes = [0, 1] if is_binary else [0, 1, 2]

  if summary_path.exists() and f1_model_path.exists():
    print(f"\n--- Skipping Tuning: {task_name} (Already computed) ---")
    return LazyLoadedModel(f1_model_path)

  print(
      f"\n--- Tuning via Optuna: {task_name} ({n_trials} trials/kernel, max {max_parallel_kernels} concurrent processes) ---"
  )

  global G_X_TRAIN, G_Y_TRAIN, G_X_VAL, G_Y_VAL, G_TRAIN_ORIG, G_VAL_ORIG, G_Y_TRAIN_BIN_RAW
  G_X_TRAIN = X_train
  G_Y_TRAIN = y_train
  G_X_VAL = X_val
  G_Y_VAL = y_val
  G_TRAIN_ORIG = train_orig
  G_VAL_ORIG = val_orig
  G_Y_TRAIN_BIN_RAW = y_train_bin

  optuna.logging.set_verbosity(optuna.logging.WARNING)

  kernel_specs = ['linear', 'rbf', ('poly', 2), ('poly', 3)]

  best_score = -float('inf')
  best_params = None
  all_trials = []

  for kernel_spec in kernel_specs:
    if isinstance(kernel_spec, tuple):
      kernel, degree = kernel_spec
      label = f"{kernel}_d{degree}"
    else:
      kernel, degree, label = kernel_spec, None, kernel_spec

    db_path = dirs['tuning'] / f"{task_name}_{label}_optuna.db"
    study_name = f"{task_name}_{label}"

    study = tune_kernel_study(kernel_spec, multi_strategy, is_binary, n_trials,
                              str(db_path.absolute()), study_name,
                              max_parallel_kernels)

    completed = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if completed and study.best_value > best_score:
      best_score = study.best_value
      bp = study.best_params.copy()
      bp['kernel'] = kernel
      if degree is not None:
        bp['degree'] = degree
      if 'gamma' not in bp:
        bp['gamma'] = 'scale'
      best_params = bp

    df = study.trials_dataframe()
    df['kernel_type'] = label
    all_trials.append(df)

  if best_params is None:
    raise RuntimeError(
        f"All kernel studies failed for {task_name}. Check logs above.")

  print(
      f"\n  -> Champion Configuration (Switch-F1: {best_score:.4f}): {best_params}"
  )

  if all_trials:
    combined_df = pd.concat(all_trials, ignore_index=True)
    combined_df.to_csv(tuning_csv_path, index=False)

  sample_weights = get_weights(y_train_bin, best_params['a'])
  
  ctx = mp.get_context('fork')
  p = ctx.Process(target=_fit_and_eval_worker, args=(best_params, multi_strategy, X_train, y_train, sample_weights, f1_model_path, X_test, y_test, is_binary, classes, task_name, test_orig, y_test_raw, dirs))
  p.start()
  p.join()
  
  if p.exitcode != 0:
    raise RuntimeError(f"Final model fitting failed for {task_name}")

  return LazyLoadedModel(f1_model_path)


def run_multi_class_suite(X_train,
                          y_train_3c,
                          X_val,
                          y_val_3c,
                          X_test,
                          y_test_3c,
                          base_name,
                          train_orig,
                          y_train_bin,
                          val_orig,
                          test_orig,
                          y_test_raw,
                          dirs,
                          n_trials=80,
                          max_parallel_kernels=4):
  mdls = {}
  mdls['CommonC_OvO'] = tune_and_evaluate(
      X_train,
      y_train_3c,
      X_val,
      y_val_3c,
      X_test,
      y_test_3c,
      f"{base_name}_CommonC_OvO",
      train_orig,
      y_train_bin,
      val_orig,
      test_orig,
      y_test_raw,
      dirs,
      multi_strategy='ovo',
      n_trials=n_trials,
      max_parallel_kernels=max_parallel_kernels)
  mdls['CommonC_OvR'] = tune_and_evaluate(
      X_train,
      y_train_3c,
      X_val,
      y_val_3c,
      X_test,
      y_test_3c,
      f"{base_name}_CommonC_OvR",
      train_orig,
      y_train_bin,
      val_orig,
      test_orig,
      y_test_raw,
      dirs,
      multi_strategy='ovr',
      n_trials=n_trials,
      max_parallel_kernels=max_parallel_kernels)
  return mdls


if __name__ == "__main__":
  args = get_args()
  dirs = setup_directories(args.feature_type, args.k)

  data = load_and_preprocess_data(args.feature_type, args.k, dirs['models'])

  X_train_raw, y_train_bin, y_train_bin, y_train_3c, train_orig = data[
      'X_train'], data['y_train_raw'], data['y_train_bin'], data[
          'y_train_3c'], data['train_orig']
  X_val_raw, y_val_raw, y_val_bin, y_val_3c, val_orig = data['X_val'], data[
      'y_val_raw'], data['y_val_bin'], data['y_val_3c'], data['val_orig']
  X_test_raw, y_test_raw, y_test_bin, y_test_3c, test_orig, test_states_idx = data[
      'X_test'], data['y_test_raw'], data['y_test_bin'], data[
          'y_test_3c'], data['test_orig'], data['test_states_idx']

  # Ensure max_parallel_kernels exists in args (default to 4 if removed from config.py)
  max_parallel_kernels = getattr(args, 'max_parallel_kernels', 4)

  print(
      f"\n===================================================================")
  print(f" STARTING EXECUTION PIPELINE FOR WEIGHT STRATEGY: BINARY")
  print(f"===================================================================")

  if 'global_binary' in args.tasks:
    tune_and_evaluate(X_train_raw,
                      y_train_bin,
                      X_val_raw,
                      y_val_bin,
                      X_test_raw,
                      y_test_bin,
                      "Global_Binary",
                      train_orig,
                      y_train_bin,
                      val_orig,
                      test_orig,
                      y_test_raw,
                      dirs,
                      n_trials=args.n_trials,
                      max_parallel_kernels=max_parallel_kernels)

  if 'global_3class' in args.tasks:
    run_multi_class_suite(X_train_raw,
                          y_train_3c,
                          X_val_raw,
                          y_val_3c,
                          X_test_raw,
                          y_test_3c,
                          "Global_3Class",
                          train_orig,
                          y_train_bin,
                          val_orig,
                          test_orig,
                          y_test_raw,
                          dirs,
                          n_trials=args.n_trials,
                          max_parallel_kernels=max_parallel_kernels)

  if 'input_dep_binary' in args.tasks or 'input_dep_3class' in args.tasks:
    collected_bin_models = {}
    collected_3c_models = {'CommonC_OvO': {}, 'CommonC_OvR': {}}

    for state_idx, state_char in enumerate(SS_CLASSES):
      m_train = (train_orig == state_char)
      m_val = (val_orig == state_char)
      m_test = (test_orig == state_char)

      if np.sum(m_train) == 0:
        print(f"\nSkipping Input-Dependent {state_char} (No training samples)")
        continue

      if 'input_dep_binary' in args.tasks:
        bin_mdl = tune_and_evaluate(X_train_raw[m_train],
                                    y_train_bin[m_train],
                                    X_val_raw[m_val],
                                    y_val_bin[m_val],
                                    X_test_raw[m_test],
                                    y_test_bin[m_test],
                                    f"InputDep_{state_char}_Binary",
                                    train_orig[m_train],
                                    y_train_bin[m_train],
                                    val_orig[m_val],
                                    test_orig[m_test],
                                    y_test_raw[m_test],
                                    dirs,
                                    n_trials=args.n_trials,
                                    max_parallel_kernels=max_parallel_kernels)
        collected_bin_models[state_idx] = bin_mdl

      if 'input_dep_3class' in args.tasks:
        mc_mdls = run_multi_class_suite(
            X_train_raw[m_train],
            y_train_3c[m_train],
            X_val_raw[m_val],
            y_val_3c[m_val],
            X_test_raw[m_test],
            y_test_3c[m_test],
            f"InputDep_{state_char}_3Class",
            train_orig[m_train],
            y_train_bin[m_train],
            val_orig[m_val],
            test_orig[m_test],
            y_test_raw[m_test],
            dirs,
            n_trials=args.n_trials,
            max_parallel_kernels=max_parallel_kernels)
        for variant, variant_models in mc_mdls.items():
          collected_3c_models[variant][state_idx] = variant_models

    if 'input_dep_binary' in args.tasks and collected_bin_models:
      print("\n--- Constructing Unified Input-Dependent Binary Model ---")
      combined_f1 = StateRoutedCombinedClassifier(collected_bin_models,
                                                  test_states_idx)

      evaluate_and_summarize(combined_f1,
                             X_test_raw,
                             y_test_bin,
                             True, [0, 1],
                             "Combined_InputDep_Binary",
                             test_orig=test_orig,
                             y_test_raw=y_test_raw,
                             dirs=dirs,
                             force_eval=True)

    if 'input_dep_3class' in args.tasks:
      for variant, state_dict in collected_3c_models.items():
        if state_dict:
          print(
              f"\n--- Constructing Unified Input-Dependent 3-Class Model ({variant}) ---"
          )
          combined_f1 = StateRoutedCombinedClassifier(state_dict,
                                                      test_states_idx)

          evaluate_and_summarize(combined_f1,
                                 X_test_raw,
                                 y_test_3c,
                                 False, [0, 1, 2],
                                 f"Combined_InputDep_3Class_{variant}",
                                 test_orig=test_orig,
                                 y_test_raw=y_test_raw,
                                 dirs=dirs,
                                 force_eval=True)
