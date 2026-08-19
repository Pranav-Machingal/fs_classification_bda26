import glob
import os
import pickle
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from metrics import compute_switch_f1
from data_loader import load_and_preprocess_data
from models import StateRoutedCombinedClassifier
from config import SS_CLASSES
import cuml

# Silence cuML warnings during massive inference loops
from cuml.common import logger
logger.set_level(logger.level_enum.error)

import __main__
from train_svm_classifiers import CuMLWrapper
__main__.CuMLWrapper = CuMLWrapper

def extract_all_scores(output_csv):
  os.makedirs(os.path.dirname(output_csv), exist_ok=True)
  base_exp_dir = Path('saved_data/svm_experiments')
  results = []
  
  # Cache datasets to avoid reloading massive numpy arrays repeatedly
  loaded_data = {}
  
  for pkl_file in base_exp_dir.rglob('*_test_summary.pkl'):
    parts = pkl_file.parts
    metrics_idx = parts.index('metrics')
    feature_type = "/".join(parts[2:metrics_idx])
    
    task_name = pkl_file.name.replace('_test_summary.pkl', '')
    
    with open(pkl_file, 'rb') as f:
      summary = pickle.load(f)
    
    metrics = summary.get('best_f1_model_test_metrics', {})
    switch_p = metrics.get('switch_precision', 0)
    switch_r = metrics.get('switch_recall', 0)
    test_f1 = 2 * (switch_p * switch_r) / (switch_p + switch_r) if (switch_p + switch_r) > 0 else 0.0
    
    tuning_dir = pkl_file.parent.parent / 'tuning'
    tuning_path = tuning_dir / f'{task_name}_tuning.csv'
    train_f1 = 'N/A'
    val_f1 = 'N/A'
    
    if tuning_path.exists():
      try:
        df = pd.read_csv(tuning_path)
        if 'user_attrs_val_f1' in df.columns:
          best_row = df.sort_values('user_attrs_val_f1', ascending=False).iloc[0]
          train_f1 = best_row['user_attrs_train_f1']
          val_f1 = best_row['user_attrs_val_f1']
        elif 'val_f1' in df.columns:
          best_row = df.sort_values('val_f1', ascending=False).iloc[0]
          train_f1 = best_row['train_f1']
          val_f1 = best_row['val_f1']
      except:
        pass
        
    is_binary = 'Binary' in task_name
    
    # Check if we need to manually compute Train/Val F1
    if train_f1 == 'N/A' or train_f1 == -1.0 or val_f1 == 'N/A' or val_f1 == -1.0:
      print(f"Computing missing scores via inference for {task_name} ({feature_type})...")
      
      # 1. Lazily load dataset
      if feature_type not in loaded_data:
        k_val = 3
        base_ft = feature_type
        if 'k_' in feature_type:
          base_ft = feature_type.split('/')[0]
          k_val = int(feature_type.split('_')[-1])
        models_dir = base_exp_dir / feature_type / 'models'
        os.makedirs(models_dir, exist_ok=True)
        loaded_data[feature_type] = load_and_preprocess_data(base_ft, k_val, models_dir)
          
      data = loaded_data[feature_type]
      model_dir = base_exp_dir / feature_type / 'models'
      model = None
      
      # 2. Load Models
      if 'Combined_InputDep' in task_name:
        base_models = {}
        for char in ['a', 'b', 'c']:
          state_idx = SS_CLASSES.index(char)
          sub_task = task_name.replace('Combined_', '')
          sub_task = sub_task.replace('InputDep_', f'InputDep_{char}_')
          sub_model_path = model_dir / f'{sub_task}_best_f1_model.joblib'
          
          if sub_model_path.exists():
            base_models[state_idx] = joblib.load(sub_model_path)
            
        if base_models:
          model = StateRoutedCombinedClassifier(base_models, test_states=None)
        else:
          print(f"  -> Skipping {task_name}, base ensemble models not found.")
      else:
        model_path = model_dir / f'{task_name}_best_f1_model.joblib'
        if model_path.exists():
          model = joblib.load(model_path)
        else:
          print(f"  -> Skipping {task_name}, model not found at {model_path}.")
          
      # 3. Perform Inference
      if model is not None:
        train_states = np.array([SS_CLASSES.index(char) for char in data['train_orig']])
        val_states = np.array([SS_CLASSES.index(char) for char in data['val_orig']])
        
        y_train_true = data['y_train_bin'] if is_binary else data['y_train_3c']
        y_val_true = data['y_val_bin'] if is_binary else data['y_val_3c']
        
        # Train Inference
        if hasattr(model, 'test_states'):
          model.test_states = train_states
        y_train_pred = model.predict(data['X_train'])
        if hasattr(y_train_pred, 'get'): y_train_pred = y_train_pred.get()
        elif hasattr(y_train_pred, 'to_numpy'): y_train_pred = y_train_pred.to_numpy()
        
        train_f1 = compute_switch_f1(y_train_true, y_train_pred, data['train_orig'], is_binary)
        
        # Val Inference
        if hasattr(model, 'test_states'):
          model.test_states = val_states
        y_val_pred = model.predict(data['X_val'])
        if hasattr(y_val_pred, 'get'): y_val_pred = y_val_pred.get()
        elif hasattr(y_val_pred, 'to_numpy'): y_val_pred = y_val_pred.to_numpy()
        
        val_f1 = compute_switch_f1(y_val_true, y_val_pred, data['val_orig'], is_binary)
            
    results.append({
      'Feature_Type': feature_type,
      'Task': task_name,
      'Train_Switch_F1': train_f1,
      'Val_Switch_F1': val_f1,
      'Test_Switch_F1': test_f1
    })

  if results:
    df_res = pd.DataFrame(results)
    
    # Filter out individual InputDep classifiers (keep Combined and Global)
    df_res = df_res[~df_res['Task'].str.startswith('InputDep_')]
    
    df_res = df_res.sort_values(by=['Feature_Type', 'Task'])
    
    for col in ['Train_Switch_F1', 'Val_Switch_F1', 'Test_Switch_F1']:
      df_res[col] = df_res[col].apply(lambda x: round(x, 4) if isinstance(x, (float, int)) else x)
        
    df_res.to_csv(output_csv, index=False)
    print(f"\nSuccessfully saved {len(df_res)} scores to {output_csv}")
  else:
    print("No test_summary.pkl files found.")

if __name__ == '__main__':
  extract_all_scores('saved_results/all_classifier_scores.csv')
