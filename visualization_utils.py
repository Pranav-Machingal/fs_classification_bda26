import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

def plot_comprehensive_evaluation(task_name, metric_focus='f1', feature_type='fusionprot', k=3):
  """
  Visualizes 2x2 CM, Global Binary P/R, Granular Sub-Type P/R, and Hyperparameter 
  tuning curves (dynamically generating separate subplots for combined ensemble models).
  """
  # 1. Path Resolution
  if feature_type == 'fusionprot':
    base_dir = Path('saved_data/svm_experiments/fusionprot')
  elif feature_type == 'saprot':
    base_dir = Path('saved_data/svm_experiments/saprot')
  elif feature_type == 'protssn':
    base_dir = Path('saved_data/svm_experiments/protssn')
  elif feature_type == 'prosst':
    base_dir = Path('saved_data/svm_experiments/prosst')
  elif feature_type == 'kncsic':
    base_dir = Path(f'saved_data/svm_experiments/kncsic/k_{k}')
  elif feature_type == 'kndrf':
    base_dir = Path(f'saved_data/svm_experiments/kndrf/k_{k}')
  else:
    raise ValueError(f"Unknown feature type: {feature_type}")
    
  metrics_dir = base_dir / 'metrics'
  tuning_dir = base_dir / 'tuning'
  
  summary_path = metrics_dir / f'{task_name}_test_summary.pkl'
  
  if not summary_path.exists():
    print(f"Error: Could not find summary file at {summary_path}")
    return
    
  with open(summary_path, 'rb') as f:
    summary = pickle.load(f)
    
  metrics = summary[f'best_{metric_focus}_model_test_metrics']
  
  # 2. Identify Sub-Models for Tuning Curves
  sub_models = []
  if task_name == "Combined_InputDep_Binary":
    sub_models = ["InputDep_a_Binary", "InputDep_b_Binary", "InputDep_c_Binary"]
  elif task_name.startswith("Combined_InputDep_3Class_"):
    variant = task_name.replace("Combined_InputDep_3Class_", "")
    sub_models = [f"InputDep_a_3Class_{variant}", f"InputDep_b_3Class_{variant}", f"InputDep_c_3Class_{variant}"]
  elif task_name.endswith("_IndivOvR_Combined"):
    base = task_name.replace("_IndivOvR_Combined", "")
    sub_models = [f"{base}_IndivOvR_Class0vsRest", f"{base}_IndivOvR_Class1vsRest", f"{base}_IndivOvR_Class2vsRest"]
  elif task_name.endswith("_IndivOvO_Combined"):
    base = task_name.replace("_IndivOvO_Combined", "")
    sub_models = [f"{base}_IndivOvO_Pair0vs1", f"{base}_IndivOvO_Pair0vs2", f"{base}_IndivOvO_Pair1vs2"]
  else:
    sub_models = [task_name] # Single base model

  n_tune_plots = len(sub_models)
  
  # 3. Initialize Advanced Grid Layout
  fig = plt.figure(figsize=(24, 6.5))
  gs = fig.add_gridspec(2, 1, height_ratios=[0.01, 1.2], hspace=0.0)
  
  # Top row adapts to the number of constituent sub-models
  gs_top = gs[0].subgridspec(1, n_tune_plots, wspace=0.2)
  # Bottom row is strictly 4 columns for the metrics
  gs_bottom = gs[1].subgridspec(1, 4, wspace=0.25)
  
  # --- ROW 1: Best Hyperparameters (1 for each sub-model) ---
  for i, sub_task in enumerate(sub_models):
    ax = fig.add_subplot(gs_top[0, i])
    tuning_path = tuning_dir / f'{sub_task}_tuning.csv'
    
    if tuning_path.exists():
      df = pd.read_csv(tuning_path)
      
      # Find best configuration
      if 'user_attrs_val_f1' in df.columns:
        best_row = df.sort_values('user_attrs_val_f1', ascending=False).iloc[0]
        train_score = best_row['user_attrs_train_f1']
        val_score = best_row['user_attrs_val_f1']
      elif 'val_f1' in df.columns:
        best_row = df.sort_values('val_f1', ascending=False).iloc[0]
        train_score = best_row['train_f1']
        val_score = best_row['val_f1']
      else:
        best_row = df.iloc[0]
        train_score, val_score = 0.0, 0.0

      # Extract params
      params_str = []
      if 'kernel_type' in best_row and pd.notna(best_row['kernel_type']):
        params_str.append(f"Kernel: {best_row['kernel_type']}")
      elif 'kernel' in best_row and pd.notna(best_row['kernel']):
        params_str.append(f"Kernel: {best_row['kernel']}")
        
      for col in df.columns:
        if col.startswith('params_') and pd.notna(best_row[col]):
          p_name = col.replace('params_', '')
          val = best_row[col]
          if isinstance(val, float) and val > 0 and val < 0.01:
            params_str.append(f"{p_name}: {val:.1e}")
          elif isinstance(val, float):
            params_str.append(f"{p_name}: {val:.4f}")
          else:
            params_str.append(f"{p_name}: {val}")
            
      # Support for alternative CSV format without 'params_' prefix
      if not any(col.startswith('params_') for col in df.columns):
        for p_name in ['C', 'gamma', 'degree', 'coef0', 'a']:
          if p_name in df.columns and pd.notna(best_row[p_name]):
            val = best_row[p_name]
            if isinstance(val, float) and val > 0 and val < 0.01:
              params_str.append(f"{p_name}: {val:.1e}")
            elif isinstance(val, float):
              params_str.append(f"{p_name}: {val:.4f}")
            else:
              params_str.append(f"{p_name}: {val}")
            
      params_text = "\n".join(params_str)
      
      # Load Sub-Task Test Metrics
      sub_summary_path = metrics_dir / f'{sub_task}_test_summary.pkl'
      test_bal_acc, test_switch_f1 = 0.0, 0.0
      if sub_summary_path.exists():
        with open(sub_summary_path, 'rb') as f:
          sub_summary = pickle.load(f)
        sub_metrics = sub_summary.get('best_f1_model_test_metrics', {})
        test_bal_acc = sub_metrics.get('balanced_accuracy', 0.0)
        switch_p = sub_metrics.get('switch_precision', 0)
        switch_r = sub_metrics.get('switch_recall', 0)
        if (switch_p + switch_r) > 0:
          test_switch_f1 = 2 * (switch_p * switch_r) / (switch_p + switch_r)
          
      score_text = (f"Train Switch-F1: {train_score:.4f}\n"
                    f"Val Switch-F1:   {val_score:.4f}\n"
                    f"Test Switch-F1:  {test_switch_f1:.4f}\n"
                    f"Test Bal-Acc:    {test_bal_acc:.4f}")
      
      # ax.text(0.3, 0.5, f"Best Hyperparameters:\n{params_text}", ha='center', va='center', fontsize=12, family='monospace', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
      # ax.text(0.75, 0.5, score_text, ha='center', va='center', fontsize=12, fontweight='bold', color='darkblue', family='monospace')
      
      # ax.set_title(f'Selected Model: {sub_task}', fontweight='bold', fontsize=14)
      ax.axis('off')
    else:
      # ax.text(0.5, 0.5, f"Tuning log missing for\n{sub_task}", ha='center', va='center', fontsize=12)
      # ax.set_title(f'Selected Model: {sub_task}', fontweight='bold', fontsize=14)
      ax.axis('off')
      
  # Extract granular data for lower plots
  sm = metrics.get('switch_metrics', {})
  switch_support = sum(sm[k]['support'] for k in sm if k[0] != k[2])
  no_switch_support = sum(sm[k]['support'] for k in sm if k[0] == k[2])
  
  # --- ROW 2, COL 1: Reconstructed 2x2 Confusion Matrix ---
  ax_cm = fig.add_subplot(gs_bottom[0, 0])
  if switch_support > 0 or no_switch_support > 0:
    s_rec = metrics.get('switch_recall', 0)
    ns_rec = metrics.get('no_switch_recall', 0)
    
    tp = int(np.round(s_rec * switch_support))
    fn = int(switch_support - tp)
    tn = int(np.round(ns_rec * no_switch_support))
    fp = int(no_switch_support - tn)
    
    cm_2x2 = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_2x2, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False, annot_kws={"size": 16})
    ax_cm.set_xticklabels(['No Switch (0)', 'Switch (1)'], fontsize=12)
    ax_cm.set_yticklabels(['No Switch (0)', 'Switch (1)'], fontsize=12, rotation=0)
    ax_cm.set_xlabel('Predicted Label', fontsize=12)
    ax_cm.set_ylabel('True Label', fontsize=12)
  ax_cm.set_title('Switch vs No-Switch\nConfusion Matrix', fontweight='bold', fontsize=14)
  
  # --- ROW 2, COL 2: Global Binary P/R Metrics ---
  ax_bin = fig.add_subplot(gs_bottom[0, 1])
  bar_labels = ['Switch\nPrecision', 'Switch\nRecall', 'No-Switch\nPrecision', 'No-Switch\nRecall']
  bar_values = [
    metrics.get('switch_precision', 0), metrics.get('switch_recall', 0),
    metrics.get('no_switch_precision', 0), metrics.get('no_switch_recall', 0)
  ]
  bar_colors = ['#ff9999', '#ff9999', '#99ccff', '#99ccff']
  
  bars = ax_bin.bar(bar_labels, bar_values, color=bar_colors, edgecolor='black')
  ax_bin.set_ylim(0, 1.1)
  ax_bin.set_title('Global Binary Performance', fontweight='bold', fontsize=14)
  ax_bin.set_ylabel('Score', fontsize=12)
  ax_bin.grid(alpha=0.3, axis='y')
  for bar in bars:
    yval = bar.get_height()
    ax_bin.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
    
  # --- ROW 2, COL 3: No-Switch Granular Types (Grouped Bar) ---
  ax_ns_types = fig.add_subplot(gs_bottom[0, 2])
  ns_keys = [k for k in sm.keys() if k[0] == k[2] and sm[k]['support'] > 0]
  if ns_keys:
    ns_p = [sm[k]['precision'] for k in ns_keys]
    ns_r = [sm[k]['recall'] for k in ns_keys]
    fmt = lambda s: s.replace('a', r'$\alpha$').replace('b', r'$\beta$').replace('c', r'$c$').replace('->', r'$\rightarrow$')
    ns_labels = [f"{fmt(k)}\n(n={sm[k]['support']})" for k in ns_keys]
    
    x = np.arange(len(ns_keys))
    width = 0.35
    bars_p = ax_ns_types.bar(x - width/2, ns_p, width, label='Precision', color='#99ccff', edgecolor='black')
    bars_r = ax_ns_types.bar(x + width/2, ns_r, width, label='Recall', color='#66b3ff', edgecolor='black')
    
    ax_ns_types.set_xticks(x)
    ax_ns_types.set_xticklabels(ns_labels, fontsize=11)
    ax_ns_types.set_ylim(0, 1.15)
    ax_ns_types.set_title('No-Switch Sub-Types', fontweight='bold', fontsize=14)
    ax_ns_types.legend(loc='upper right', fontsize=10)
    ax_ns_types.grid(alpha=0.3, axis='y')
    
    for bar in bars_p + bars_r:
      yval = bar.get_height()
      if yval > 0:
        ax_ns_types.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
  else:
    ax_ns_types.text(0.5, 0.5, "No No-Switch types\npresent in test set", ha='center', va='center', fontsize=12)
    ax_ns_types.axis('off')

  # --- ROW 2, COL 4: Switch Granular Types (Grouped Bar) ---
  ax_s_types = fig.add_subplot(gs_bottom[0, 3])
  s_keys = [k for k in sm.keys() if k[0] != k[2] and sm[k]['support'] > 0]
  if s_keys:
    s_p = [sm[k]['precision'] for k in s_keys]
    s_r = [sm[k]['recall'] for k in s_keys]
    s_labels = [f"{fmt(k)}\n(n={sm[k]['support']})" for k in s_keys]
    
    x = np.arange(len(s_keys))
    width = 0.35
    bars_p = ax_s_types.bar(x - width/2, s_p, width, label='Precision', color='#ff9999', edgecolor='black')
    bars_r = ax_s_types.bar(x + width/2, s_r, width, label='Recall', color='#ff6666', edgecolor='black')
    
    ax_s_types.set_xticks(x)
    # Rotate labels if there are a lot of switch types to prevent crowding
    ax_s_types.set_xticklabels(s_labels, rotation=45 if len(s_keys) > 3 else 0, fontsize=11)
    ax_s_types.set_ylim(0, 1.15)
    ax_s_types.set_title('Switch Sub-Types', fontweight='bold', fontsize=14)
    ax_s_types.legend(loc='upper right', fontsize=10)
    ax_s_types.grid(alpha=0.3, axis='y')
    
    for bar in bars_p + bars_r:
      yval = bar.get_height()
      if yval > 0:
        ax_s_types.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
  else:
    ax_s_types.text(0.5, 0.5, "No Switch types\npresent in test set", ha='center', va='center', fontsize=12)
    ax_s_types.axis('off')
    
  # Calculate Switch F1-Score
  switch_p = metrics.get('switch_precision', 0)
  switch_r = metrics.get('switch_recall', 0)
  switch_f1 = 2 * (switch_p * switch_r) / (switch_p + switch_r) if (switch_p + switch_r) > 0 else 0.0
  
  # Main Title
  # fig.suptitle(
  #   f"Comprehensive Evaluation: {task_name}\n"
  #   f"Features: {feature_type.upper()}\n"
  #   f"Switch F1-Score: {switch_f1:.4f}",
  #   fontsize=18, fontweight='bold', y=1.03
  # )
  
  # Set tight layout specifically constrained to not overlap the big title
  plt.tight_layout(rect=[0, 0, 1, 0.98])
  plt.show()

def rank_tasks_by_switch_f1(feature_types=['fusionprot', 'saprot', 'protssn', 'prosst', 'kndrf', 'kncsic'], 
                            k_values=[1, 3, 5, 7],
                            tasks = [
                            'Global_Binary', 
                            'Global_3Class_CommonC_OvR', 
                            'Global_3Class_CommonC_OvO', 
                            'Combined_InputDep_Binary', 
                            'Combined_InputDep_3Class_CommonC_OvR',
                            'Combined_InputDep_3Class_CommonC_OvO'
                            ]):
  """
  Ranks a predefined list of tasks based on their switch-prediction F1-scores
  across multiple feature types.
  """
  
  results = []
  
  for feature_type in feature_types:
    if feature_type == 'fusionprot':
      base_dirs = [(Path('saved_data/svm_experiments/fusionprot'), 'fusionprot')]
    elif feature_type == 'saprot':
      base_dirs = [(Path('saved_data/svm_experiments/saprot'), 'saprot')]
    elif feature_type == 'protssn':
      base_dirs = [(Path('saved_data/svm_experiments/protssn'), 'protssn')]
    elif feature_type == 'prosst':
      base_dirs = [(Path('saved_data/svm_experiments/prosst'), 'prosst')]
    elif feature_type == 'kncsic':
      base_dirs = [(Path(f'saved_data/svm_experiments/kncsic/k_{k}'), f'kncsic (k={k})') for k in k_values]
    elif feature_type == 'kndrf':
      base_dirs = [(Path(f'saved_data/svm_experiments/kndrf/k_{k}'), f'kndrf (k={k})') for k in k_values]
    else:
      raise ValueError(f"Unknown feature type: {feature_type}")
      
    for base_dir, feature_label in base_dirs:
      metrics_dir = base_dir / 'metrics'
      
      for task_name in tasks:
        summary_path = metrics_dir / f'{task_name}_test_summary.pkl'
        
        # Skip configurations that haven't been trained yet
        if not summary_path.exists():
          continue
          
        with open(summary_path, 'rb') as f:
          summary = pickle.load(f)
          
        metrics = summary.get('best_f1_model_test_metrics', {})
        
        # Calculate Switch F1-Score
        switch_p = metrics.get('switch_precision', 0)
        switch_r = metrics.get('switch_recall', 0)
        switch_f1 = 2 * (switch_p * switch_r) / (switch_p + switch_r) if (switch_p + switch_r) > 0 else 0.0
        
        results.append({
          'Task': task_name, 
          'Feature_Type': feature_label,
          'Switch_F1': switch_f1
        })
          
  if not results:
    print("No evaluation results found for the specified configurations.")
    return pd.DataFrame()
    
  # Create a DataFrame and sort descending by Switch F1
  df_rank = pd.DataFrame(results)
  df_rank = df_rank.sort_values(by='Switch_F1', ascending=False).reset_index(drop=True)
  
  print(f"\n=========================================================================================================")
  print(f" TASK RANKING (Across all specified Feature Types)")
  print(f"=========================================================================================================")
  print(f"Rank | {'Task Name':<38} | {'Features':<15} | Switch F1")
  print(f"---------------------------------------------------------------------------------------------------------")
  for idx, row in df_rank.iterrows():
    print(f"{idx + 1:4d} | {row['Task']:<38} | {row['Feature_Type']:<15} | {row['Switch_F1']:.4f}")
  print(f"=========================================================================================================\n")
    
  return df_rank

# ==========================================
# USAGE EXAMPLE:
# ==========================================
if __name__ == "__main__":
  # For FusionProt embeddings:
  # plot_comprehensive_evaluation(task_name="Global_Binary", feature_type="fusionprot")
  
  # Or for a unified 3-class model (This will generate 3 separate tuning plots in the top row!):
  # plot_comprehensive_evaluation(task_name="Combined_InputDep_3Class_CommonC_OvR", feature_type="fusionprot")
  
  # Rank the tasks across everything:
  # rank_tasks_by_switch_f1(feature_types=['fusionprot', 'kndrf'], k_values=[3, 5])
  pass