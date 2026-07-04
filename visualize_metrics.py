import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

def plot_detailed_evaluation(task_name, metric_focus='f1', k=3):
  """
  Loads and visualizes the detailed evaluation curves, confusion matrices, 
  and fold-switch specific metrics for a given model.
  """
  # 1. Define Paths
  base_dir = Path(f'saved_data/svm_experiments_k{k}')
  metrics_dir = base_dir / 'metrics'
  curves_dir = base_dir / 'curves'
  
  prefix = f'{task_name}_best_{metric_focus}'
  curves_path = curves_dir / f'{prefix}_curves.pkl'
  cm_path = metrics_dir / f'{prefix}_confusion_matrix.npy'
  summary_path = metrics_dir / f'{task_name}_test_summary.pkl'
  
  # Check if files exist
  if not curves_path.exists() or not cm_path.exists() or not summary_path.exists():
    print(f"Error: Could not find artifacts for {prefix} in {base_dir}")
    return
    
  # 2. Load Artifacts
  with open(curves_path, 'rb') as f:
    curves = pickle.load(f)
  cm = np.load(cm_path)
  with open(summary_path, 'rb') as f:
    summary = pickle.load(f)
    
  metrics = summary[f'best_{metric_focus}_model_test_metrics']
  is_multiclass = 'class_0' in curves
  
  # 3. Initialize Plotting Grid (2 Rows, 3 Columns)
  fig, axes = plt.subplots(2, 3, figsize=(20, 12))
  fig.suptitle(
    f"Task: {task_name} (Optimized for {metric_focus.upper()}) | k={k}\n"
    f"Test Acc: {metrics['accuracy']:.4f} | Bal-Acc: {metrics['balanced_accuracy']:.4f} | F1: {metrics['f1_score']:.4f}",
    fontsize=16, fontweight='bold', y=1.02
  )
  
  # --- ROW 1, COL 1: ROC Curve ---
  ax_roc = axes[0, 0]
  if not is_multiclass:
    ax_roc.plot(curves['fpr'], curves['tpr'], color='darkorange', lw=2, label=f"AUC = {curves['roc_auc']:.3f}")
  else:
    colors = ['blue', 'green', 'red']
    for i in range(3):
      c_data = curves[f'class_{i}']
      ax_roc.plot(c_data['fpr'], c_data['tpr'], color=colors[i], lw=2, label=f"Class {i} AUC = {c_data['roc_auc']:.3f}")
      
  ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
  ax_roc.set_xlim([0.0, 1.0])
  ax_roc.set_ylim([0.0, 1.05])
  ax_roc.set_xlabel('False Positive Rate')
  ax_roc.set_ylabel('True Positive Rate')
  ax_roc.set_title('ROC Curve', fontweight='bold')
  ax_roc.legend(loc="lower right")
  ax_roc.grid(alpha=0.3)

  # --- ROW 1, COL 2: Precision-Recall Curve ---
  ax_pr = axes[0, 1]
  if not is_multiclass:
    ax_pr.plot(curves['recall'], curves['precision'], color='purple', lw=2, label=f"AP = {curves['ap']:.3f}")
  else:
    for i in range(3):
      c_data = curves[f'class_{i}']
      ax_pr.plot(c_data['recall'], c_data['precision'], color=colors[i], lw=2, label=f"Class {i} AP = {c_data['ap']:.3f}")
      
  ax_pr.set_xlim([0.0, 1.0])
  ax_pr.set_ylim([0.0, 1.05])
  ax_pr.set_xlabel('Recall')
  ax_pr.set_ylabel('Precision')
  ax_pr.set_title('Precision-Recall Curve', fontweight='bold')
  ax_pr.legend(loc="lower left")
  ax_pr.grid(alpha=0.3)

  # --- ROW 1, COL 3: Native Confusion Matrix ---
  ax_cm = axes[0, 2]
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False, annot_kws={"size": 14})
  labels = ['No Switch (0)', 'Switch (1)'] if not is_multiclass else ['Helix (0)', 'Sheet (1)', 'Coil (2)']
  ax_cm.set_xticklabels(labels, fontsize=11)
  ax_cm.set_yticklabels(labels, fontsize=11, rotation=0)
  ax_cm.set_xlabel('Predicted Label')
  ax_cm.set_ylabel('True Label')
  ax_cm.set_title('Native Confusion Matrix', fontweight='bold')

  # --- ROW 2, COL 1: Overall Switch vs No-Switch Metrics ---
  ax_bar_main = axes[1, 0]
  if 'switch_precision' in metrics:
    bar_labels = ['Switch\nPrecision', 'Switch\nRecall', 'No-Switch\nPrecision', 'No-Switch\nRecall']
    bar_values = [
      metrics['switch_precision'], metrics['switch_recall'],
      metrics['no_switch_precision'], metrics['no_switch_recall']
    ]
    colors = ['#ff9999', '#ff9999', '#99ccff', '#99ccff']
    
    bars = ax_bar_main.bar(bar_labels, bar_values, color=colors, edgecolor='black')
    ax_bar_main.set_ylim(0, 1.1)
    ax_bar_main.set_title('Binary Switch vs. No-Switch Performance', fontweight='bold')
    ax_bar_main.set_ylabel('Score')
    ax_bar_main.grid(alpha=0.3, axis='y')
    
    # Add text labels on bars
    for bar in bars:
      yval = bar.get_height()
      ax_bar_main.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
  else:
    ax_bar_main.text(0.5, 0.5, "Binary Switch Metrics\nNot Available in this Run", ha='center', va='center')
    ax_bar_main.axis('off')

  # --- ROW 2, COL 2 & 3: Fold-Switch Specific Precision & Recall ---
  ax_prec = axes[1, 1]
  ax_rec = axes[1, 2]
  
  if 'switch_metrics' in metrics:
    sm = metrics['switch_metrics']
    # Filter only the actual switch events (ignoring a/a, b/b, c/c)
    switch_keys = [k for k in sm.keys() if k[0] != k[2]] 
    
    precisions = [sm[k]['precision'] for k in switch_keys]
    recalls = [sm[k]['recall'] for k in switch_keys]
    supports = [sm[k]['support'] for k in switch_keys]
    
    # Custom labels including support count (n)
    x_labels = [f"{k}\n(n={s})" for k, s in zip(switch_keys, supports)]
    
    # Precision Bar Chart
    bars_p = ax_prec.bar(x_labels, precisions, color='mediumpurple', edgecolor='black')
    ax_prec.set_ylim(0, 1.1)
    ax_prec.set_title('Precision by Fold-Switch Type', fontweight='bold')
    ax_prec.set_ylabel('Precision')
    ax_prec.grid(alpha=0.3, axis='y')
    for bar in bars_p:
      yval = bar.get_height()
      ax_prec.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
      
    # Recall Bar Chart
    bars_r = ax_rec.bar(x_labels, recalls, color='lightseagreen', edgecolor='black')
    ax_rec.set_ylim(0, 1.1)
    ax_rec.set_title('Recall by Fold-Switch Type', fontweight='bold')
    ax_rec.set_ylabel('Recall')
    ax_rec.grid(alpha=0.3, axis='y')
    for bar in bars_r:
      yval = bar.get_height()
      ax_rec.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
  else:
    ax_prec.text(0.5, 0.5, "Detailed Switch Metrics Not Available", ha='center', va='center')
    ax_rec.text(0.5, 0.5, "Detailed Switch Metrics Not Available", ha='center', va='center')
    ax_prec.axis('off')
    ax_rec.axis('off')

  plt.tight_layout()
  plt.show()

def plot_tuning_and_evaluation(task_name, metric_focus='f1', k=3):
  """
  Loads the hyperparameter tuning CSV, prints the best config and scores, 
  and visualizes the train/val curves alongside the confusion matrix and switch metrics.
  """
  # 1. Define Paths
  base_dir = Path(f'saved_data/svm_experiments_k{k}')
  metrics_dir = base_dir / 'metrics'
  tuning_dir = base_dir / 'tuning'
  
  prefix = f'{task_name}_best_{metric_focus}'
  cm_path = metrics_dir / f'{prefix}_confusion_matrix.npy'
  summary_path = metrics_dir / f'{task_name}_test_summary.pkl'
  tuning_path = tuning_dir / f'{task_name}_tuning.csv'
  
  if not cm_path.exists() or not summary_path.exists():
    print(f"Error: Could not find model artifacts for {prefix} in {base_dir}")
    return
    
  # 2. Load Evaluation Artifacts
  cm = np.load(cm_path)
  with open(summary_path, 'rb') as f:
    summary = pickle.load(f)
  metrics = summary[f'best_{metric_focus}_model_test_metrics']
  is_multiclass = (cm.shape[0] == 3)
  
  # 3. Load Tuning Data & Print Scores
  col_suffix = 'f1' if metric_focus == 'f1' else 'balanced_accuracy'
  train_col = f'train_{col_suffix}'
  val_col = f'val_{col_suffix}'
  
  print(f"{'='*60}")
  print(f" TASK: {task_name} | OPTIMIZED FOR: {metric_focus.upper()} | k={k}")
  print(f"{'='*60}")
  
  if tuning_path.exists():
    df = pd.read_csv(tuning_path)
    best_idx = df[val_col].idxmax()
    best_row = df.loc[best_idx]
    
    print(f"Best Hyperparameters:")
    print(f"  Kernel: {best_row['kernel']}")
    print(f"  Degree: {best_row['degree']}")
    print(f"  C:      {best_row['C']}\n")
    print(f"Scores ({metric_focus.upper()}):")
    print(f"  Train:  {best_row[train_col]:.4f}")
    print(f"  Val:    {best_row[val_col]:.4f}")
  else:
    df = None
    print(f"Notice: Combined ensemble task detected. No unified tuning CSV available.")
    print(f"Scores ({metric_focus.upper()}):")
  
  print(f"  Test:   {metrics[f'{col_suffix}_score' if metric_focus == 'f1' else col_suffix]:.4f}")
  print(f"{'-'*60}\n")

  # 4. Initialize Plotting Grid (2 Rows, 3 Columns)
  fig = plt.figure(figsize=(20, 12))
  gs = fig.add_gridspec(2, 3)
  
  ax_tune = fig.add_subplot(gs[0, 0:2])  # Spans first two columns
  ax_cm = fig.add_subplot(gs[0, 2])
  ax_bar_main = fig.add_subplot(gs[1, 0])
  ax_prec = fig.add_subplot(gs[1, 1])
  ax_rec = fig.add_subplot(gs[1, 2])
  
  # --- ROW 1, COL 1-2: Tuning Curves ---
  if df is not None:
    # Format config name
    df['config'] = df.apply(
      lambda row: f"{row['kernel']}{'_'+str(row['degree']) if str(row['degree']) not in ['N/A', 'nan'] else ''}", 
      axis=1
    )
    colors = plt.cm.tab10.colors
    
    for i, config in enumerate(df['config'].unique()):
      sub_df = df[df['config'] == config].sort_values('C')
      color = colors[i % len(colors)]
      ax_tune.plot(sub_df['C'], sub_df[train_col], linestyle='--', color=color, marker='o', alpha=0.5, label=f"{config} (Train)")
      ax_tune.plot(sub_df['C'], sub_df[val_col], linestyle='-', color=color, marker='s', linewidth=2, label=f"{config} (Val)")
      
    ax_tune.set_xscale('log')
    ax_tune.set_xlabel('C (Regularization Parameter)', fontsize=12)
    ax_tune.set_ylabel(f'Score ({metric_focus.upper()})', fontsize=12)
    ax_tune.set_title('Hyperparameter Tuning Curves', fontweight='bold', fontsize=14)
    ax_tune.legend(loc='best', fontsize=10, ncol=2)
    ax_tune.grid(True, alpha=0.3)
  else:
    ax_tune.text(0.5, 0.5, "Tuning curves not applicable for\ncombined multi-class ensembles.", 
                 ha='center', va='center', fontsize=14, color='gray')
    ax_tune.axis('off')

  # --- ROW 1, COL 3: Native Confusion Matrix ---
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False, annot_kws={"size": 14})
  labels = ['No Switch (0)', 'Switch (1)'] if not is_multiclass else ['Helix (0)', 'Sheet (1)', 'Coil (2)']
  ax_cm.set_xticklabels(labels, fontsize=11)
  ax_cm.set_yticklabels(labels, fontsize=11, rotation=0)
  ax_cm.set_xlabel('Predicted Label')
  ax_cm.set_ylabel('True Label')
  ax_cm.set_title('Test Set Confusion Matrix', fontweight='bold', fontsize=14)

  # --- ROW 2, COL 1: Overall Switch vs No-Switch Metrics ---
  if 'switch_precision' in metrics:
    bar_labels = ['Switch\nPrecision', 'Switch\nRecall', 'No-Switch\nPrecision', 'No-Switch\nRecall']
    bar_values = [
      metrics['switch_precision'], metrics['switch_recall'],
      metrics['no_switch_precision'], metrics['no_switch_recall']
    ]
    bar_colors = ['#ff9999', '#ff9999', '#99ccff', '#99ccff']
    
    bars = ax_bar_main.bar(bar_labels, bar_values, color=bar_colors, edgecolor='black')
    ax_bar_main.set_ylim(0, 1.1)
    ax_bar_main.set_title('Binary Switch vs. No-Switch Performance', fontweight='bold')
    ax_bar_main.set_ylabel('Score')
    ax_bar_main.grid(alpha=0.3, axis='y')
    
    for bar in bars:
      yval = bar.get_height()
      ax_bar_main.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
  else:
    ax_bar_main.text(0.5, 0.5, "Binary Switch Metrics\nNot Available", ha='center', va='center')
    ax_bar_main.axis('off')

  # --- ROW 2, COL 2 & 3: Fold-Switch Specific Precision & Recall ---
  if 'switch_metrics' in metrics:
    sm = metrics['switch_metrics']
    switch_keys = [k for k in sm.keys() if k[0] != k[2]] 
    
    precisions = [sm[k]['precision'] for k in switch_keys]
    recalls = [sm[k]['recall'] for k in switch_keys]
    supports = [sm[k]['support'] for k in switch_keys]
    
    x_labels = [f"{k}\n(n={s})" for k, s in zip(switch_keys, supports)]
    
    # Precision Bar Chart
    bars_p = ax_prec.bar(x_labels, precisions, color='mediumpurple', edgecolor='black')
    ax_prec.set_ylim(0, 1.1)
    ax_prec.set_title('Precision by Fold-Switch Type', fontweight='bold')
    ax_prec.set_ylabel('Precision')
    ax_prec.grid(alpha=0.3, axis='y')
    for bar in bars_p:
      yval = bar.get_height()
      ax_prec.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
      
    # Recall Bar Chart
    bars_r = ax_rec.bar(x_labels, recalls, color='lightseagreen', edgecolor='black')
    ax_rec.set_ylim(0, 1.1)
    ax_rec.set_title('Recall by Fold-Switch Type', fontweight='bold')
    ax_rec.set_ylabel('Recall')
    ax_rec.grid(alpha=0.3, axis='y')
    for bar in bars_r:
      yval = bar.get_height()
      ax_rec.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
  else:
    ax_prec.text(0.5, 0.5, "Detailed Switch Metrics Not Available", ha='center', va='center')
    ax_rec.text(0.5, 0.5, "Detailed Switch Metrics Not Available", ha='center', va='center')
    ax_prec.axis('off')
    ax_rec.axis('off')

  plt.tight_layout()
  plt.show()

def plot_k_performance(task_name, k_values=[1, 3, 5, 7], metric_focus='f1'):
  """
  Plots Train/Val/Test scores across different k values.
  Automatically identifies the best k and displays its confusion matrix, 
  binary performance, and granular switch metrics.
  """
  train_scores = []
  val_scores = []
  test_scores = []
  valid_ks = []

  best_k = None
  best_test_score = -1
  best_metrics = None
  best_cm = None
  best_is_multiclass = False

  col_suffix = 'f1' if metric_focus == 'f1' else 'balanced_accuracy'
  val_col = f'val_{col_suffix}'
  train_col = f'train_{col_suffix}'

  # 1. Aggregate scores across all k values
  for k in k_values:
    base_dir = Path(f'saved_data/svm_experiments_k{k}')
    metrics_dir = base_dir / 'metrics'
    tuning_dir = base_dir / 'tuning'
    
    prefix = f'{task_name}_best_{metric_focus}'
    cm_path = metrics_dir / f'{prefix}_confusion_matrix.npy'
    summary_path = metrics_dir / f'{task_name}_test_summary.pkl'
    tuning_path = tuning_dir / f'{task_name}_tuning.csv'
    
    # Skip if this k was not evaluated
    if not cm_path.exists() or not summary_path.exists():
      continue
      
    with open(summary_path, 'rb') as f:
      summary = pickle.load(f)
    cm = np.load(cm_path)
    metrics = summary[f'best_{metric_focus}_model_test_metrics']
    
    # Extract Test score
    test_score = metrics[f'{col_suffix}_score' if metric_focus == 'f1' else col_suffix]
    
    # Extract Train/Val scores from tuning log if available
    tr_score, v_score = np.nan, np.nan
    if tuning_path.exists():
      df = pd.read_csv(tuning_path)
      if val_col in df.columns:
        best_idx = df[val_col].idxmax()
        tr_score = df.loc[best_idx, train_col]
        v_score = df.loc[best_idx, val_col]
        
    valid_ks.append(k)
    train_scores.append(tr_score)
    val_scores.append(v_score)
    test_scores.append(test_score)
    
    # Track the ultimate champion k
    if test_score > best_test_score:
      best_test_score = test_score
      best_k = k
      best_metrics = metrics
      best_cm = cm
      best_is_multiclass = (cm.shape[0] == 3)

  if not valid_ks:
    print(f"Error: No valid evaluation data found for task '{task_name}' across k_values={k_values}")
    return

  # 2. Initialize Plotting Grid (2 Rows, 3 Columns)
  fig = plt.figure(figsize=(20, 12))
  gs = fig.add_gridspec(2, 3)
  
  ax_k = fig.add_subplot(gs[0, 0:2])
  ax_cm = fig.add_subplot(gs[0, 2])
  ax_bin = fig.add_subplot(gs[1, 0])
  ax_prec = fig.add_subplot(gs[1, 1])
  ax_rec = fig.add_subplot(gs[1, 2])
  
  fig.suptitle(
    f"Performance vs Context Window (k): {task_name}\n"
    f"Best k = {best_k} (Test {metric_focus.upper()}: {best_test_score:.4f})",
    fontsize=16, fontweight='bold', y=1.02
  )

  # --- ROW 1, COL 1-2: Performance vs K Line Plot ---
  ax_k.plot(valid_ks, train_scores, marker='o', linestyle='--', color='tab:blue', label='Train', alpha=0.6)
  ax_k.plot(valid_ks, val_scores, marker='s', linestyle='-.', color='tab:orange', label='Validation', alpha=0.8)
  ax_k.plot(valid_ks, test_scores, marker='D', linestyle='-', color='tab:green', linewidth=2.5, label='Test')
  
  ax_k.axvline(x=best_k, color='red', linestyle=':', alpha=0.7, label=f'Best k ({best_k})')
  ax_k.set_xlabel('k (Context Window Half-Size)', fontsize=12)
  ax_k.set_ylabel(f'Score ({metric_focus.upper()})', fontsize=12)
  ax_k.set_title('SVM Performance across Context Window Sizes', fontweight='bold', fontsize=14)
  ax_k.set_xticks(valid_ks)
  ax_k.legend(fontsize=11)
  ax_k.grid(True, alpha=0.3)

  # --- ROW 1, COL 3: Confusion Matrix (For Best K) ---
  sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, cbar=False, annot_kws={"size": 14})
  labels = ['No Switch (0)', 'Switch (1)'] if not best_is_multiclass else ['Helix (0)', 'Sheet (1)', 'Coil (2)']
  ax_cm.set_xticklabels(labels, fontsize=11)
  ax_cm.set_yticklabels(labels, fontsize=11, rotation=0)
  ax_cm.set_xlabel('Predicted Label')
  ax_cm.set_ylabel('True Label')
  ax_cm.set_title(f'Confusion Matrix (Optimal k={best_k})', fontweight='bold', fontsize=14)

  # --- ROW 2, COL 1: Binary Switch Metrics (For Best K) ---
  if 'switch_precision' in best_metrics:
    bar_labels = ['Switch\nPrecision', 'Switch\nRecall', 'No-Switch\nPrecision', 'No-Switch\nRecall']
    bar_values = [
      best_metrics['switch_precision'], best_metrics['switch_recall'],
      best_metrics['no_switch_precision'], best_metrics['no_switch_recall']
    ]
    bar_colors = ['#ff9999', '#ff9999', '#99ccff', '#99ccff']
    
    bars = ax_bin.bar(bar_labels, bar_values, color=bar_colors, edgecolor='black')
    ax_bin.set_ylim(0, 1.1)
    ax_bin.set_title(f'Binary Switch Performance (Optimal k={best_k})', fontweight='bold')
    ax_bin.set_ylabel('Score')
    ax_bin.grid(alpha=0.3, axis='y')
    
    for bar in bars:
      yval = bar.get_height()
      ax_bin.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
  else:
    ax_bin.text(0.5, 0.5, "Binary Switch Metrics\nNot Available", ha='center', va='center')
    ax_bin.axis('off')

  # --- ROW 2, COL 2 & 3: Granular Switch Metrics (For Best K) ---
  if 'switch_metrics' in best_metrics:
    sm = best_metrics['switch_metrics']
    switch_keys = [key for key in sm.keys() if key[0] != key[2]]
    
    precisions = [sm[key]['precision'] for key in switch_keys]
    recalls = [sm[key]['recall'] for key in switch_keys]
    supports = [sm[key]['support'] for key in switch_keys]
    
    x_labels = [f"{key}\n(n={s})" for key, s in zip(switch_keys, supports)]
    
    # Precision Bar Chart
    bars_p = ax_prec.bar(x_labels, precisions, color='mediumpurple', edgecolor='black')
    ax_prec.set_ylim(0, 1.1)
    ax_prec.set_title(f'Precision by Fold-Switch Type (Optimal k={best_k})', fontweight='bold')
    ax_prec.set_ylabel('Precision')
    ax_prec.grid(alpha=0.3, axis='y')
    for bar in bars_p:
      yval = bar.get_height()
      ax_prec.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
      
    # Recall Bar Chart
    bars_r = ax_rec.bar(x_labels, recalls, color='lightseagreen', edgecolor='black')
    ax_rec.set_ylim(0, 1.1)
    ax_rec.set_title(f'Recall by Fold-Switch Type (Optimal k={best_k})', fontweight='bold')
    ax_rec.set_ylabel('Recall')
    ax_rec.grid(alpha=0.3, axis='y')
    for bar in bars_r:
      yval = bar.get_height()
      ax_rec.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom', fontsize=9)
  else:
    ax_prec.text(0.5, 0.5, "Detailed Switch Metrics Not Available", ha='center', va='center')
    ax_rec.text(0.5, 0.5, "Detailed Switch Metrics Not Available", ha='center', va='center')
    ax_prec.axis('off')
    ax_rec.axis('off')

  plt.tight_layout()
  plt.show()

# ==========================================
# EXAMPLES OF HOW TO CALL THIS IN YOUR NOTEBOOK:
# ==========================================
if __name__ == "__main__":
  # 1. Visualize the hyperparameter tuning and results for the binary model
  # plot_tuning_and_evaluation(task_name="Global_Binary", metric_focus="f1", k=3)
  
  # 2. You can still use the older detailed ROC/PR function if you prefer:
  # plot_detailed_evaluation(task_name="Global_Binary", metric_focus="f1", k=3)
  
  # 3. New function to evaluate performance across a sweep of k values:
  # plot_k_performance(task_name="Global_Binary", k_values=[1, 3, 5, 7], metric_focus="f1")
  pass