import numpy as np
from sklearn.metrics import (
  balanced_accuracy_score, f1_score, accuracy_score,
  roc_curve, auc, precision_recall_curve, average_precision_score,
  confusion_matrix, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from config import SS_CLASSES, SWITCH_LABELS

def decode_labels(y_indices):
  """Converts 9-class index into original state, binary, and 3-class targets."""
  labels = [SWITCH_LABELS[i] for i in y_indices]
  
  orig_ss = np.array([lbl.split('/')[0] for lbl in labels])
  target_ss = np.array([lbl.split('/')[1] for lbl in labels])
  
  # Binary: 1 if fold-switch (orig != target), else 0
  y_binary = (orig_ss != target_ss).astype(int)
  
  # 3-Class: predict target_ss
  y_3class = np.array([SS_CLASSES.index(t) for t in target_ss])
  
  return orig_ss, y_binary, y_3class

def compute_switch_f1(y_true, y_pred, orig_ss, is_binary):
  """Calculates the F1 score specifically for the fold-switch event."""
  if is_binary:
    return float(f1_score(y_true, y_pred, zero_division=0))
  else:
    # Convert 3-class predictions back to binary switch events
    y_pred_char = np.array([SS_CLASSES[p] for p in y_pred])
    y_true_char = np.array([SS_CLASSES[t] for t in y_true])
    
    switch_pred = (y_pred_char != orig_ss).astype(int)
    switch_true = (y_true_char != orig_ss).astype(int)
    
    return float(f1_score(switch_true, switch_pred, zero_division=0))

def get_switch_metrics(test_orig, y_test_raw, y_test_pred, is_binary):
  """Calculates precision and recall for each specific type of fold switch."""
  switch_metrics = {}
  for i, label in enumerate(SWITCH_LABELS):
    s1, s2 = label.split('/')
    true_mask = (y_test_raw == i)
    
    if is_binary:
      if s1 == s2:
        pred_mask = (test_orig == s1) & (y_test_pred == 0)
      else:
        pred_mask = (test_orig == s1) & (y_test_pred == 1)
    else:
      pred_mask = (test_orig == s1) & (y_test_pred == SS_CLASSES.index(s2))
      
    tp = np.sum(true_mask & pred_mask)
    p_total = np.sum(pred_mask)
    t_total = np.sum(true_mask)
    
    precision = tp / p_total if p_total > 0 else 0.0
    recall = tp / t_total if t_total > 0 else 0.0
    
    switch_metrics[label] = {
      'precision': float(precision), 
      'recall': float(recall),
      'support': int(t_total)
    }
    
  return switch_metrics

def compute_test_metrics(y_test, y_test_pred, y_score, is_binary, classes=None, test_orig=None, y_test_raw=None):
  """Computes performance metrics, confusion matrices, and ROC/PR curves."""
  acc = accuracy_score(y_test, y_test_pred)
  cm = confusion_matrix(y_test, y_test_pred)
  
  if is_binary:
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)
    
    curves = {
      'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 
      'precision': precision, 'recall': recall, 'ap': ap
    }
    
  else:
    bal_acc = balanced_accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, average='macro')
    
    y_test_bin = label_binarize(y_test, classes=classes)
    curves = {}
    
    for i in range(len(classes)):
      fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
      roc_auc = auc(fpr, tpr)
      precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
      ap = average_precision_score(y_test_bin[:, i], y_score[:, i])
      curves[f'class_{i}'] = {
        'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc, 
        'precision': precision, 'recall': recall, 'ap': ap
      }
      
  metrics_dict = {'accuracy': acc, 'balanced_accuracy': bal_acc, 'f1_score': f1}
  
  # Calculate overall Switch / No-Switch precision and recall
  if is_binary:
    metrics_dict['switch_precision'] = float(precision_score(y_test, y_test_pred, pos_label=1, zero_division=0))
    metrics_dict['switch_recall'] = float(recall_score(y_test, y_test_pred, pos_label=1, zero_division=0))
    metrics_dict['no_switch_precision'] = float(precision_score(y_test, y_test_pred, pos_label=0, zero_division=0))
    metrics_dict['no_switch_recall'] = float(recall_score(y_test, y_test_pred, pos_label=0, zero_division=0))
  elif test_orig is not None:
    y_test_pred_char = np.array([SS_CLASSES[p] for p in y_test_pred])
    y_true_bin = (test_orig != np.array([SS_CLASSES[t] for t in y_test])).astype(int)
    y_pred_bin = (test_orig != y_test_pred_char).astype(int)
    
    metrics_dict['switch_precision'] = float(precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
    metrics_dict['switch_recall'] = float(recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
    metrics_dict['no_switch_precision'] = float(precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0))
    metrics_dict['no_switch_recall'] = float(recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0))
  
  if test_orig is not None and y_test_raw is not None:
    metrics_dict['switch_metrics'] = get_switch_metrics(test_orig, y_test_raw, y_test_pred, is_binary)
    
  return metrics_dict, cm, curves
