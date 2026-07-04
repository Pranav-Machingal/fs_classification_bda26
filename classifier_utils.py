import numpy as np
from sklearn.metrics import (
  balanced_accuracy_score, f1_score, accuracy_score,
  roc_curve, auc, precision_recall_curve, average_precision_score,
  confusion_matrix, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize

# Secondary Structure Mappings
SWITCH_LABELS = [f'{s1}/{s2}' for s1 in ['a', 'b', 'c'] for s2 in ['a', 'b', 'c']]
SS_CLASSES = ['a', 'b', 'c']

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

def compute_sample_weights(y_raw, strategy='binary'):
  """Calculates sample weights for SVM training based on switch types."""
  n = len(y_raw)
  weights = np.ones(n, dtype=np.float64)
  
  if strategy == 'binary':
    _, y_binary, _ = decode_labels(y_raw)
    n_pos = np.sum(y_binary == 1)
    n_neg = np.sum(y_binary == 0)
    
    w_pos = n / (2 * n_pos) if n_pos > 0 else 0
    w_neg = n / (2 * n_neg) if n_neg > 0 else 0
    
    weights[y_binary == 1] = w_pos
    weights[y_binary == 0] = w_neg
    
  elif strategy == 'type':
    # 9 possible transition classes
    for c in range(9):
      mask = (y_raw == c)
      n_c = np.sum(mask)
      if n_c > 0:
        weights[mask] = n / (9 * n_c)
        
  return weights

def get_switch_metrics(test_orig, y_test_raw, y_test_pred, is_binary):
  """Calculates precision and recall for each specific type of fold switch."""
  switch_metrics = {}
  for i, label in enumerate(SWITCH_LABELS):
    s1, s2 = label.split('/')
    true_mask = (y_test_raw == i)
    
    if is_binary:
      # Assumes y_test_pred: 1 = switch, 0 = no-switch
      if s1 == s2:
        pred_mask = (test_orig == s1) & (y_test_pred == 0)
      else:
        pred_mask = (test_orig == s1) & (y_test_pred == 1)
    else:
      # Assumes y_test_pred is 3-class target: 0='a', 1='b', 2='c'
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
    # Convert 3-Class predictions back to binary Switch/No-Switch outcomes
    y_test_pred_char = np.array([SS_CLASSES[p] for p in y_test_pred])
    y_true_bin = (test_orig != np.array([SS_CLASSES[t] for t in y_test])).astype(int)
    y_pred_bin = (test_orig != y_test_pred_char).astype(int)
    
    metrics_dict['switch_precision'] = float(precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
    metrics_dict['switch_recall'] = float(recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0))
    metrics_dict['no_switch_precision'] = float(precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0))
    metrics_dict['no_switch_recall'] = float(recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0))
  
  # Embed the 9-class specific fold switch tracking
  if test_orig is not None and y_test_raw is not None:
    metrics_dict['switch_metrics'] = get_switch_metrics(test_orig, y_test_raw, y_test_pred, is_binary)
    
  return metrics_dict, cm, curves

class IndividuallyTunedOvR:
  """Combines independently trained binary OvR classifiers."""
  def __init__(self, estimators, classes):
    self.estimators = estimators
    self.classes_ = np.array(classes)
    
  def decision_function(self, X):
    scores = []
    for est in self.estimators:
      if hasattr(est, 'decision_function'):
        scores.append(est.decision_function(X))
      else:
        scores.append(est.predict_proba(X)[:, 1])
    return np.column_stack(scores)
    
  def predict(self, X):
    scores = self.decision_function(X)
    return self.classes_[np.argmax(scores, axis=1)]

  def predict_proba(self, X):
    probs = np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators])
    # Normalize approximations to sum to 1
    return probs / np.sum(probs, axis=1, keepdims=True)

class IndividuallyTunedOvO:
  """Combines independently trained binary OvO pairwise classifiers."""
  def __init__(self, estimators, classes, pairs):
    self.estimators = estimators
    self.classes_ = np.array(classes)
    self.pairs = pairs
    
  def predict(self, X):
    votes = np.zeros((X.shape[0], len(self.classes_)))
    for est, (c1, c2) in zip(self.estimators, self.pairs):
      pred = est.predict(X)
      # Class 1 is mapped to 1, Class 2 is mapped to 0 during training
      votes[np.arange(X.shape[0]), np.where(pred == 1, c1, c2)] += 1
    return self.classes_[np.argmax(votes, axis=1)]
    
  def decision_function(self, X):
    scores = np.zeros((X.shape[0], len(self.classes_)))
    for est, (c1, c2) in zip(self.estimators, self.pairs):
      if hasattr(est, "predict_proba"):
        prob = est.predict_proba(X)[:, 1]
        scores[:, c1] += prob
        scores[:, c2] += (1 - prob)
      else:
        df = est.decision_function(X)
        scores[:, c1] += df
        scores[:, c2] -= df
    return scores

class InputDependentCombinedClassifier:
  """Routes inputs to the appropriate sub-model based on the central residue's SSE."""
  def __init__(self, models_dict, k):
    self.models_dict = models_dict
    self.k = k
    first_model = next(iter(models_dict.values()))
    self.classes_ = getattr(first_model, 'classes_', None)
    
  def _get_states(self, X):
    # Extract the one-hot encoded SSE of the central residue
    # Features per residue = 29. Central residue is at index k. SSE offset = 26.
    sse_start = self.k * 29 + 26
    return np.argmax(X[:, sse_start : sse_start + 3], axis=1)

  def predict(self, X):
    states = self._get_states(X)
    preds = np.zeros(X.shape[0], dtype=int) if self.classes_ is None else np.zeros(X.shape[0], dtype=self.classes_.dtype)
    for state_idx in np.unique(states):
      mask = (states == state_idx)
      if state_idx in self.models_dict:
        preds[mask] = self.models_dict[state_idx].predict(X[mask])
      else:
        preds[mask] = self.classes_[0] if self.classes_ is not None else 0
    return preds

  @property
  def predict_proba(self):
    first_model = next(iter(self.models_dict.values()))
    if not hasattr(first_model, 'predict_proba'):
      raise AttributeError(f"'{type(first_model).__name__}' object has no attribute 'predict_proba'")
    return self._predict_proba

  def _predict_proba(self, X):
    states = self._get_states(X)
    first_model = next(iter(self.models_dict.values()))
    
    if hasattr(first_model, 'classes_'):
      n_classes = len(first_model.classes_)
    else:
      n_classes = first_model.predict_proba(X[0:1]).shape[1]
      
    probs = np.zeros((X.shape[0], n_classes))
    for state_idx in np.unique(states):
      mask = (states == state_idx)
      if state_idx in self.models_dict:
        probs[mask] = self.models_dict[state_idx].predict_proba(X[mask])
      else:
        probs[mask, 0] = 1.0 # fallback for missing states
    return probs

  @property
  def decision_function(self):
    first_model = next(iter(self.models_dict.values()))
    if not hasattr(first_model, 'decision_function'):
      raise AttributeError(f"'{type(first_model).__name__}' object has no attribute 'decision_function'")
    return self._decision_function

  def _decision_function(self, X):
    states = self._get_states(X)
    first_model = next(iter(self.models_dict.values()))
    
    sample_df = first_model.decision_function(X[0:1])
    scores = np.zeros((X.shape[0], *sample_df.shape[1:]))
    
    for state_idx in np.unique(states):
      mask = (states == state_idx)
      if state_idx in self.models_dict:
        scores[mask] = self.models_dict[state_idx].decision_function(X[mask])
    return scores


class FPInputDependentCombinedClassifier:
  """Routes inputs to sub-models using a pre-aligned states array, bypassing k-NDRF structural assumptions."""
  def __init__(self, models_dict, test_states):
    self.models_dict = models_dict
    self.test_states = test_states
    first_model = next(iter(models_dict.values()))
    self.classes_ = getattr(first_model, 'classes_', None)

  def predict(self, X):
    preds = np.zeros(X.shape[0], dtype=int) if self.classes_ is None else np.zeros(X.shape[0], dtype=self.classes_.dtype)
    for state_idx in np.unique(self.test_states):
      mask = (self.test_states == state_idx)
      if state_idx in self.models_dict:
        preds[mask] = self.models_dict[state_idx].predict(X[mask])
      else:
        preds[mask] = self.classes_[0] if self.classes_ is not None else 0
    return preds

  @property
  def predict_proba(self):
    first_model = next(iter(self.models_dict.values()))
    if not hasattr(first_model, 'predict_proba'):
      raise AttributeError(f"'{type(first_model).__name__}' object has no attribute 'predict_proba'")
    return self._predict_proba

  def _predict_proba(self, X):
    first_model = next(iter(self.models_dict.values()))
    n_classes = len(first_model.classes_) if hasattr(first_model, 'classes_') else first_model.predict_proba(X[0:1]).shape[1]
    probs = np.zeros((X.shape[0], n_classes))
    for state_idx in np.unique(self.test_states):
      mask = (self.test_states == state_idx)
      if state_idx in self.models_dict:
        probs[mask] = self.models_dict[state_idx].predict_proba(X[mask])
      else:
        probs[mask, 0] = 1.0 # fallback for missing states
    return probs

  @property
  def decision_function(self):
    first_model = next(iter(self.models_dict.values()))
    if not hasattr(first_model, 'decision_function'):
      raise AttributeError(f"'{type(first_model).__name__}' object has no attribute 'decision_function'")
    return self._decision_function

  def _decision_function(self, X):
    first_model = next(iter(self.models_dict.values()))
    sample_df = first_model.decision_function(X[0:1])
    scores = np.zeros((X.shape[0], *sample_df.shape[1:]))
    for state_idx in np.unique(self.test_states):
      mask = (self.test_states == state_idx)
      if state_idx in self.models_dict:
        scores[mask] = self.models_dict[state_idx].decision_function(X[mask])
    return scores

class StateRoutedCombinedClassifier:
  """Universally routes inputs to sub-models using a pre-aligned states array."""
  def __init__(self, models_dict, test_states):
    self.models_dict = models_dict
    self.test_states = test_states
    first_model = next(iter(models_dict.values()))
    self.classes_ = getattr(first_model, 'classes_', None)

  def predict(self, X):
    preds = np.zeros(X.shape[0], dtype=int) if self.classes_ is None else np.zeros(X.shape[0], dtype=self.classes_.dtype)
    for state_idx in np.unique(self.test_states):
      mask = (self.test_states == state_idx)
      if state_idx in self.models_dict:
        preds[mask] = self.models_dict[state_idx].predict(X[mask])
      else:
        preds[mask] = self.classes_[0] if self.classes_ is not None else 0
    return preds

  @property
  def predict_proba(self):
    first_model = next(iter(self.models_dict.values()))
    if not hasattr(first_model, 'predict_proba'):
      raise AttributeError(f"'{type(first_model).__name__}' object has no attribute 'predict_proba'")
    return self._predict_proba

  def _predict_proba(self, X):
    first_model = next(iter(self.models_dict.values()))
    n_classes = len(first_model.classes_) if hasattr(first_model, 'classes_') else first_model.predict_proba(X[0:1]).shape[1]
    probs = np.zeros((X.shape[0], n_classes))
    for state_idx in np.unique(self.test_states):
      mask = (self.test_states == state_idx)
      if state_idx in self.models_dict:
        probs[mask] = self.models_dict[state_idx].predict_proba(X[mask])
      else:
        probs[mask, 0] = 1.0 # fallback for missing states
    return probs

  @property
  def decision_function(self):
    first_model = next(iter(self.models_dict.values()))
    if not hasattr(first_model, 'decision_function'):
      raise AttributeError(f"'{type(first_model).__name__}' object has no attribute 'decision_function'")
    return self._decision_function

  def _decision_function(self, X):
    first_model = next(iter(self.models_dict.values()))
    sample_df = first_model.decision_function(X[0:1])
    scores = np.zeros((X.shape[0], *sample_df.shape[1:]))
    for state_idx in np.unique(self.test_states):
      mask = (self.test_states == state_idx)
      if state_idx in self.models_dict:
        scores[mask] = self.models_dict[state_idx].decision_function(X[mask])
    return scores