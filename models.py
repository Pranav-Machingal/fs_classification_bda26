import numpy as np

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
