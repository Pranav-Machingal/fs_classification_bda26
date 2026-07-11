import numpy as np

def get_weights(y_bin, a):
  """Computes binary sample weights dynamically based on struct parameter 'a'"""
  r_plus = 1.0 / (2 * a)
  r_minus = 1.0 / (2 * (1 - a))
  weights = np.zeros_like(y_bin, dtype=np.float32)
  weights[y_bin == 1] = r_plus
  weights[y_bin == 0] = r_minus
  return weights

def get_log_grid(best_val, step, round_num):
  """Logarithmic grid generator for C and gamma parameters"""
  if round_num == 1:
    return np.array([-4.0, -2.0, 0.0, 2.0, 4.0]), 2.0
  elif round_num == 2:
    new_step = step / 3.0
    return np.array([best_val - 2*new_step, best_val - new_step, best_val, best_val + new_step, best_val + 2*new_step]), new_step
  elif round_num == 3:
    new_step = step / 2.0
    return np.array([best_val - new_step, best_val, best_val + new_step]), new_step

def get_linear_grid(best_val, step, round_num, a_bar):
  """Linear grid generator for structural parameter 'a'"""
  if round_num == 1:
    new_step = a_bar / 4.0
    return np.array([a_bar - 2*new_step, a_bar - new_step, a_bar, a_bar + new_step, a_bar + 2*new_step]), new_step
  elif round_num == 2:
    new_step = step / 3.0
    return np.array([best_val - 2*new_step, best_val - new_step, best_val, best_val + new_step, best_val + 2*new_step]), new_step
  elif round_num == 3:
    new_step = step / 2.0
    return np.array([best_val - new_step, best_val, best_val + new_step]), new_step
