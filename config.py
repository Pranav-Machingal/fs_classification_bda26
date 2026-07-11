import argparse
from pathlib import Path

# Secondary Structure Mappings
SWITCH_LABELS = [f'{s1}/{s2}' for s1 in ['a', 'b', 'c'] for s2 in ['a', 'b', 'c']]
SS_CLASSES = ['a', 'b', 'c']

def get_args():
    parser = argparse.ArgumentParser(description="Train cuML SVM classifiers (GridSearch Pipeline).")
    parser.add_argument('--feature_type', type=str, default='fusionprot', choices=['kndrf', 'kncsic', 'fusionprot', 'saprot', 'protssn', 'prosst'], help="Feature set to use for training")
    parser.add_argument('--k', type=int, default=3, help="k value (Used only if feature_type is 'kndrf' or 'kncsic')")
    parser.add_argument('--tasks', nargs='+', default=['all'], 
                        choices=['all', 'global_binary', 'global_3class', 'input_dep_binary', 'input_dep_3class'],
                        help="Which classifier suites to train")
    args = parser.parse_args()

    if 'all' in args.tasks:
        args.tasks = ['global_binary', 'global_3class', 'input_dep_binary', 'input_dep_3class']

    return args

def setup_directories(feature_type, k_val):
    if feature_type == 'fusionprot':
        base_dir = Path('saved_data/svm_experiments_fusionprot')
    elif feature_type == 'saprot':
        base_dir = Path('saved_data/svm_experiments_saprot')
    elif feature_type == 'protssn':
        base_dir = Path('saved_data/svm_experiments_protssn')
    elif feature_type == 'prosst':
        base_dir = Path('saved_data/svm_experiments_prosst')
    elif feature_type == 'kncsic':
        base_dir = Path(f'saved_data/svm_experiments_k{k_val}_ncsic')
    elif feature_type == 'kndrf':
        base_dir = Path(f'saved_data/svm_experiments_k{k_val}_ndrf')
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")

    dirs = {
        'base': base_dir,
        'models': base_dir / 'models',
        'metrics': base_dir / 'metrics',
        'tuning': base_dir / 'tuning',
        'curves': base_dir / 'curves'
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs
