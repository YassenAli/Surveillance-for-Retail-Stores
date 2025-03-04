import os
from trackeval import Evaluator
from trackeval.metrics import HOTA, CLEAR, Identity
from trackeval.datasets import MotChallenge2DBox

# Configuration
dataset_config = {
    'GT_FOLDER': os.path.join('data', 'tracking', 'train'),
    'TRACKERS_FOLDER': os.path.join('results', 'trackers'),  # Your tracker results folder
    'OUTPUT_FOLDER': os.path.join('results', 'eval'),
    'TRACKERS_TO_EVAL': ['your_tracker'],  # Name of your tracker folder
    'BENCHMARK': 'MOT15',
    'SPLIT_TO_EVAL': 'train',
    'GT_LOC_FORMAT': os.path.join('{seq}', 'gt', 'gt.txt'),
    'TRACKER_SUB_FOLDER': '',  # Tracker files are directly in tracker folder
    'TRACKER_LOC_FORMAT': os.path.join('{seq}', 'gt', 'gt.txt'),  # Your tracker format
    'SEQ_INFO': {str(i).zfill(2): {} for i in range(2, 4)},  # For sequences 01-02
    'SKIP_SPLIT_FOL': True,
    'DO_PREPROC': True,
    'CLASSES_TO_EVAL': ['pedestrian'],
}

metric_config = [HOTA(), CLEAR(), Identity()]

# Validate file structure
for seq in dataset_config['SEQ_INFO'].keys():
    gt_path = os.path.join(dataset_config['GT_FOLDER'], seq, 'gt', 'gt.txt')
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Missing GT file: {gt_path}")

# Run evaluation
dataset = MotChallenge2DBox(dataset_config)
evaluator = Evaluator(dataset, metric_config)
eval_results = evaluator.evaluate()

# Print summary
print("\nFinal Results:")
print(f"HOTA: {eval_results['MotChallenge2DBox']['your_tracker']['COMBINED_SEQ']['HOTA']['HOTA']:.2f}")
print(f"DetA: {eval_results['MotChallenge2DBox']['your_tracker']['COMBINED_SEQ']['HOTA']['DetA']:.2f}")
print(f"AssA: {eval_results['MotChallenge2DBox']['your_tracker']['COMBINED_SEQ']['HOTA']['AssA']:.2f}")