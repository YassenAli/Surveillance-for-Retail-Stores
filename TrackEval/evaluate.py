import numpy as np
np.float = float
np.int = int

import sys
import os
sys.path.append(os.path.abspath('TrackEval'))

from trackeval import Evaluator
from trackeval.datasets import MotChallenge2DBox
from trackeval.metrics import HOTA, CLEAR, Identity

dataset = MotChallenge2DBox({
    'GT_FOLDER': os.path.join('data', 'gt', 'mot_challenge', 'train'),
    'TRACKERS_FOLDER': os.path.join('data', 'trackers', 'mot_challenge', 'train'),
    'TRACKERS_TO_EVAL': ['my_tracker'],
    'SPLIT_TO_EVAL': 'train',
    'SEQMAP_FILE': None,  # Let TrackEval auto-generate seqmap
    'SEQ_INFO': {'02': 2782, '03': 2405, '05': 3315},  # List your sequences here
    'DO_PREPROC': True,
    'BENCHMARK': 'MOT17',
})

metrics = [HOTA(), CLEAR(), Identity()]

# Run evaluation
evaluator = Evaluator({})
eval_results = evaluator.evaluate([dataset], metrics)