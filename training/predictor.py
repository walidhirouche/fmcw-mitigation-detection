"""Load training checkpoints or inference exports with the same preprocessing."""
from pathlib import Path
import math


def checkpoint_settings(state, threshold=None):
    if 'model' not in state or 'model_config' not in state:
        raise ValueError('Expected model weights and model_config in the checkpoint.')
    preprocessing = state.get('preprocessing')
    if preprocessing is not None:
        if (preprocessing.get('FFT_FACTOR') != 2 or preprocessing.get('AMP_SCALE') != 0.001
                or preprocessing.get('window', 'rectangular') != 'rectangular'):
            raise ValueError('Unsupported FFT preprocessing.')
    if threshold is not None:
        value, source = float(threshold), 'user'
    elif state.get('threshold') is not None:
        value, source = float(state['threshold']), 'checkpoint'
    else:
        value, source = 0.5, 'default_0.5'
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Presence threshold must be between 0 and 1.')
    return value, source


class Predictor:
    def __init__(self, weights=None, device='cpu', threshold=None):
        import torch
        from fmcw_training.inference import Predictor as SignalPredictor
        from fmcw_training.model import build_model
        weights = Path(weights) if weights else Path(__file__).parent/'weights/epoch_0115.pt'
        state = torch.load(weights, map_location='cpu', weights_only=True)
        value, self.threshold_source = checkpoint_settings(state, threshold)
        self.epoch = state.get('epoch')
        # Reuse the original waveform validation, preprocessing and inference.
        self.predictor = SignalPredictor.__new__(SignalPredictor)
        self.predictor.device = torch.device(device)
        self.predictor.model = build_model(state['model_config']).to(self.predictor.device)
        self.predictor.model.load_state_dict(state['model'], strict=True)
        self.predictor.model.eval()
        self.predictor.threshold = value

    def __call__(self, observed, **radar):
        result = self.predictor(observed, **radar)
        result['threshold_source'] = self.threshold_source
        return result
