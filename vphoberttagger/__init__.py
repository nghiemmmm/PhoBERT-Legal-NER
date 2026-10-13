__version__ = '1.0.0'
__author__ = ''

from .constant import LOGGER

import vphoberttagger.trainer as Trainer
try:
    import vphoberttagger.predictor as Predictor
except Exception:
    Predictor = None


__all__ = ['Trainer', 'Predictor', 'LOGGER']