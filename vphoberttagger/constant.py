from .helper import init_logger
from vphoberttagger.models import PhoBertCrf
from .processor import convert_word_segment_examples_features
from datetime import datetime


LOGGER = init_logger(datetime.now().strftime('%d%b%Y_%H-%M-%S.log'))

# PAP_NER Label scheme (11 BIO labels for Vietnamese e-Government & Legal entities)
LABEL2ID_PAP_NER = [
    'O',
    'B-CQ', 'I-CQ',
    'B-ĐT', 'I-ĐT',
    'B-VBPL', 'I-VBPL',
    'B-NG', 'I-NG',
    'B-SL', 'I-SL'
]

PROCESSOR_MAPPING = {
    'vinai/phobert-base': convert_word_segment_examples_features,
    'vinai/phobert-large': convert_word_segment_examples_features,
    'trungbb8/distilphobert': convert_word_segment_examples_features,
}

MODEL_MAPPING = {
    'vinai/phobert-base': {
        'crf': PhoBertCrf,
    },
    'vinai/phobert-large': {
        'crf': PhoBertCrf,
    },
    'trungbb8/distilphobert': {
        'crf': PhoBertCrf,
    }
}

LABEL_MAPPING = {
    'pap_ner': {
        'label2id': LABEL2ID_PAP_NER,
        'id2label': {idx: label for idx, label in enumerate(LABEL2ID_PAP_NER)},
        'header': ['token', 'pos', 'chunk', 'ner', 'tmp']
    }
}