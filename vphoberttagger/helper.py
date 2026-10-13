import argparse
import os
import re
import random
import logging

import torch
import numpy as np

if hasattr(torch.serialization, 'add_safe_globals'):
    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
    except Exception:
        pass


def safe_torch_load(f, map_location=None, **kwargs):
    """Safely loads torch files supporting PyTorch 2.6+ where weights_only defaults to True."""
    try:
        return torch.load(f, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(f, map_location=map_location, **kwargs)


def set_ramdom_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        try:
            if not os.path.isdir('./logs'):
                os.makedirs('./logs', exist_ok=True)
            log_path = os.path.join('./logs', log_file)
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(log_file_level)
            file_handler.setFormatter(log_format)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Notice: Could not write log file ({e}), logging to console only.")
    return logger


def get_total_model_parameters(model):
    total_params, trainable_params = 0, 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        if parameter.requires_grad:
            trainable_params += params
        total_params += params
    return total_params,


def normalize_text(txt: str) -> str:
    # Remove special character
    txt = re.sub("\xad|\u200b|\ufeff", "", txt)
    # Normalize vietnamese accents
    txt = re.sub(r"òa", "oà", txt)
    txt = re.sub(r"óa", "oá", txt)
    txt = re.sub(r"ỏa", "oả", txt)
    txt = re.sub(r"õa", "oã", txt)
    txt = re.sub(r"ọa", "oạ", txt)
    txt = re.sub(r"òe", "oè", txt)
    txt = re.sub(r"óe", "oé", txt)
    txt = re.sub(r"ỏe", "oẻ", txt)
    txt = re.sub(r"õe", "oẽ", txt)
    txt = re.sub(r"ọe", "oẹ", txt)
    txt = re.sub(r"ùy", "uỳ", txt)
    txt = re.sub(r"úy", "uý", txt)
    txt = re.sub(r"ủy", "uỷ", txt)
    txt = re.sub(r"ũy", "uỹ", txt)
    txt = re.sub(r"ụy", "uỵ", txt)
    txt = re.sub(r"Ủy", "Uỷ", txt)

    txt = re.sub(r'"', '”', txt)

    # Remove multi-space
    txt = re.sub(" +", " ", txt)
    return txt.strip()


def _get_tags(sents):
    tags = []
    for sent_idx, iob_tags in enumerate(sents):
        curr_tag = {'type': None, 'start_idx': None,
                    'end_idx': None, 'sent_idx': None}
        for i, tag in enumerate(iob_tags):
            if tag == 'O' and curr_tag['type']:
                tags.append(tuple(curr_tag.values()))
                curr_tag = {'type': None, 'start_idx': None,
                            'end_idx': None, 'sent_idx': None}
            elif tag.startswith('B'):
                curr_tag['type'] = tag[2:]
                curr_tag['start_idx'] = i
                curr_tag['end_idx'] = i
                curr_tag['sent_idx'] = sent_idx
            elif tag.startswith('I'):
                curr_tag['end_idx'] = i
        if curr_tag['type']:
            tags.append(tuple(curr_tag.values()))
    tags = set(tags)
    return tags


def plot_confusion_matrix(y_true, y_pred, classes, labels,
                          normalize=False,
                          title=None,
                          output_dir='./',
                          cmap=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    if cmap is None:
        cmap = plt.cm.Blues

    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    # Only use the labels that appear in the data

    if normalize:
        row_sums = cm.sum(axis=1)[:, np.newaxis]
        cm = np.divide(cm.astype('float'), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        print("Normalized confusion matrix")
    else:
         print('Confusion matrix, without normalization')

    plt.rcParams['figure.figsize'] = [10, 10]
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close(fig)
    return ax
