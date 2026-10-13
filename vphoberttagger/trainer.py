import transformers

from vphoberttagger.constant import LOGGER, MODEL_MAPPING, LABEL_MAPPING
from vphoberttagger.models import PhoBertCrf
from vphoberttagger.helper import set_ramdom_seed, plot_confusion_matrix, get_total_model_parameters, safe_torch_load
from vphoberttagger.arguments import get_train_argument, get_test_argument
from vphoberttagger.dataset import build_dataset
from vphoberttagger.conlleval import evaluate

from tqdm import tqdm
from pathlib import Path
from typing import Union
from prettytable import PrettyTable
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from torch.utils.data import RandomSampler, DataLoader
from transformers import AutoTokenizer, AutoConfig, get_linear_schedule_with_warmup, RobertaForSequenceClassification

import os
import sys
import torch
import time
import datetime
import itertools
import math


def save_model(args, saved_file, model):
    saved_data = {
        'model': model.state_dict(),
        'classes': args.label2id,
        'args': args
    }
    torch.save(saved_data, saved_file)


def validate(model, task, iterator, cur_epoch: int, output_dir: Union[str, os.PathLike] = './', is_test=False, device='cuda', use_amp=False):
    start_time = time.time()
    model.eval()
    eval_loss = 0.0
    eval_golds, eval_preds = [], []
    # Run one step on sub-dataset
    with torch.no_grad():
        tqdm_desc = f'[EVAL- Epoch {cur_epoch}]'
        eval_bar = tqdm(enumerate(iterator), total=len(iterator), desc=tqdm_desc)
        for idx, batch in eval_bar:
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            if use_amp and str(device).startswith('cuda'):
                amp_ctx = torch.amp.autocast('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()
                with amp_ctx:
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            eval_loss_val = outputs.loss.detach().item()
            if not math.isnan(eval_loss_val) and not math.isinf(eval_loss_val):
                eval_loss += eval_loss_val
            active_accuracy = batch['label_masks'].view(-1) != 0
            labels = torch.masked_select(batch['labels'].view(-1), active_accuracy)
            eval_golds.extend(labels.detach().cpu().tolist())
            if isinstance(outputs.tags[-1], list):
                eval_preds.extend(list(itertools.chain(*outputs.tags)))
            else:
                eval_preds.extend(outputs.tags)
    epoch_loss = eval_loss / len(iterator)
    if is_test:
        evaluate([LABEL_MAPPING[task]["id2label"][tag] for tag in eval_golds],
                 [LABEL_MAPPING[task]["id2label"][tag] for tag in eval_preds])
        reports: dict = classification_report(eval_golds, eval_preds,
                                              output_dict=False,
                                              zero_division=0,
                                              digits=4,
                                              target_names=LABEL_MAPPING[task]["label2id"])
        LOGGER.info(reports)
        label_index_to_print = list(range(len(LABEL_MAPPING[task]["label2id"])))
        plot_confusion_matrix(eval_golds, eval_preds,
                              classes=LABEL_MAPPING[task]["label2id"],
                              labels=label_index_to_print,
                              output_dir=output_dir,
                              title=f'Normalized confusion matrix of {task.upper()}',
                              normalize=True)
    else:
        reports: dict = classification_report(eval_golds, eval_preds,
                                              output_dict=True,
                                              zero_division=0)
        epoch_avg_f1 = reports['macro avg']['f1-score']
        epoch_avg_acc = reports['accuracy']
        LOGGER.info(f"\t{'*' * 20}Validate Summary{'*' * 20}")
        LOGGER.info(f"\tValidation Loss: {epoch_loss:.4f};\n"
                    f"\tBIO-Accuracy: {epoch_avg_acc:.4f};\n"
                    f"\tBIO-Macro-F1 score: {epoch_avg_f1:.4f}; "
                    f"\tSpend time: {datetime.timedelta(seconds=(time.time() - start_time))}")
        return epoch_loss, epoch_avg_acc, epoch_avg_f1


def train_one_epoch(model, iterator, optim, cur_epoch: int, max_grad_norm: float = 1.0, scheduler=None, scaler=None, device='cuda'):
    start_time = time.time()
    tr_loss = 0.0
    tr_crf = 0.0
    tr_supcon = 0.0
    has_supcon = False
    model.train()
    use_amp = (scaler is not None and str(device).startswith('cuda'))
    tqdm_bar = tqdm(enumerate(iterator), total=len(iterator), desc=f'[TRAIN-EPOCH {cur_epoch}]')
    for idx, batch in tqdm_bar:
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optim.zero_grad()
        if use_amp:
            amp_ctx = torch.amp.autocast('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast()
            with amp_ctx:
                outputs = model(**batch)
            scaler.scale(outputs.loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            scale_before = scaler.get_scale()
            scaler.step(optim)
            scaler.update()
            scale_after = scaler.get_scale()
            if scheduler and (scale_after >= scale_before):
                scheduler.step()
        else:
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optim.step()
            if scheduler:
                scheduler.step()
        loss_val = outputs.loss.detach().item()
        if not math.isnan(loss_val) and not math.isinf(loss_val):
            tr_loss += loss_val
        if hasattr(outputs, 'loss_crf') and outputs.loss_crf is not None:
            has_supcon = True
            crf_val = outputs.loss_crf.detach().item()
            if not math.isnan(crf_val) and not math.isinf(crf_val):
                tr_crf += crf_val
            if hasattr(outputs, 'loss_supcon') and outputs.loss_supcon is not None:
                supcon_val = outputs.loss_supcon.detach().item()
                if not math.isnan(supcon_val) and not math.isinf(supcon_val):
                    tr_supcon += supcon_val

    epoch_loss = tr_loss / len(iterator)
    epoch_crf = tr_crf / len(iterator) if has_supcon else epoch_loss
    epoch_supcon = tr_supcon / len(iterator) if has_supcon else 0.0

    LOGGER.info(f"\t{'*' * 20}Train Summary{'*' * 20}")
    log_msg = f"\tTraining Lr: {optim.param_groups[0]['lr']}; Loss: {epoch_loss:.4f}; "
    if has_supcon:
        log_msg += f"CRF Loss: {epoch_crf:.4f}; SupCon Loss: {epoch_supcon:.4f}; "
    log_msg += f"Spend time: {datetime.timedelta(seconds=(time.time() - start_time))}"
    LOGGER.info(log_msg)
    return epoch_loss, epoch_crf, epoch_supcon


def train():
    args = get_train_argument()
    LOGGER.info(f"Arguments: {args}")
    set_ramdom_seed(args.seed)
    device = 'cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu'
    use_crf = True if 'crf' in args.model_arch else False
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    tensorboard_writer = SummaryWriter()
    assert os.path.isdir(args.data_dir), f'{args.data_dir} not found!'
    setattr(args, 'label2id', LABEL_MAPPING[args.task]['label2id'])
    setattr(args, 'id2label', LABEL_MAPPING[args.task]['id2label'])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = build_dataset(args.data_dir,
                                  tokenizer,
                                  label2id=args.label2id,
                                  header=LABEL_MAPPING[args.task]['header'],
                                  dtype='train',
                                  max_seq_len=args.max_seq_length,
                                  device='cpu',
                                  use_crf=use_crf,
                                  overwrite_data=args.overwrite_data)
    eval_dtype = 'dev' if (os.path.exists(os.path.join(args.data_dir, 'dev_data.txt')) or os.path.exists(os.path.join(args.data_dir, 'dev.txt'))) else 'test'
    eval_dataset = build_dataset(args.data_dir,
                                 tokenizer,
                                 label2id=args.label2id,
                                 header=LABEL_MAPPING[args.task]['header'],
                                 dtype=eval_dtype,
                                 max_seq_len=args.max_seq_length,
                                 device='cpu',
                                 use_crf=use_crf,
                                 overwrite_data=args.overwrite_data)

    config = AutoConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=len(args.label2id),
                                        finetuning_task=args.task)
    setattr(config, 'use_supcon', getattr(args, 'use_supcon', False))
    setattr(config, 'lambda_supcon', getattr(args, 'lambda_supcon', 0.3))
    setattr(config, 'supcon_temp', getattr(args, 'supcon_temp', 0.07))
    setattr(config, 'supcon_mode', getattr(args, 'supcon_mode', 'exact'))
    setattr(config, 'supcon_alpha', getattr(args, 'supcon_alpha', 0.3))
    o_id = args.label2id.index('O') if isinstance(args.label2id, list) else args.label2id.get('O', 0)
    setattr(config, 'o_label_id', o_id)

    model_clss = PhoBertCrf
    model = model_clss.from_pretrained(pretrained_model_name_or_path=args.model_name_or_path,
                                       config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    if args.load_weights is not None:
        LOGGER.info(f'Load pretrained model weights from "{args.load_weights}"')
        if device == 'cpu':
            checkpoint_data = safe_torch_load(args.load_weights, map_location='cpu')
        else:
            checkpoint_data = safe_torch_load(args.load_weights)
        model.load_state_dict(checkpoint_data['model'], strict=False)
        checkpoint_data = None

    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
    if hasattr(model, 'roberta'):
        bert_param_optimizer = list(model.roberta.named_parameters())
    elif hasattr(model, 'bert'):
        bert_param_optimizer = list(model.bert.named_parameters())
    else:
        raise AttributeError(f"Unsupported model architecture: cannot find encoder attribute ('roberta' or 'bert') on {type(model)}.")
    ner_param_optimizer = list(model.classifier.named_parameters())
    if 'lstm' in args.model_arch:
        ner_param_optimizer.extend(list(model.lstm.named_parameters()))
    if 'crf' in args.model_arch:
        ner_param_optimizer.extend(list(model.crf.named_parameters()))
    if getattr(model, 'projection_head', None) is not None:
        ner_param_optimizer.extend(list(model.projection_head.named_parameters()))

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in ner_param_optimizer],
         'lr': args.classifier_learning_rate, 'weight_decay': args.weight_decay}
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    train_steps_per_epoch = len(train_dataset) // args.train_batch_size
    total_train_steps = (train_steps_per_epoch * int(args.epochs)) // args.gradient_accumulation_steps
    warmup_steps = int(args.warmup_proportion * total_train_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_train_steps)
    use_amp = (not args.no_cuda and torch.cuda.is_available() and getattr(args, 'fp16', True))
    scaler = (torch.amp.GradScaler('cuda') if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler') else torch.cuda.amp.GradScaler()) if use_amp else None
    pin_memory = (str(device).startswith('cuda'))
    LOGGER.info(f"Using Mixed Precision (AMP FP16): {use_amp}, Pin Memory: {pin_memory}, Num Workers: {args.num_workers}")
    train_sampler = RandomSampler(train_dataset)
    train_iterator = DataLoader(train_dataset,
                                sampler=train_sampler,
                                batch_size=args.train_batch_size,
                                num_workers=args.num_workers,
                                pin_memory=pin_memory)
    eval_iterator = DataLoader(eval_dataset,
                               batch_size=args.eval_batch_size,
                               num_workers=args.num_workers,
                               pin_memory=pin_memory)
    best_score = 0.0
    best_loss = float('inf')
    cumulative_early_steps = 0
    for epoch in range(int(args.epochs)):
        if cumulative_early_steps > args.early_stop:
            LOGGER.info(f"Early stopping. Check your saved model.")
            break
        LOGGER.info(f"\n{'=' * 30}Training epoch {epoch}{'=' * 30}")
        # Fit model with dataset
        tr_loss, tr_crf, tr_supcon = train_one_epoch(model=model,
                                                     optim=optimizer,
                                                     iterator=train_iterator,
                                                     cur_epoch=epoch,
                                                     max_grad_norm=args.max_grad_norm,
                                                     scheduler=scheduler,
                                                     scaler=scaler,
                                                     device=device)

        tensorboard_writer.add_scalar('TRAIN/Loss', tr_loss, epoch)
        if getattr(args, 'use_supcon', False):
            tensorboard_writer.add_scalar('TRAIN/Loss_CRF', tr_crf, epoch)
            tensorboard_writer.add_scalar('TRAIN/Loss_SupCon', tr_supcon, epoch)

        # Validate trained model on dataset
        eval_loss, eval_acc, eval_f1 = validate(model=model,
                                 task=args.task,
                                 iterator=eval_iterator,
                                 cur_epoch=epoch,
                                 is_test=False,
                                 device=device,
                                 use_amp=use_amp)

        tensorboard_writer.add_scalar('EVAL_RESULT/Loss', eval_loss, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/BIO-Accuracy', eval_acc, epoch)
        tensorboard_writer.add_scalar('EVAL_RESULT/BIO-F1-score', eval_f1, epoch)

        LOGGER.info(f"\t{'*' * 20}Epoch Summary{'*' * 20}")
        LOGGER.info(f"\tEpoch Loss = {eval_loss:.6f} ; Best loss = {best_loss:.6f}")
        LOGGER.info(f"\tEpoch BIO-F1 score = {eval_f1:.6f} ; Best score = {best_score:.6f}")

        if eval_loss < best_loss:
            best_loss = eval_loss
        if eval_f1 > best_score:
            cumulative_early_steps = 0
            best_score = eval_f1
            saved_file = Path(args.output_dir + f"/best_model.pt")
            LOGGER.info(f"\t***New best model, saving to {saved_file}...***")
            save_model(args, saved_file, model)
        else:
            cumulative_early_steps += 1
    if args.run_test:
        test_dataset = build_dataset(args.data_dir,
                                     tokenizer,
                                     label2id=args.label2id,
                                     header=LABEL_MAPPING[args.task]['header'],
                                     dtype='test',
                                     max_seq_len=args.max_seq_length,
                                     device='cpu',
                                     use_crf=use_crf,
                                     overwrite_data=args.overwrite_data)
        test_iterator = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, pin_memory=pin_memory)
        if device == 'cpu':
            checkpoint_data = safe_torch_load(args.output_dir + f"/best_model.pt", map_location='cpu')
        else:
            checkpoint_data = safe_torch_load(args.output_dir + f"/best_model.pt")
        model.load_state_dict(checkpoint_data['model'], strict=False)
        validate(model=model,
                 task=args.task,
                 iterator=test_iterator,
                 cur_epoch=0,
                 is_test=True,
                 output_dir=args.output_dir,
                 device=device,
                 use_amp=use_amp)

def test():
    args = get_test_argument()
    if not args.no_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {args.model_path}")

    LOGGER.info(f"Loading checkpoint from '{args.model_path}' on device '{device}'...")
    checkpoint = safe_torch_load(args.model_path, map_location=device)
    train_args = checkpoint.get('args', None)

    task = getattr(train_args, 'task', 'pap_ner') if train_args else 'pap_ner'
    model_name_or_path = getattr(train_args, 'model_name_or_path', 'vinai/phobert-base') if train_args else 'vinai/phobert-base'
    max_seq_length = getattr(train_args, 'max_seq_length', 256) if train_args else 256
    classes = checkpoint.get('classes', LABEL_MAPPING[task]['label2id'])

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path,
                                        num_labels=len(classes),
                                        finetuning_task=task)
    use_supcon = getattr(train_args, 'use_supcon', False) if train_args else False
    setattr(config, 'use_supcon', use_supcon)
    model = PhoBertCrf.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                       config=config)
    model.resize_token_embeddings(len(tokenizer))
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model'], strict=False)
    if missing_keys:
        LOGGER.warning(f"Missing keys when loading checkpoint: {missing_keys}")
    if unexpected_keys:
        LOGGER.warning(f"Unexpected keys when loading checkpoint: {unexpected_keys}")
    model.to(device)

    num_workers = getattr(args, 'num_workers', getattr(args, 'num_worker', 0))
    pin_memory = (str(device).startswith('cuda'))
    use_amp = (str(device).startswith('cuda'))
    test_dataset = build_dataset(args.data_dir,
                                 tokenizer,
                                 label2id=classes,
                                 header=LABEL_MAPPING[task]['header'],
                                 dtype='test',
                                 max_seq_len=max_seq_length,
                                 device='cpu',
                                 use_crf=True,
                                 overwrite_data=args.overwrite_data)
    test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers, pin_memory=pin_memory)

    output_dir = os.path.dirname(args.model_path) or './'
    LOGGER.info(f"Running evaluation on test set ({len(test_dataset)} samples)...")
    validate(model=model,
             task=task,
             iterator=test_iterator,
             cur_epoch=0,
             is_test=True,
             output_dir=output_dir,
             device=device,
             use_amp=use_amp)


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        LOGGER.info("Start TRAIN process...")
        train()
    elif sys.argv[1] == 'test':
        LOGGER.info("Start TEST process...")
        test()
    else:
        LOGGER.error(f'[ERROR] - `{sys.argv[1]}` not found!!!')
