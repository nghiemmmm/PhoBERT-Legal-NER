from argparse import ArgumentParser


def get_predict_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What processs to be run')
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def get_test_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What process to be run')
    parser.add_argument("--data_dir", default='Dataset', type=str,
                        help="The input data dir containing dataset files.")
    parser.add_argument("--model_path", default='outputs/best_model.pt', type=str,
                        help="")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwrite splitted dataset")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Total batch size for eval.")
    parser.add_argument('--num_worker', type=int, default=2,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded in "
                             "the main process.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    return parser.parse_args()


def get_train_argument():
    parser = ArgumentParser()
    parser.add_argument('type', choices=['train', 'test', 'predict', 'demo'],
                        help='What process to be run')
    parser.add_argument("--task", default='pap_ner', type=str, choices=['pap_ner'],
                        help="Training task: pap_ner (Vietnamese e-Government / Legal NER)")
    parser.add_argument("--data_dir", default='Dataset', type=str,
                        help="The input data dir. Should contain train_data.txt, dev_data.txt, test_data.txt")
    parser.add_argument("--overwrite_data", action='store_true', default=False,
                        help="Whether not to overwrite splitted dataset")
    parser.add_argument("--load_weights", default=None, type=str,
                        help='Path of pretrained file.')
    parser.add_argument("--model_name_or_path", default='vinai/phobert-base', type=str,
                        help="Pre-trained model selected in the list: vinai/phobert-base, trungbb8/distilphobert, vinai/phobert-large...")
    parser.add_argument("--model_arch", default='crf', type=str, choices=['crf'],
                        help="Model architecture: PhoBERT + CRF for sequence tagging")
    parser.add_argument("--use_supcon", action='store_true', default=False,
                        help="Whether to enable Supervised Contrastive Loss (SupCon) for entity representations")
    parser.add_argument("--lambda_supcon", default=0.3, type=float,
                        help="Weight lambda for SupCon loss (default: 0.3)")
    parser.add_argument("--supcon_temp", default=0.07, type=float,
                        help="Temperature tau for SupCon loss (default: 0.07)")
    parser.add_argument("--supcon_mode", default='exact', choices=['exact', 'hierarchical'], type=str,
                        help="Matching strategy for SupCon loss: 'exact' (Exact BIO matching) or 'hierarchical' (Entity-type weighted matching).")
    parser.add_argument("--supcon_alpha", default=0.3, type=float,
                        help="Weak positive weight for same entity-type but different BIO prefix in hierarchical SupCon (default: 0.3).")
    parser.add_argument("--output_dir", default='outputs/', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for PhoBERT (paper: 2e-5).")
    parser.add_argument("--classifier_learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for CRF and Linear layer (paper: 1e-3).")
    parser.add_argument("--epochs", default=20, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--early_stop", default=3.0, type=float,
                        help="Early stopping patience in epochs.")
    parser.add_argument("--run_test", action='store_true', default=False,
                        help="Whether not to run test after train")
    parser.add_argument("--no_cuda", action='store_true', default=False,
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--num_workers', type=int, default=2,
                        help="How many subprocesses to use for data loading. 0 means that the data will be loaded "
                             "in the main process.")
    parser.add_argument("--fp16", action='store_true', default=True,
                        help="Whether to use 16-bit (mixed) precision (through torch.cuda.amp)")
    parser.add_argument("--no_fp16", dest='fp16', action='store_false',
                        help="Disable 16-bit (mixed) precision")
    parser.add_argument('--save_step', type=int, default=20000,
                        help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser.parse_args()
