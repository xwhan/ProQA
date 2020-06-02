import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--bert_model_name",
                        default="bert-base-uncased", type=str)
    parser.add_argument("--output_dir", default="logs", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")

    # Other parameters
    parser.add_argument("--load", default=False, action='store_true')
    parser.add_argument("--num_workers", default=5, type=int)
    parser.add_argument("--train_file", type=str,
                        default="../../data/mrqa-train/HotpotQA-tokenized.jsonl")
    parser.add_argument("--predict_file", type=str,
                        default="../../data/mrqa-dev/HotpotQA-tokenized.jsonl")
    parser.add_argument("--init_checkpoint", type=str,
                        help="Initial checkpoint (usually from a pre-trained BERT model).",
                        default="")
    parser.add_argument("--do_lower_case", default=True, action='store_true',
                        help="Whether to lower case the input text. Should be True for uncased"
                             "models and False for cased models.")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_query_length", default=50, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", default=False,
                        action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", default=False,
                        action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=8,
                        type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=100,
                        type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5,
                        type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=200, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--wait_step', type=int, default=100)
    parser.add_argument("--save_checkpoints_steps", default=1000, type=int,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--iterations_per_loop", default=1000, type=int,
                        help="How many steps to make in each estimator call.")
    parser.add_argument("--no_cuda", default=False, action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument('--seed', type=int, default=3,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument('--eval_period', type=int, default=1000, help="setting to -1: eval only after each epoch")
    parser.add_argument('--verbose', action="store_true", default=False)
    parser.add_argument('--efficient_eval', action="store_true", help="whether to use fp16 for evaluation")
    parser.add_argument('--max_answer_len', default=20, type=int)
    parser.add_argument("--max_grad_norm", default=5.0, type=float, help="Max gradient norm.")

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")

    # BERT QA 
    parser.add_argument("--qa-drop", default=0, type=float)
    parser.add_argument("--rank-drop", default=0, type=float)
    
    parser.add_argument("--MI", action="store_true", help="Use MI regularization to improve weak supervision")
    parser.add_argument("--mi-k", default=10, type=int, help="negative sample number")
    parser.add_argument("--max-pool", action="store_true", help="CLS or maxpooling")
    parser.add_argument("--eval-workers", default=16, help="parallel data loader", type=int)
    parser.add_argument("--save-pred", action="store_true",  help="uncertainty analysis")
    parser.add_argument("--retriever-path", type=str, default="", help="pretrained retriever checkpoint")

    parser.add_argument("--raw-train-data", type=str,
                        default="../data/nq-train.txt")
    parser.add_argument("--raw-eval-data", type=str,
                        default="../data/nq-dev.txt")
    parser.add_argument("--fix-para-encoder", action="store_true")
    parser.add_argument("--db-path", type=str,
                        default='../data/nq_paras.db')
    parser.add_argument("--index-path", type=str,
                        default="retrieval/index_data/para_embed_100k.npy")
    parser.add_argument("--matched-para-path", type=str,
                        default="../data/wq_ft_train_matched.txt")
                    
    parser.add_argument("--use-spanbert", action="store_true", help="use spanbert for question answering")
    parser.add_argument("--spanbert-path",
                        default="../data/span_bert", type=str)
    parser.add_argument("--eval-k", default=5, type=int)
    parser.add_argument("--regex", action="store_true", help="for CuratedTrec")

    # investigate different kinds of loss functions
    parser.add_argument("--separate", action="store_true", help="separate the rank and reader loss")
    parser.add_argument("--add-select", action="store_true", help="replace the rank probability with the selection probility from the reader model ([CLS])")
    parser.add_argument("--drop-early", action="store_true", help="drop the early loss on topk5000")
    parser.add_argument("--shared-norm", action="store_true",
                        help="normalize span logits across different paragraphs")

#     parser.add_argument("--fix-retriever", action="store_true")
#     parser.add_argument("--joint-train", action="store_true")
#     parser.add_argument("--mixed", action="store_true",help="shared norm and also use the rank probabilities in loss")
#     parser.add_argument("--use-adam", action="store_true")
#     parser.add_argument("--para-embed-path", type=str, default="")
#     parser.add_argument("--retrieved-path", type=str, default="")

    # For evaluation
    parser.add_argument('--prefix', type=str, default="eval")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--use-top-passage', action="store_true")
    parser.add_argument('--topk', default=30, type=int)
    parser.add_argument('--save-all', action="store_true", help="save the predictions")
    parser.add_argument('--candidates', default="", type=str, help="restrict the predicted spans to be entities")

    args = parser.parse_args()

    return args
