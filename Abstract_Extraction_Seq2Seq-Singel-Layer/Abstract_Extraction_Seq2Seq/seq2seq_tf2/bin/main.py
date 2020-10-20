# coding=utf-8
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(BASE_DIR)
sys.path.append(BASE_DIR)
import tensorflow as tf
import argparse
from seq2seq_tf2.train_eval_test import train, test, evaluate
from utils.data_utils import get_result_filename
# from utils.log_utils import define_logger
import pathlib

NUM_SAMPLES = 82706

# 获取项目根目录
# root = pathlib.Path(os.path.abspath(__file__)).parent.parent


def main():
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--max_enc_len", default=200, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=40, help="Decoder input max sequence length", type=int)
    parser.add_argument("--max_dec_steps", default=100,
                        help="maximum number of words of the predicted abstract", type=int)
    parser.add_argument("--min_dec_steps", default=30,
                        help="Minimum number of words of the predicted abstract", type=int)
    parser.add_argument("--batch_size", default=64, help="batch size", type=int)
    parser.add_argument("--beam_size", default=3,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)",
                        type=int)
    parser.add_argument("--vocab_size", default=30000, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=256, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256,
                        help="[context vector, decoder state, decoder input] feedforward result dimension - "
                             "this result is used to compute the attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer "
                             "API documentation on tensorflow site for more details.", type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped",
                        type=float)
    parser.add_argument('--cov_loss_wt', default=0.5, help='Weight of coverage loss (lambda in the paper).'
                                                           ' If zero, then no incentive to minimize coverage loss.',
                        type=float)

    # path
    # /ckpt/checkpoint/checkpoint
    parser.add_argument("--seq2seq_model_dir", default='{}/ckpt/seq2seq'.format(BASE_DIR), help="Model folder")
    parser.add_argument("--pgn_model_dir", default='{}/ckpt/pgn'.format(BASE_DIR), help="Model folder")
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--train_seg_x_dir", default='{}/datasets/train_set.seg_x.txt'.format(BASE_DIR), help="train_seg_x_dir")
    parser.add_argument("--train_seg_y_dir", default='{}/datasets/train_set.seg_y.txt'.format(BASE_DIR), help="train_seg_y_dir")
    parser.add_argument("--test_seg_x_dir", default='{}/datasets/test_set.seg_x.txt'.format(BASE_DIR), help="test_seg_x_dir")
    parser.add_argument("--vocab_path", default='{}/datasets/vocab.txt'.format(BASE_DIR), help="Vocab path")
    parser.add_argument("--word2vec_output", default='{}/datasets/word2vec.txt'.format(BASE_DIR), help="Vocab path")
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)
    parser.add_argument("--test_save_dir", default='{}/datasets/'.format(BASE_DIR), help="test_save_dir")
    parser.add_argument("--test_x_dir", default='{}/datasets/AutoMaster_TestSet.csv'.format(BASE_DIR), help="test_x_dir")

    # others
    parser.add_argument("--steps_per_epoch", default=1300, help="max_train_steps", type=int)
    parser.add_argument("--checkpoints_save_steps", default=10, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--max_steps", default=10000, help="Max number of iterations", type=int)
    parser.add_argument("--num_to_test", default=10, help="Number of examples to test", type=int)
    parser.add_argument("--max_num_to_eval", default=5, help="max_num_to_eval", type=int)
    parser.add_argument("--epochs", default=20, help="train epochs", type=int)
    
    # mode
    parser.add_argument("--mode", default='train', help="training, eval or test options")
    parser.add_argument("--model", default='SequenceToSequence', help="which model to be slected")
    parser.add_argument("--greedy_decode", default=True, help="greedy_decoder")
    parser.add_argument("--beam_search_decode", default=True, help="beam_search_decode")
    parser.add_argument("--decoder_mode", default="beam_search_decode", help="decoder mode")

    parser.add_argument("--repetition_penalty", default=1.5, help="decoder mode")   # 增加重复惩罚项

    args = parser.parse_args()
    params = vars(args)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')

    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        # tf.config.experimental.set_visible_devices(devices=gpus[1], device_type='GPU')
        # tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_memory_growth(gpus[1], True)
    if params["mode"] == "train":
        params["steps_per_epoch"] = NUM_SAMPLES//params["batch_size"]
        train(params)
    
    elif params["mode"] == "eval":
        evaluate(params)

    elif params["mode"] == "test":
        # params["batch_size"] = params["beam_size"]
        if params["decoder_mode"] == "beam_search_decode":
            params["batch_size"] = 10
        test(params)


if __name__ == '__main__':
    main()
