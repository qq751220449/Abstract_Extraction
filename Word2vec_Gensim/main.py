import os
from utils.data_reader import vocab_create
from utils.preprocess import parse_data
from config.config import src_train_data_path, src_test_data_path
from config.config import seg_train_x_data_path, seg_train_y_data_path, seg_test_x_data_path
from config.config import word2vec_path, sentence_path
from utils.build_w2v import build


def main():
    if not os.path.exists(seg_train_x_data_path) and not os.path.exists(seg_train_y_data_path) and not os.path.exists(seg_test_x_data_path):
        # 若文件不存在,则运行代码生成分词后的文件
        print("test")
        parse_data(src_train_data_path, src_test_data_path)
    vocab_create(seg_train_x_data_path, seg_train_y_data_path, seg_test_x_data_path, use_Counter=True)          # 构建词向量映射
    build(seg_train_x_data_path, seg_train_y_data_path, seg_test_x_data_path, word2vec_path, sentence_path)     # 训练词向量


if __name__ == "__main__":
    main()
