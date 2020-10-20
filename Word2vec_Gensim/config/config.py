import os

# 原始数据集路径设置
src_dataset_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../dataset/src_dataset"))
src_train_data_name = "AutoMaster_TrainSet.csv"
src_test_data_name = "AutoMaster_TestSet.csv"
src_train_data_path = os.path.join(src_dataset_path, src_train_data_name)   # train文件所在路径
src_test_data_path = os.path.join(src_dataset_path, src_test_data_name)     # test文件所在路径

# segment数据集保存路径
seg_dataset_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../dataset/seg_dataset"))
if not os.path.exists(seg_dataset_path):
    # 路径不存在,则新建路径
    os.mkdir(seg_dataset_path)
seg_train_x_data_path = os.path.join(seg_dataset_path, "train_set.seg_x.txt")
seg_train_y_data_path = os.path.join(seg_dataset_path, "train_set.seg_y.txt")
seg_test_x_data_path = os.path.join(seg_dataset_path, "test_set.seg_x.txt")

# vocab保存路径
vocab_save_path = os.path.join(seg_dataset_path, "vocab.txt")
reverse_vocab_save_path = os.path.join(seg_dataset_path, "reverse_vocab.txt")

# word2vec保存路径
word2vec_save_path = os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../dataset/word2vec_model"))
if not os.path.exists(word2vec_save_path):
    # 路径不存在,则新建路径
    os.mkdir(word2vec_save_path)
word2vec_path = os.path.join(word2vec_save_path, "word2vec.txt")
sentence_path = os.path.join(word2vec_save_path, "sentences.txt")

