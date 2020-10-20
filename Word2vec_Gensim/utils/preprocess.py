import pandas as pd

from utils.tokenizer import segment
from config.config import seg_train_x_data_path, seg_train_y_data_path, seg_test_x_data_path


#设置显示的最大列、宽等参数，消除打印不完全中间的省略号
pd.set_option("display.width", 1000)
#加了这一行那表格就不会分段出现了
pd.set_option("display.width", 1000)
#显示所有列
pd.set_option("display.max_columns", None)
#显示所有行
pd.set_option("display.max_rows", None)


REMOVE_WORDS = ['|', '[', ']', '语音', '图片', ' ']


def read_stopwords(path):
    lines = set()
    with open(path, mode='r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            lines.add(line)
    return lines


def remove_words(words_list):
    # 删除不需要的单词
    words_list = [word for word in words_list if word not in REMOVE_WORDS]
    return words_list


def dataset_info(train_path):
    train_df = pd.read_csv(train_path, encoding="utf-8")
    print(train_df.head(20))


def parse_data(train_path, test_path):
    # 训练集数据的分词
    train_df = pd.read_csv(train_path, encoding='utf-8')
    train_df.dropna(subset=['Report'], how='any', inplace=True)     # 删除没有标签的行,Report属于标签列,若找不到标签则删除
    train_df.fillna('', inplace=True)                               # 填充其他列中的缺失值
    # print(train_df.head(20))
    train_x = train_df.Question.str.cat(train_df.Dialogue)          # Question与Dialogue两列的字符进行拼接
    train_x = train_x.apply(preprocess_sentence)                    # 字符串分词
    print('train_x is ', len(train_x))
    train_y = train_df.Report
    train_y = train_y.apply(preprocess_sentence)
    print('train_y is ', len(train_y))

    # 对测试集做同样的处理
    test_df = pd.read_csv(test_path, encoding='utf-8')
    test_df.fillna('', inplace=True)
    test_x = test_df.Question.str.cat(test_df.Dialogue)
    test_x = test_x.apply(preprocess_sentence)
    print('test_x is ', len(test_x))
    train_x.to_csv(seg_train_x_data_path, index=None, header=False)
    train_y.to_csv(seg_train_y_data_path, index=None, header=False)
    test_x.to_csv(seg_test_x_data_path, index=None, header=False)


def preprocess_sentence(sentence):
    seg_list = segment(sentence.strip(), cut_type='word')
    seg_list = remove_words(seg_list)
    seg_line = ' '.join(seg_list)
    return seg_line


if __name__ == '__main__':
    # 需要更换成自己数据的存储地址
    from config.config import src_train_data_path, src_test_data_path
    parse_data(src_train_data_path, src_test_data_path)
    # dataset_info(src_train_data_path)

