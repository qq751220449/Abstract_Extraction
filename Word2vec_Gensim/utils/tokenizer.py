import jieba
from jieba import posseg
# from LAC import LAC

# lac = LAC(mode='seg')


# def segment_line(line):
#     tokens = jieba.cut(line, cut_all=False)
#     return " ".join(tokens)
#
#
# def process_line(line):
#     if isinstance(line, str):
#         tokens = line.split("|")
#         result = [segment_line(t) for t in tokens]
#         return " | ".join(result)


def segment(sentence, cut_type='word', pos=False):
    """
    切词
    :param sentence:
    :param cut_type: 'word' use jieba.lcut; 'char' use list(sentence)
    :param pos: enable POS   # 词性标注 (我, 'r')
    :return: list
    """
    if pos:
        if cut_type == 'word':                      # 单词模式
            word_pos_seq = posseg.lcut(sentence)    # 词性标注
            word_seq, pos_seq = [], []
            for w, p in word_pos_seq:
                word_seq.append(w)                  # 单词列表
                pos_seq.append(p)                   # 词性列表
            return word_seq, pos_seq
        elif cut_type == 'char':                    # 字符模式
            word_seq = list(sentence)
            pos_seq = []
            for w in word_seq:
                w_p = posseg.lcut(w)
                pos_seq.append(w_p[0].flag)
            return word_seq, pos_seq
    else:
        if cut_type == 'word':
            return jieba.lcut(sentence)             # 列表返回
        elif cut_type == 'char':
            return list(sentence)
