from collections import defaultdict
import os
from collections import Counter
from config.config import vocab_save_path, reverse_vocab_save_path


def save_word_dict(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            # print(line)
            w, i = line
            f.write("%s\t%d\n" % (w, i))


def save_index_word_list(vocab, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in vocab:
            # print(line)
            i, w = line
            f.write("%d\t%s\n" % (i, w))


def read_data(path_1, path_2, path_3):
    with open(path_1, 'r', encoding='utf-8') as f1, \
            open(path_2, 'r', encoding='utf-8') as f2, \
            open(path_3, 'r', encoding='utf-8') as f3:
        words = []
        # print(f1)
        for line in f1:
            words += line.split()

        for line in f2:
            words += line.split(' ')

        for line in f3:
            words += line.split(' ')

    return words


def build_vocab(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # sort by count
        dic = defaultdict(int)
        for item in items:
            for i in item.split(" "):
                i = i.strip()
                if not i: continue
                i = i if not lower else item.lower()
                dic[i] += 1
        # sort
        """
        按照字典里的词频进行排序，出现次数多的排在前面
        your code(one line)
        """
        """*********************************************"""
        dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)     # 按照词频降序排序
        """*********************************************"""
        # print("After sorted, dic is: \n", dic)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    your code
    vocab = (one line)
    reverse_vocab = (one line)
    """
    """*********************************************"""
    vocab = [(w, c) for c, w in enumerate(result)]
    reverse_vocab = [(c, w) for c, w in enumerate(result)]
    """*********************************************"""
    # print(vocab, "\n", reverse_vocab)
    return vocab, reverse_vocab


def build_vocab_with_counter(items, sort=True, min_count=0, lower=False):
    """
    构建词典列表
    :param items: list  [item1, item2, ... ]
    :param sort: 是否按频率排序，否则按items排序
    :param min_count: 词典最小频次
    :param lower: 是否小写
    :return: list: word set
    """
    result = []
    if sort:
        # 使用Counter会导致结果不一样,原因是这里没有去除换行符
        items_strip = [i.strip() for i in items if i.strip()]
        word_counts = Counter(items_strip)
        # sort
        """
        按照字典里的词频进行排序，出现次数多的排在前面
        your code(one line)
        """
        """*********************************************"""
        dic = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)     # 按照词频降序排序
        """*********************************************"""
        # print("After sorted, dic is: \n", dic)
        for i, item in enumerate(dic):
            key = item[0]
            if min_count and min_count > item[1]:
                continue
            result.append(key)
    else:
        # sort by items
        for i, item in enumerate(items):
            item = item if not lower else item.lower()
            result.append(item)
    """
    建立项目的vocab和reverse_vocab，vocab的结构是（词，index）
    your code
    vocab = (one line)
    reverse_vocab = (one line)
    """
    """*********************************************"""
    vocab = [(w, c) for c, w in enumerate(result)]
    reverse_vocab = [(c, w) for c, w in enumerate(result)]
    """*********************************************"""
    # print(vocab, "\n", reverse_vocab)
    return vocab, reverse_vocab


def vocab_create(path_1, path_2, path_3, use_Counter=False, sort=True, min_count=0, lower=False):
    # use_Counter=True 使用Counter统计词频
    words = read_data(path_1, path_2, path_3)
    if use_Counter:
        vocab, reverse_vocab = build_vocab_with_counter(words, sort=sort, min_count=min_count, lower=lower)
    else:
        vocab, reverse_vocab = build_vocab(words, sort=sort, min_count=min_count, lower=lower)
    save_word_dict(vocab, vocab_save_path)
    save_index_word_list(reverse_vocab, reverse_vocab_save_path)



