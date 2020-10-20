import tensorflow as tf
from tqdm import tqdm
import numpy as np


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, state):
        # list of all the tokens from time 0 to the current time step t
        self.tokens = tokens
        # list of the log probabilities of the tokens of the tokens
        self.log_probs = log_probs
        # decoder state after the last token decoding
        self.state = state

    def extend(self, token, log_prob, state):
        """Method to extend the current hypothesis by adding the next decoded token and all
        the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          state=state,  # we update the state
                          )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def beam_decode(model, dataset, vocab, params):
    # 存储结果
    batch_size = params["batch_size"]
    result = []
    sample_size = 30
    step_epoch = sample_size // batch_size + 1
    dataset_iter = iter(dataset)
    for i in tqdm(range(step_epoch)):
        try:
            enc_data, _ = next(dataset_iter)
        except Exception as Error:
            print("Stopping")
            break
        # print(enc_data["enc_input"])
        result += batch_beam_decode(model, enc_data, vocab, params)
    return result


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty):
    # create logit penalties for already seen input_ids
    token_penalties = np.ones(shape_list(logits))
    prev_input_ids = [np.unique(input_id) for input_id in input_ids.numpy()]
    for i, prev_input_id in enumerate(prev_input_ids):
        logit_penalized = logits[i].numpy()[prev_input_id]
        logit_penalties = np.zeros(logit_penalized.shape)
        # if previous logit score is < 0 then multiply repetition penalty else divide
        logit_penalties[logit_penalized < 0] = repetition_penalty
        logit_penalties[logit_penalized > 0] = 1 / repetition_penalty
        np.put(token_penalties[i], prev_input_id, logit_penalties)
    return tf.convert_to_tensor(token_penalties, dtype=tf.float32)


def batch_beam_decode(model, batch, vocab, params):

    def decode_onestep(input_ids, enc_outputs, dec_inputs, dec_hiddens, k=params["beam_size"]):
        context_vector, _ = model.attention(dec_hiddens, enc_outputs)
        _, logits, dec_hiddens = model.decoder(dec_inputs,
                                               dec_hiddens,
                                               enc_outputs,
                                               context_vector)

        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if params["repetition_penalty"] != 1.0:
            next_token_logits_penalties = _create_next_token_logits_penalties(
                input_ids, logits, params["repetition_penalty"]
            )
            next_token_logits = tf.math.multiply(logits, next_token_logits_penalties)
            logits = next_token_logits

        pred = tf.keras.activations.softmax(logits, axis=1)

        top_k_probs, top_k_ids = tf.nn.top_k(pred, k=k)
        top_k_log_probs = tf.math.log(top_k_probs)
        # top_k_log_probs   shape=[batch_size*beam_size,3]
        # top_k_ids         shape=[batch_size*beam_size,3]
        # dec_hiddens       shape=[batch_size*beam_size, 256]
        return top_k_log_probs, top_k_ids, dec_hiddens

    # 输入数据
    batch_data = batch["enc_input"]
    # 输入batch大小
    batch_size = batch_data.shape[0]

    # 文本数据encoder
    enc_output, enc_hidden = model.call_encoder(batch_data)   # 输入文本编码
    # enc_output  shape=[batch_size, seq_len, hidden_size](3, 115, 256)     enc_hidden  shape=(3, 256)
    # print(enc_output[1])

    # 将enc_output进行复制  变换之后shape=[batch_size*beam_size, 115, 256]
    enc_output = tf.concat([tf.stack([enc_output[i, ...] for _ in range(params["beam_size"])], axis=0) for i in range(enc_output.shape[0])], axis=0)

    # encoder隐层复制到decoder初始隐层
    dec_hidden = enc_hidden

    # 为了计算方便,将Batch中的每条数据,复制beam_size份
    hyps = [[Hypothesis(tokens=[vocab.word_to_id('[START]')], log_probs=[0.0], state=dec_hidden[i])
             for _ in range(params['beam_size'])] for i in range(batch_size)]

    for step in range(params["max_dec_steps"]):
        # 构建隐层状态输入变量  shape=[batch_size*beam_size, 256]
        dec_hiddens = tf.concat([tf.stack([h.state for h in beam], axis=0) for beam in hyps], axis=0)

        # 构建Batch_size*beam_size的输入 shape=[batch_size*beam_size, 1]
        dec_inputs = tf.concat([tf.stack([[h.latest_token] for h in beam], axis=0) for beam in hyps], axis=0)

        input_ids = tf.concat([tf.stack([[h.tokens] for h in beam], axis=0) for beam in hyps], axis=0)
        # 单步解码操作
        top_k_log_probs, top_k_ids, dec_hiddens = decode_onestep(input_ids, enc_output, dec_inputs, dec_hiddens, k=params["beam_size"]*2)
        # print(top_k_log_probs)
        # print(top_k_ids)
        # print(dec_hiddens)

        # 预测结果按照batch_size大小进行切分
        top_k_log_probs_list = tf.split(top_k_log_probs, axis=0, num_or_size_splits=batch_size)
        # print(top_k_log_probs_list[0])
        # print(top_k_log_probs_list.shape)
        top_k_ids_list = tf.split(top_k_ids, axis=0, num_or_size_splits=batch_size)
        dec_hiddens_list = tf.split(dec_hiddens, axis=0, num_or_size_splits=batch_size)
        # print(top_k_ids_list[0])

        # 依次处理每个Batch
        for bc in range(batch_size):
            bc_hyps = hyps[bc]   # 对应batch的状态保存变量
            bc_top_k_log_probs = top_k_log_probs_list[bc]
            bc_top_k_ids = top_k_ids_list[bc]
            bc_dec_hidden = dec_hiddens_list[bc]
            # print(bc_top_k_ids)
            # print(len(bc_hyps))
            all_hyps = []         # 存放所有更新后的hyps
            num_orig_hyps = 1 if step == 0 else len(bc_hyps)
            # print(num_orig_hyps)
            for i in range(num_orig_hyps):
                h, new_state = bc_hyps[i], bc_dec_hidden[i]   # 隐层
                for j in range(params["beam_size"]):
                    new_h = h.extend(
                        token=bc_top_k_ids[i][j].numpy(),
                        log_prob=bc_top_k_log_probs[i][j].numpy(),
                        state=new_state
                    )
                    all_hyps.append(new_h)

            bc_hyps = []   # batch中的句子按照概率排序

            sorted_all_hyps = sorted(all_hyps, key=lambda h:h.avg_log_prob, reverse=True)

            for index, h in enumerate(sorted_all_hyps):
                bc_hyps.append(h)
                # print(h.tokens)
                if index == params["beam_size"] - 1:   # 句子个数满足要求
                    break

            # 更新
            hyps[bc] = bc_hyps

    results = [[]] * batch_size  # 存放每个Batch的结果
    # 获取最优的结果
    for bc in range(batch_size):
        bc_hyps = hyps[bc]
        # 优先选取有结束符的结果
        for i in range(params["beam_size"]):
            hyp = bc_hyps[i]
            tokens = hyp.tokens[:]
            if vocab.word_to_id('[STOP]') in tokens:
                tokens = tokens[1:tokens.index(vocab.word_to_id('[STOP]'))]
                # 有结束符且满足最小长度要求
                if len(tokens) > params["min_dec_steps"]:
                    results[bc] = tokens
                    break
        # 如果找到了满足要求的结果，则直接处理下一句
        if results[bc]:
            continue
        # 若在上述条件下未找到合适结果，则只找没有结束符的结果
        for i in range(params["beam_size"]):
            hyp = bc_hyps[i]
            tokens = hyp.tokens[:]
            if vocab.word_to_id('[STOP]') in tokens:  # 长度不合适的情况,跳过
                continue
            results[bc] = tokens[1:]

    def get_abstract(tokens):
        return " ".join([vocab.id_to_word(i) for i in tokens])

    return [get_abstract(token) for token in results]


