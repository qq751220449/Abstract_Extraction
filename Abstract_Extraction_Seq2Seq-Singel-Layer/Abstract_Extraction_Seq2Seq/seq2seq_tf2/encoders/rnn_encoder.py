import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class Encoder(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz, embedding_matrix):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        # self.enc_units = enc_units
        self.enc_units = enc_units // 2
        """
        定义Embedding层，加载预训练的词向量
        your code
        """
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, weights=[embedding_matrix], trainable=False)
        # tf.keras.layers.GRU自动匹配cpu、gpu
        """
        定义单向的RNN、GRU、LSTM层
        your code
        """
        self.gru = tf.keras.layers.GRU(units=self.enc_units,        # 设定隐层状态数
                                       return_sequences=True,       # 需要将状态进行返回
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform',
                                       )
        self.bigru = tf.keras.layers.Bidirectional(self.gru, merge_mode='concat')

    def call(self, x, hidden):
        x = self.embedding(x)
        hidden = tf.split(hidden, num_or_size_splits=2, axis=1)
        output, forward_state, backward_state = self.bigru(x, initial_state=hidden)
        state = tf.concat([forward_state, backward_state], axis=1)
        # output, state = self.gru(x, initial_state=hidden)
        # print(output.shape, forward_state.shape, backward_state.shape)
        # output [batch_size, seq_len, hidden_size]
        # forward_state 前向传递过程中的最后时刻的隐层状态输出   [batch_size, hidden_size]
        # backward_state 反向传递过程中的最后时刻的隐层状态输出  [batch_size, hidden_size]
        # state [batch_size, 2 * hidden_size]
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, 2*self.enc_units))


if __name__ == "__main__":
    import numpy as np
    embedding_matrix = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7]], dtype=np.float64)
    net = Encoder(5, 3, 4, 2, embedding_matrix)
    hidden = net.initialize_hidden_state()
    word_id = tf.convert_to_tensor([[1, 2, 0], [1, 0, 0]], dtype=tf.int64)
    word_embed = net(word_id, hidden)
    print(word_embed)
