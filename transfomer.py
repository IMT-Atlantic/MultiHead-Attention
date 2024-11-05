import tensorflow as tf
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K
import numpy as np

# 实现位置前馈网络，是每个编码器和解码器层的另一个关键组成部分
class PositionWiseFeedForward(Layer):
    # 构造方法 确定几个重点传入参数 trainable=True明确参数是否可训练 其他关键词参数通过**kwargs传递给父类
    def __init__(self, model_dim, inner_dim, trainable=True, **kwargs):
        self._model_dim = model_dim
        self._inner_dim = inner_dim
        self._trainable = trainable
        super(PositionWiseFeedForward, self).__init__(**kwargs)

    # 构建方法
    def build(self, input_shape):
        # 确定输入与输出权重以及隐藏层、输出层的偏置项
        self.weights_inner = self.add_weight(
            shape=(input_shape[-1], self._inner_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_inner")

        self.weights_out = self.add_weight(
            shape=(self._inner_dim, self._model_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name="weights_out")

        self.bais_inner = self.add_weight(
            shape=(self._inner_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_inner")

        self.bais_out = self.add_weight(
            shape=(self._model_dim,),
            initializer='uniform',
            trainable=self._trainable,
            name="bais_out")
        super(PositionWiseFeedForward, self).build(input_shape)

    def call(self, inputs):
        if K.dtype(inputs) != 'float32':
            inputs = K.cast(inputs, 'float32')
        inner_out = K.relu(K.dot(inputs, self.weights_inner) + self.bais_inner)
        outputs = K.dot(inner_out, self.weights_out) + self.bais_out
        return outputs

    def compute_output_shape(self, input_shape):
        return self._model_dim

# 为输入添加位置编码信息，以便Transformer能够利用序列顺序
class PositionEncoding(Layer):

    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i%2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape

# 输入层的归一化
class LayerNormalization(Layer):

    def __init__(self, epsilon=1e-8, **kwargs):
        self._epsilon = epsilon
        super(LayerNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zero',
            name='beta')
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='one',
            name='gamma')
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** 0.5)
        outputs = self.gamma * normalized + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

# 缩放点积注意力机制 Transformer核心组件
# 核心操作是计算 queries 和 keys 之间的点积，然后使用 softmax 计算权重，并使用这些权重和 values 进行加权求和
class ScaledDotProductAttention(Layer):
    def __init__(self, masking=True, future=False, dropout_rate=0., **kwargs):
        self._masking = masking
        self._future = future
        self._dropout_rate = dropout_rate
        self._masking_num = -2 ** 32 + 1
        super(ScaledDotProductAttention, self).__init__(**kwargs)

    # 应用掩码（mask）以隐藏某些元素
    def mask(self, inputs, masks):
        masks = K.cast(masks, 'float32')
        masks = K.tile(masks, [K.shape(inputs)[0] // K.shape(masks)[0], 1])
        masks = K.expand_dims(masks, 1)
        outputs = inputs + masks * self._masking_num
        return outputs

    # 应用掩码以隐藏未来信息（仅在自回归模型中使用）
    def future_mask(self, inputs):
        diag_vals = tf.ones_like(inputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        future_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(inputs)[0], 1, 1])
        paddings = tf.ones_like(future_masks) * self._masking_num
        outputs = tf.where(tf.equal(future_masks, 0), paddings, inputs)
        return outputs

    # 核心计算model
    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        if K.dtype(queries) != 'float32':  queries = K.cast(queries, 'float32')
        if K.dtype(keys) != 'float32':  keys = K.cast(keys, 'float32')
        if K.dtype(values) != 'float32':  values = K.cast(values, 'float32')

        matmul = K.batch_dot(queries, tf.transpose(keys, [0, 2, 1]))  # MatMul
        # print type
        print(f"matmul type: {type(matmul)}, shape: {matmul.shape}")
        scaled_matmul = matmul / int(queries.shape[-1]) ** 0.5  # Scale
        print(f"scaled_matmul type: {type(scaled_matmul)}, shape: {scaled_matmul.shape}")
        if self._masking:
            scaled_matmul = self.mask(scaled_matmul, masks)  # Mask(opt.)
            print(f"masked scaled_matmul type: {type(scaled_matmul)}, shape: {scaled_matmul.shape}")

        if self._future:
            scaled_matmul = self.future_mask(scaled_matmul)
            print(f"future masked scaled_matmul type: {type(scaled_matmul)}, shape: {scaled_matmul.shape}")

        softmax_out = K.softmax(scaled_matmul)  # SoftMax
        print(f"softmax_out type: {type(softmax_out)}, shape: {softmax_out.shape}")
        # Dropout
        out = K.dropout(softmax_out, self._dropout_rate)
        print(f"dropout out type: {type(out)}, shape: {out.shape}")

        outputs = K.batch_dot(out, values) # MatMul
        print(f"outputs type: {type(outputs)}, shape: {outputs.shape}")

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

# 多头注意力机制：实现多头注意力机制，每个头计算一个缩放点积注意力，并最终将它们连接在一起
class MultiHeadAttention(Layer):

    def __init__(self, n_heads, head_dim, dropout_rate=.1, masking=False, future=False, trainable=True, **kwargs):
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dropout_rate = dropout_rate
        self._masking = masking
        self._future = future
        self._trainable = trainable
        super(MultiHeadAttention, self).__init__(**kwargs)

    # 创建查询、键、值的权重矩阵
    def build(self, input_shape):
        self._weights_queries = self.add_weight(
            shape=(input_shape[0][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_queries')
        self._weights_keys = self.add_weight(
            shape=(input_shape[1][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_keys')
        self._weights_values = self.add_weight(
            shape=(input_shape[2][-1], self._n_heads * self._head_dim),
            initializer='glorot_uniform',
            trainable=self._trainable,
            name='weights_values')
        super(MultiHeadAttention, self).build(input_shape)

    # 多头注意力的计算
    def call(self, inputs):
        if self._masking:
            assert len(inputs) == 4, "inputs should be set [queries, keys, values, masks]."
            queries, keys, values, masks = inputs
        else:
            assert len(inputs) == 3, "inputs should be set [queries, keys, values]."
            queries, keys, values = inputs

        queries_linear = K.dot(queries, self._weights_queries)
        keys_linear = K.dot(keys, self._weights_keys)
        values_linear = K.dot(values, self._weights_values)

        queries_multi_heads = tf.concat(tf.split(queries_linear, self._n_heads, axis=2), axis=0)
        keys_multi_heads = tf.concat(tf.split(keys_linear, self._n_heads, axis=2), axis=0)
        values_multi_heads = tf.concat(tf.split(values_linear, self._n_heads, axis=2), axis=0)

        print(f"queries_multi_heads type: {type(queries_multi_heads)}, shape: {queries_multi_heads.shape}")
        print(f"keys_multi_heads type: {type(keys_multi_heads)}, shape: {keys_multi_heads.shape}")
        print(f"values_multi_heads type: {type(values_multi_heads)}, shape: {values_multi_heads.shape}")

        if self._masking:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads, masks]
        else:
            att_inputs = [queries_multi_heads, keys_multi_heads, values_multi_heads]

        attention = ScaledDotProductAttention(
            masking=self._masking, future=self._future, dropout_rate=self._dropout_rate)
        att_out = attention(att_inputs)

        print(f"att_out type before concat: {type(att_out)}, shape: {att_out}")

        # Ensure att_out is always a tensor
        if isinstance(att_out, list):
            att_out = tf.concat(att_out, axis=-1)

        # Ensure output concatenation
        outputs = tf.concat(tf.split(att_out, self._n_heads, axis=0), axis=2)
        print(f"outputs type: {type(outputs)}, shape: {outputs.shape}")

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

# 简单封装tf.nn.dropout以用于Dropout操作
class DropoutWrapper(Layer):
    def __init__(self, rate, **kwargs):
        super(DropoutWrapper, self).__init__(**kwargs)
        self.rate = rate

    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)

# 打包
class Transformer_Merged(tf.Module):
    def __init__(self, model_dim=64, n_heads=8,
                 encoder_stack=3,
                 feed_forward_size=128,
                 dropout_rate=0.1):
        super(Transformer_Merged, self).__init__()
        self._encoder_stack = encoder_stack
        self._model_dim = model_dim
        self._n_heads = n_heads
        self._feed_forward_size = feed_forward_size
        self._dropout_rate = dropout_rate

        self.positionencodeing = PositionEncoding(self._model_dim)
        self.feedforward = PositionWiseFeedForward(self._model_dim, self._feed_forward_size)
        self.attention = MultiHeadAttention(self._n_heads, self._model_dim // self._n_heads)
        self.dropout_layer = DropoutWrapper(self._dropout_rate)

    def __call__(self, inputs):
        embeddings = inputs

        # Position Encodings
        position_encodings = self.positionencodeing(embeddings)
        # Embedings + Position-encodings
        encodings = embeddings + position_encodings
        # Dropout
        encodings = self.dropout_layer(encodings)

        for i in range(self._encoder_stack):
            # Multi-head Attention
            attention_input = [encodings, encodings, encodings]
            attention_out = self.attention(attention_input)
            # Print data type
            print(f"type of attention_out after attention: {type(attention_out)}, content: {attention_out}")

            # Add & Norm
            attention_out += encodings
            # Print data type
            print(f"type of combined attention_out: {type(attention_out)}, content: {attention_out}")
            attention_out = LayerNormalization()(attention_out)

            # Feed-Forward
            ff_out = self.feedforward(attention_out)

            # Add & Norm
            ff_out += attention_out
            encodings = LayerNormalization()(ff_out)

        return encodings