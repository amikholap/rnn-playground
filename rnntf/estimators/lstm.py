import tensorflow as tf


def model_fn(features, labels, mode, params):
    feature_columns = params['feature_columns']
    hidden_units = params['hidden_units']
    output_units = params['output_units']

    input_layer, sequence_length = tf.contrib.feature_column.sequence_input_layer(
        features,
        feature_columns,
    )

    cell = tf.contrib.rnn.LSTMCell(hidden_units)
    outputs, _ = tf.nn.dynamic_rnn(
        cell,
        inputs=input_layer,
        sequence_length=sequence_length,
        dtype=tf.float32,
    )
    output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_units])

    with tf.variable_scope('LSTMEstimator', reuse=tf.AUTO_REUSE):
        softmax_w = tf.get_variable(
            'softmax_w',
            [hidden_units, output_units],
            dtype=tf.float32,
        )
        softmax_b = tf.get_variable(
            'softmax_b',
            [output_units],
            dtype=tf.float32,
        )

    logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
    logits = tf.reshape(logits, tf.shape(input_layer))
    predictions = tf.argmax(logits, 2)  # pylint: disable=unused-variable

    if labels is not None:
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            labels,
            tf.ones(tf.shape(labels), dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True,
        )
        loss = tf.reduce_sum(loss)
    else:
        loss = None

    if mode == tf.estimator.ModeKeys.PREDICT:
        spec = tf.estimator.EstimatorSpec(
            mode,
            predictions={
                'predictions': predictions,
            },
        )
    elif mode == tf.estimator.ModeKeys.EVAL:
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
        )
    elif mode == tf.estimator.ModeKeys.TRAIN:
        learning_rate = tf.train.exponential_decay(
            learning_rate=0.1,
            global_step=tf.train.get_global_step(),
            decay_steps=100,
            decay_rate=0.95,
        )
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
        )
    else:
        raise RuntimeError('unreachable')

    return spec


class LSTMEstimator(tf.estimator.Estimator):

    def __init__(self, hidden_units, output_units, feature_columns):
        super().__init__(
            model_fn=model_fn,
            params={
                'hidden_units': hidden_units,
                'output_units': output_units,
                'feature_columns': feature_columns,
            },
        )
