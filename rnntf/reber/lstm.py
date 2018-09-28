import tensorflow as tf

from .fsm import EmbeddedReberFSM, ReberFSM
from .input_providers import (
    FileReberInputProvider, FSMEmbeddedInputProvider, FSMReberInputProvider, ReberInputProvider,
)


def run(args):
    model = LSTMModel(
        hidden_size=args.hidden_size,
        output_size=ReberInputProvider.get_vocabulary_size(),
        is_training=True,
    )

    if args.dataset_path:
        input_ = FileReberInputProvider(args.dataset_path, batch_size=args.batch_size)
    elif args.embedded:
        input_ = FSMEmbeddedInputProvider(batch_size=args.batch_size)
    else:
        input_ = FSMReberInputProvider(batch_size=args.batch_size)

    if args.embedded:
        fsm_type = EmbeddedReberFSM
    else:
        fsm_type = ReberFSM

    with tf.Session() as session:
        run_epoch(
            session=session,
            model=model,
            input_=input_,
            fsm_type=fsm_type,
        )


def run_epoch(session, model, input_, fsm_type):
    checker_fsm = fsm_type()

    for i, (x, y) in enumerate(input_):
        print('=' * 32, 'Batch', i)

        result = model.process_batch(x, y, learning_rate=0.1)

        if i <= 1:
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())

        fetches = {
            'targets': y,
            'predictions': result.predictions,
            'cost': result.cost,
            'state': result.state,
            'train_op': result.train_op,
        }

        output = session.run(fetches)

        accuracy_hit = accuracy_miss = 0
        for target, prediction in zip(output['targets'], output['predictions']):
            target_string = ReberInputProvider.get_line_by_char_ids(target)
            prediction_string = ReberInputProvider.get_line_by_char_ids(prediction)

            checker_fsm.reset_to_post_start()
            for target_char, prediction_char in zip(target_string, prediction_string):
                if target_char == ReberInputProvider.get_eos():
                    if prediction_char == ReberInputProvider.get_eos():
                        accuracy_hit += 1
                    else:
                        accuracy_miss += 1
                    break
                else:
                    if checker_fsm.is_output_possible(prediction_char):
                        accuracy_hit += 1
                    else:
                        accuracy_miss += 1
                    checker_fsm.transition_by_output(target_char)

        accuracy = accuracy_hit / (accuracy_hit + accuracy_miss)
        print('accuracy: {:.2%}'.format(accuracy))

        cost = output['cost'] / session.run(tf.size(y))
        print('cost: {}'.format(cost))


class LSTMModel:

    class ProcessBatchResult:
        def __init__(self, predictions, cost, state, train_op):
            self.predictions = predictions
            self.cost = cost
            self.state = state
            self.train_op = train_op

    def __init__(self, hidden_size, output_size, is_training):
        self._hidden_size = hidden_size
        self._output_size = output_size
        self._cell = tf.contrib.rnn.LSTMCell(self._hidden_size)
        self._is_training = is_training

    def process_batch(self, x, y, learning_rate=None):
        if self._is_training:
            assert learning_rate is not None

        initial_state = self._cell.zero_state(x.shape[0], dtype=tf.float32)
        outputs, state = tf.nn.dynamic_rnn(
            self._cell,
            x,
            initial_state=initial_state,
            dtype=tf.float32,
        )
        output = tf.reshape(tf.concat(outputs, 1), [-1, self._hidden_size])

        with tf.variable_scope('LSTMModel', reuse=tf.AUTO_REUSE):
            softmax_w = tf.get_variable(
                'softmax_w',
                [self._hidden_size, self._output_size],
                dtype=tf.float32,
            )
            softmax_b = tf.get_variable(
                'softmax_b',
                [self._output_size],
                dtype=tf.float32,
            )

        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(logits, x.shape)
        predictions = tf.argmax(logits, 2)

        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            y,
            tf.ones([x.shape[0], x.shape[1]], dtype=tf.float32),
            average_across_timesteps=False,
            average_across_batch=True,
        )
        cost = tf.reduce_sum(loss)

        if self._is_training:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(cost)
        else:
            train_op = None

        result = self.ProcessBatchResult(
            predictions=predictions,
            cost=cost,
            state=state,
            train_op=train_op,
        )

        return result
