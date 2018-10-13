import tensorflow as tf

from .fsm import EmbeddedReberFSM, ReberFSM
from .inputs import (
    FileReberInput, FSMEmbeddedInput, FSMReberInput, ReberInput,
)
from .metrics import ReberAccuracy
from ..estimators.lstm import LSTMEstimator


def run(args):
    if args.dataset_path:
        input_ = FileReberInput(args.dataset_path, batch_size=args.batch_size)
    elif args.embedded:
        input_ = FSMEmbeddedInput(batch_size=args.batch_size)
    else:
        input_ = FSMReberInput(batch_size=args.batch_size)

    if args.embedded:
        fsm_type = EmbeddedReberFSM
    else:
        fsm_type = ReberFSM
    accuracy_evaluator = ReberAccuracy(fsm=fsm_type())

    if args.estimator == 'lstm':
        model = LSTMEstimator(
            hidden_units=args.hidden_units,
            output_units=ReberInput.get_vocabulary_size(),
            feature_columns=input_.get_feature_columns(),
        )
    else:
        raise RuntimeError('unreachable')

    test_input_gen = iter(input_)

    while True:
        model.train(
            input_fn=input_.input_fn,
        )
        eval_result = model.evaluate(
            input_fn=input_.input_fn,
        )

        test_x, test_y = next(test_input_gen)
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={
                'char': test_x,
            },
            batch_size=test_x.shape[0],
            shuffle=False,
        )
        predict_result = list(model.predict(
            input_fn=test_input_fn,
        ))
        predictions = [x['predictions'] for x in predict_result]
        accuracy = accuracy_evaluator(labels=test_y, predictions=predictions)
        print(
            '--- step: {}\nsoft accuracy: {:.2%}\nloss: {:.3}'
            .format(eval_result['global_step'], accuracy, eval_result['loss'])
        )
