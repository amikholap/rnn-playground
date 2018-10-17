import random

import tensorflow as tf
from tensorflow.python import debug as tf_debug  # pylint: disable=no-name-in-module

from .fsm import EmbeddedReberFSM, ReberFSM
from .inputs import (
    FileReberInput, FSMEmbeddedInput, FSMReberInput, ReberInput,
)
from .metrics import ReberAccuracy
from .. import estimators


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

    common_params = {
        'model_dir': args.model_dir,
    }
    if args.estimator == 'lstm':
        model = estimators.lstm.LSTMEstimator(
            params={
                'hidden_units': args.hidden_units,
                'output_units': ReberInput.get_vocabulary_size(),
                'feature_columns': input_.get_feature_columns(),
                'learning_rate': args.learning_rate,
            }
            **common_params,
        )
    elif args.estimator == 'transformer':
        model = estimators.transformer.TransformerEstimator(
            params={
                'allow_ffn_pad': False,
                'alpha': 0.6,
                'attention_dropout': 0.0,
                'beam_size': 4,
                'extra_decode_length': 25,
                'filter_size': 32,
                'hidden_size': 32,
                'initializer_gain': 1.0,
                'label_smoothing': 0.0,
                'layer_postprocess_dropout': 0,
                'learning_rate': args.learning_rate,
                'learning_rate_warmup_steps': 10,
                'num_heads': 1,
                'num_hidden_layers': 1,
                'optimizer_adam_beta1': 0.9,
                'optimizer_adam_beta2': 0.99,
                'optimizer_adam_epsilon': 1e-09,
                'relu_dropout': 0.0,
                'tpu': False,
                'use_tpu': False,
                'vocab_size': ReberInput.get_vocabulary_size(),
            },
            **common_params,
        )
    else:
        raise RuntimeError('unreachable')

    test_input_gen = iter(input_)
    test_x, test_y = next(test_input_gen)
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={
            'char': test_x,
        },
        y=test_y,
        batch_size=len(test_x),
        shuffle=False,
    )

    train_hooks = None
    if args.debug:
        train_hooks = [tf_debug.LocalCLIDebugHook()]

    while True:
        model.train(
            input_fn=input_.input_fn,
            hooks=train_hooks,
        )
        eval_result = model.evaluate(
            input_fn=input_.input_fn,
        )

        predict_result = list(model.predict(
            input_fn=test_input_fn,
        ))
        predictions = [x['outputs'] for x in predict_result]
        accuracy = accuracy_evaluator(labels=test_y, predictions=predictions)
        print(
            '--- step: {}\nsoft accuracy: {:.2%}\nloss: {:.3}'
            .format(eval_result['global_step'], accuracy, eval_result['loss'])
        )

        idx = random.randrange(len(test_x))
        print('input: ', ' '.join([str(x).rjust(2) for x in test_x[idx]]))
        print('target:', ' '.join([str(x).rjust(2) for x in test_y[idx]]))
        print('output:', ' '.join([str(x).rjust(2) for x in predictions[idx]]))
