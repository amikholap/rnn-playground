import tensorflow as tf

from .model import transformer
from .utils import metrics


class TransformerEstimator(tf.estimator.Estimator):

    def __init__(self, params, **kwargs):
        super().__init__(
            model_fn=model_fn,
            params=params,
            **kwargs,
        )


def model_fn(features, labels, mode, params):
    """Defines how to train, evaluate and predict from the transformer model."""

    features = features['char']

    with tf.variable_scope("model"):
        inputs, targets = features, labels

        # Create model and get output logits.
        model = transformer.Transformer(params, mode == tf.estimator.ModeKeys.TRAIN)

        logits = model(inputs, targets)

        # When in prediction mode, the labels/targets is None. The model output
        # is the prediction
        if mode == tf.estimator.ModeKeys.PREDICT:
            if params["use_tpu"]:
                raise NotImplementedError("Prediction is not yet supported on TPUs.")
            return tf.estimator.EstimatorSpec(
                tf.estimator.ModeKeys.PREDICT,
                predictions=logits,
                export_outputs={
                    "translate": tf.estimator.export.PredictOutput(logits)
                })

        # Explicitly set the shape of the logits for XLA (TPU). This is needed
        # because the logits are passed back to the host VM CPU for metric
        # evaluation, and the shape of [?, ?, vocab_size] is too vague. However
        # it is known from Transformer that the first two dimensions of logits
        # are the dimensions of targets. Note that the ambiguous shape of logits is
        # not a problem when computing xentropy, because padded_cross_entropy_loss
        # resolves the shape on the TPU.
        logits.set_shape(targets.shape.as_list() + logits.shape.as_list()[2:])

        # Calculate model loss.
        # xentropy contains the cross entropy loss of every nonpadding token in the
        # targets.
        xentropy, weights = metrics.padded_cross_entropy_loss(
            logits, targets, params["label_smoothing"], params["vocab_size"])
        loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

        # Save loss as named tensor that will be logged with the logging hook.
        tf.identity(loss, "cross_entropy")

        if mode == tf.estimator.ModeKeys.EVAL:  # pylint: disable=no-else-return
            return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss, predictions={"predictions": logits},
                eval_metric_ops=metrics.get_eval_metrics(logits, labels, params))
        else:
            train_op, metric_dict = get_train_op_and_metrics(loss, params)

            # Epochs can be quite long. This gives some intermediate information
            # in TensorBoard.
            metric_dict["minibatch_loss"] = loss
            record_scalars(metric_dict)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)


def get_train_op_and_metrics(loss, params):
    """Generate training op and metrics to save in TensorBoard."""
    with tf.variable_scope("get_train_op"):
        learning_rate = get_learning_rate(
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            learning_rate_warmup_steps=params["learning_rate_warmup_steps"])
        tf.contrib.summary.scalar(name='learning_rate', tensor=learning_rate)

        # Create optimizer. Use LazyAdamOptimizer from TF contrib, which is faster
        # than the TF core Adam optimizer.
        optimizer = tf.contrib.opt.LazyAdamOptimizer(
            learning_rate,
            beta1=params["optimizer_adam_beta1"],
            beta2=params["optimizer_adam_beta2"],
            epsilon=params["optimizer_adam_epsilon"])

        # Calculate and apply gradients using LazyAdamOptimizer.
        global_step = tf.train.get_global_step()
        tvars = tf.trainable_variables()
        gradients = optimizer.compute_gradients(
            loss, tvars, colocate_gradients_with_ops=True)
        minimize_op = optimizer.apply_gradients(
            gradients, global_step=global_step, name="train")
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)

        train_metrics = {"learning_rate": learning_rate}

        if not params["use_tpu"]:
            # gradient norm is not included as a summary when running on TPU, as
            # it can cause instability between the TPU and the host controller.
            gradient_norm = tf.global_norm(list(zip(*gradients))[0])
            train_metrics["global_norm/gradient_norm"] = gradient_norm

        return train_op, train_metrics


def record_scalars(metric_dict):
    for key, value in metric_dict.items():
        tf.contrib.summary.scalar(name=key, tensor=value)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps):
    """Calculate learning rate with linear warmup and rsqrt decay."""
    with tf.name_scope("learning_rate"):
        warmup_steps = tf.to_float(learning_rate_warmup_steps)
        step = tf.to_float(tf.train.get_or_create_global_step())

        learning_rate *= (hidden_size ** -0.5)
        # Apply linear warmup
        learning_rate *= tf.minimum(1.0, step / warmup_steps)
        # Apply rsqrt decay
        learning_rate *= tf.rsqrt(tf.maximum(step, warmup_steps))

        # Create a named tensor that will be logged using the logging hook.
        # The full name includes variable and names scope. In this case, the name
        # is model/get_train_op/learning_rate/learning_rate
        tf.identity(learning_rate, "learning_rate")

        return learning_rate
