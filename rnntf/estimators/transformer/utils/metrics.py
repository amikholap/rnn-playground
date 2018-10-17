# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions for calculating loss, accuracy, and other model metrics.

Metrics:
 - Padded loss, accuracy, and negative log perplexity. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/metrics.py
 - BLEU approximation. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/bleu_hook.py
 - ROUGE score. Source:
     https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/rouge.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf


def _pad_tensors_to_same_length(x, y):
  """Pad x and y so that the results have the same length (second dimension)."""
  with tf.name_scope("pad_to_same_length"):
    x_length = tf.shape(x)[1]
    y_length = tf.shape(y)[1]

    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
    y = tf.pad(y, [[0, 0], [0, max_length - y_length]])
    return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
  """Calculate cross entropy loss while ignoring padding.

  Args:
    logits: Tensor of size [batch_size, length_logits, vocab_size]
    labels: Tensor of size [batch_size, length_labels]
    smoothing: Label smoothing constant, used to determine the on and off values
    vocab_size: int size of the vocabulary
  Returns:
    Returns the cross entropy loss and weight tensors: float32 tensors with
      shape [batch_size, max(length_logits, length_labels)]
  """
  with tf.name_scope("loss", [logits, labels]):
    logits, labels = _pad_tensors_to_same_length(logits, labels)

    # Calculate smoothing cross entropy
    with tf.name_scope("smoothing_cross_entropy", [logits, labels]):
      confidence = 1.0 - smoothing
      low_confidence = (1.0 - confidence) / tf.to_float(vocab_size - 1)
      soft_targets = tf.one_hot(
          tf.cast(labels, tf.int32),
          depth=vocab_size,
          on_value=confidence,
          off_value=low_confidence)
      xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=logits, labels=soft_targets)

      # Calculate the best (lowest) possible value of cross entropy, and
      # subtract from the cross entropy loss.
      normalizing_constant = -(
          confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
          low_confidence * tf.log(low_confidence + 1e-20))
      xentropy -= normalizing_constant

    weights = tf.to_float(tf.not_equal(labels, 0))
    return xentropy * weights, weights


def _convert_to_eval_metric(metric_fn):
  """Wrap a metric fn that returns scores and weights as an eval metric fn.

  The input metric_fn returns values for the current batch. The wrapper
  aggregates the return values collected over all of the batches evaluated.

  Args:
    metric_fn: function that returns scores and weights for the current batch's
      logits and predicted labels.

  Returns:
    function that aggregates the scores and weights from metric_fn.
  """
  def problem_metric_fn(*args):
    """Returns an aggregation of the metric_fn's returned values."""
    (scores, weights) = metric_fn(*args)

    # The tf.metrics.mean function assures correct aggregation.
    return tf.metrics.mean(scores, weights)
  return problem_metric_fn


def get_eval_metrics(logits, labels, params):
  """Return dictionary of model evaluation metrics."""
  tf.Print(logits, [logits], message='Logits: ')
  tf.Print(labels, [labels], message='Labels: ')
  metrics = {
      "accuracy": _convert_to_eval_metric(padded_accuracy)(logits, labels),
      "accuracy_top5": _convert_to_eval_metric(padded_accuracy_top5)(
          logits, labels),
      "accuracy_per_sequence": _convert_to_eval_metric(
          padded_sequence_accuracy)(logits, labels),
  }

  # Prefix each of the metric names with "metrics/". This allows the metric
  # graphs to display under the "metrics" category in TensorBoard.
  metrics = {"metrics/%s" % k: v for k, v in six.iteritems(metrics)}
  return metrics


def padded_accuracy(logits, labels):
  """Percentage of times that predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy", values=[logits, labels]):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    outputs = tf.to_int32(tf.argmax(logits, axis=-1))
    padded_labels = tf.to_int32(labels)
    return tf.to_float(tf.equal(outputs, padded_labels)), weights


def padded_accuracy_topk(logits, labels, k):
  """Percentage of times that top-k predictions matches labels on non-0s."""
  with tf.variable_scope("padded_accuracy_topk", values=[logits, labels]):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    effective_k = tf.minimum(k, tf.shape(logits)[-1])
    _, outputs = tf.nn.top_k(logits, k=effective_k)
    outputs = tf.to_int32(outputs)
    padded_labels = tf.to_int32(labels)
    padded_labels = tf.expand_dims(padded_labels, axis=-1)
    padded_labels += tf.zeros_like(outputs)  # Pad to same shape.
    same = tf.to_float(tf.equal(outputs, padded_labels))
    same_topk = tf.reduce_sum(same, axis=-1)
    return same_topk, weights


def padded_accuracy_top5(logits, labels):
  return padded_accuracy_topk(logits, labels, 5)


def padded_sequence_accuracy(logits, labels):
  """Percentage of times that predictions matches labels everywhere (non-0)."""
  with tf.variable_scope("padded_sequence_accuracy", values=[logits, labels]):
    logits, labels = _pad_tensors_to_same_length(logits, labels)
    weights = tf.to_float(tf.not_equal(labels, 0))
    outputs = tf.to_int32(tf.argmax(logits, axis=-1))
    padded_labels = tf.to_int32(labels)
    not_correct = tf.to_float(tf.not_equal(outputs, padded_labels)) * weights
    axis = list(range(1, len(outputs.get_shape())))
    correct_seq = 1.0 - tf.minimum(1.0, tf.reduce_sum(not_correct, axis=axis))
    return correct_seq, tf.constant(1.0)
