import numpy as np
import tensorflow as tf

from .fsm import EmbeddedReberFSM, ReberFSM


class ReberInput:

    def __init__(self, *, batch_size):
        self._batch_size = batch_size

    @classmethod
    def get_feature_columns(cls):
        feature_columns = [
            tf.feature_column.indicator_column(
                tf.contrib.feature_column.sequence_categorical_column_with_identity(
                    'char',
                    num_buckets=cls.get_vocabulary_size(),
                ),
            ),
        ]
        return feature_columns

    @classmethod
    def get_vocabulary_size(cls):
        return cls.get_eos_id() - cls._get_char_id('A') + 1

    @classmethod
    def get_char_by_id(cls, char_id):
        return chr(ord('A') + char_id)

    @classmethod
    def get_line_by_char_ids(cls, char_ids):
        return ''.join([cls.get_char_by_id(char_id) for char_id in char_ids])

    @classmethod
    def get_eos(cls):
        return chr(ord('Z') + 1)

    @classmethod
    def get_eos_id(cls):
        return cls._get_char_id(cls.get_eos())

    @classmethod
    def _get_char_id(cls, char):
        return ord(char) - ord('A')

    def __iter__(self):
        raise NotImplementedError

    def input_fn(self, labels=True):
        x, y = next(iter(self))
        features = {
            'char': x,
        }

        if labels:
            slices = (features, y)
        else:
            slices = features

        dataset = tf.data.Dataset.from_tensor_slices(slices)
        dataset = dataset.batch(self._batch_size)

        return dataset

    def input_fn_no_labels(self):
        return self.input_fn(labels=False)

    def _make_x_y(self, seq, padding=None):
        x_chars = []
        for char in seq:
            char_id = self._get_char_id(char)
            x_chars.append(char_id)

        if padding and len(x_chars) < padding:
            x_chars += [self.get_eos_id()] * (padding - len(x_chars))

        y_chars = x_chars[1:]
        y_chars.append(self.get_eos_id())

        return x_chars, y_chars

    def _make_x_y_batch(self, seq_batch):
        max_length = max([len(seq) for seq in seq_batch])

        x_char_batch = []
        y_char_batch = []
        for seq in seq_batch:
            x_chars, y_chars = self._make_x_y(seq, padding=max_length)
            x_char_batch.append(x_chars)
            y_char_batch.append(y_chars)

        x = np.array(x_char_batch, dtype=np.int8)
        y = np.array(y_char_batch, dtype=np.int32)

        return x, y


class BaseFSMReberInput(ReberInput):

    def _get_fsm(self):
        raise NotImplementedError

    def __iter__(self):
        fsm = self._get_fsm()

        while True:
            seq_batch = []
            for _ in range(self._batch_size):
                seq = fsm.generate_sequence()
                fsm.reset()
                seq_batch.append(seq)

            x, y = self._make_x_y_batch(seq_batch)

            yield x, y


class FSMEmbeddedInput(BaseFSMReberInput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fsm = EmbeddedReberFSM()

    def _get_fsm(self):
        return self._fsm


class FSMReberInput(BaseFSMReberInput):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fsm = ReberFSM()

    def _get_fsm(self):
        return self._fsm


class FileReberInput(ReberInput):

    def __init__(self, file_, **kwargs):
        super().__init__(**kwargs)
        self._file = file_

    def __iter__(self):
        while True:
            seq_batch = []
            while True:
                seq = list(self._file.readline())
                assert seq.pop() == '\n'  # Remove newline.
                seq_batch.append(seq)
                if len(seq_batch) >= self._batch_size or not seq:
                    break

            if not seq_batch:
                break

            x, y = self._make_x_y_batch(seq_batch)

            yield x, y
