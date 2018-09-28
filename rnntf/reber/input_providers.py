import tensorflow as tf

from .fsm import EmbeddedReberFSM, ReberFSM


class ReberInputProvider:

    def __init__(self, *, batch_size):
        self._batch_size = batch_size

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

    def _make_x_y(self, seq_batch):
        max_length = max([len(seq) for seq in seq_batch])

        x_char_batch = []
        y_char_batch = []

        for seq in seq_batch:
            x_chars = []
            for char in seq:
                char_id = self._get_char_id(char)
                x_chars.append(char_id)
            if len(x_chars) < max_length:
                x_chars += [self.get_eos_id()] * (max_length - len(x_chars))

            y_chars = x_chars[1:]
            y_chars.append(self.get_eos_id())

            x_char_batch.append(x_chars)
            y_char_batch.append(y_chars)

        x = tf.reshape(
            tf.one_hot(x_char_batch, depth=self.get_vocabulary_size()),
            [len(x_char_batch), max_length, self.get_vocabulary_size()],
        )
        y = tf.reshape(
            tf.convert_to_tensor(y_char_batch, dtype=tf.int32),
            [len(y_char_batch), max_length],
        )

        return x, y


class BaseFSMReberInputProvider(ReberInputProvider):

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

            x, y = self._make_x_y(seq_batch)

            yield x, y


class FSMEmbeddedInputProvider(BaseFSMReberInputProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fsm = EmbeddedReberFSM()

    def _get_fsm(self):
        return self._fsm


class FSMReberInputProvider(BaseFSMReberInputProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fsm = ReberFSM()

    def _get_fsm(self):
        return self._fsm


class FileReberInputProvider(ReberInputProvider):

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

            x, y = self._make_x_y(seq_batch)

            yield x, y
