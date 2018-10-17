from .inputs import ReberInput


class ReberAccuracy:

    def __init__(self, fsm):
        self._fsm = fsm

    def __call__(self, labels, predictions):
        accuracy_hit = accuracy_miss = 0
        for target, prediction in zip(labels, predictions):
            target_string = ReberInput.get_line_by_char_ids(target)
            prediction_string = ReberInput.get_line_by_char_ids(prediction)

            self._fsm.reset_to_post_start()
            for target_char, prediction_char in zip(target_string, prediction_string):
                if target_char == ReberInput.get_eos():
                    if prediction_char == ReberInput.get_eos():
                        accuracy_hit += 1
                    else:
                        accuracy_miss += 1
                    break
                else:
                    if self._fsm.is_output_possible(prediction_char):
                        accuracy_hit += 1
                    else:
                        accuracy_miss += 1
                    self._fsm.transition_by_output(target_char)

                # Don't count trailing EOS symbols.
                if target_char == ReberInput.get_eos_id():
                    break

        value = accuracy_hit / (accuracy_hit + accuracy_miss)

        return value
