import collections
import random
import uuid


class FSM:

    def __init__(self):
        self._transitions = set()
        self._state_transitions = {}
        self._initial_state = None
        self._current_state = None

    def add_state(self):
        state = FSMState(id=uuid.uuid4())
        self._state_transitions[state] = []
        return state

    def add_transition(self, s1, s2, output):  # pylint: disable=invalid-name
        transition = FSMStateTransition(s1=s1, s2=s2, output=output)
        assert transition not in self._transitions
        self._transitions.add(transition)
        self._state_transitions[s1].append(transition)
        return transition

    def set_initial_state(self, state):
        self._initial_state = state

    def reset(self):
        self._current_state = self._initial_state

    def generate_sequence(self):
        self._check_initialized()

        if self._current_state is None:
            self._current_state = self._initial_state

        seq = []

        available_transitions = self._state_transitions[self._current_state]
        while available_transitions:
            transition = random.choice(available_transitions)
            seq.append(transition.output)
            self._current_state = transition.s2
            available_transitions = self._state_transitions[self._current_state]

        return seq

    def is_output_possible(self, output):
        self._check_initialized()

        for transition in self._state_transitions[self._current_state]:
            if transition.output == output:
                return True

        return False

    def transition_by_output(self, output):
        transition = None

        for trans in self._state_transitions[self._current_state]:
            if trans.output == output:
                transition = trans
                break

        if transition is None:
            raise RuntimeError('invalid transition: {}'.format(output))

        self._current_state = transition.s2

        return transition.output

    def _check_initialized(self):
        assert self._initial_state is not None, 'FSM is not initialized'


FSMState = collections.namedtuple('FSMState', ['id'])

FSMStateTransition = collections.namedtuple('FSMStateTransition', ['s1', 's2', 'output'])
