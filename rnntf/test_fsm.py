# pylint: disable=invalid-name
import unittest

from .fsm import FSM


class FSMTestCase(unittest.TestCase):

    def test_not_initialized(self):
        fsm = FSM()
        with self.assertRaises(AssertionError):
            fsm.generate_sequence()

    def test_transition(self):
        fsm = FSM()
        s1 = fsm.add_state()
        s2 = fsm.add_state()
        fsm.set_initial_state(s1)
        fsm.add_transition(s1, s2, '.')
        self.assertEqual(fsm.generate_sequence(), ['.'])

    def test_simple_loop(self):
        fsm = FSM()

        s1 = fsm.add_state()
        s2 = fsm.add_state()
        fsm.set_initial_state(s1)

        fsm.add_transition(s1, s1, 'z')
        fsm.add_transition(s1, s2, '.')

        seq = fsm.generate_sequence()
        self.assertEqual(seq[-1], '.')
        self.assertEqual(seq[:-1], ['z'] * (len(seq) - 1))
