from rnntf.fsm import FSM


class BaseReberFSM(FSM):

    def reset_to_post_start(self):
        self.reset()
        self.transition_by_output('B')


class ReberFSM(BaseReberFSM):

    def __init__(self, *args, **kwargs):
        # pylint: disable=invalid-name
        super().__init__(*args, **kwargs)

        s_start = self.add_state()
        s1 = self.add_state()
        s2 = self.add_state()
        s3 = self.add_state()
        s4 = self.add_state()
        s5 = self.add_state()
        s6 = self.add_state()
        s_end = self.add_state()

        self.add_transition(s_start, s1, 'B')
        self.add_transition(s1, s2, 'T')
        self.add_transition(s1, s3, 'P')
        self.add_transition(s2, s2, 'S')
        self.add_transition(s2, s4, 'X')
        self.add_transition(s3, s3, 'T')
        self.add_transition(s3, s5, 'V')
        self.add_transition(s4, s3, 'X')
        self.add_transition(s4, s6, 'S')
        self.add_transition(s5, s4, 'P')
        self.add_transition(s5, s6, 'V')
        self.add_transition(s6, s_end, 'E')

        self.set_initial_state(s_start)


class EmbeddedReberFSM(BaseReberFSM):

    def __init__(self, *args, **kwargs):
        # pylint: disable=invalid-name
        super().__init__(*args, **kwargs)

        s_pre_start = self.add_state()
        s_post_start = self.add_state()

        s1_start = self.add_state()
        s11 = self.add_state()
        s12 = self.add_state()
        s13 = self.add_state()
        s14 = self.add_state()
        s15 = self.add_state()
        s16 = self.add_state()
        s1_end = self.add_state()

        s2_start = self.add_state()
        s21 = self.add_state()
        s22 = self.add_state()
        s23 = self.add_state()
        s24 = self.add_state()
        s25 = self.add_state()
        s26 = self.add_state()
        s2_end = self.add_state()

        s_pre_end = self.add_state()
        s_post_end = self.add_state()

        self.add_transition(s_pre_start, s_post_start, 'B')

        self.add_transition(s_post_start, s1_start, 'T')
        self.add_transition(s1_start, s11, 'B')
        self.add_transition(s11, s12, 'T')
        self.add_transition(s11, s13, 'P')
        self.add_transition(s12, s12, 'S')
        self.add_transition(s12, s14, 'X')
        self.add_transition(s13, s13, 'T')
        self.add_transition(s13, s15, 'V')
        self.add_transition(s14, s13, 'X')
        self.add_transition(s14, s16, 'S')
        self.add_transition(s15, s14, 'P')
        self.add_transition(s15, s16, 'V')
        self.add_transition(s16, s1_end, 'E')
        self.add_transition(s1_end, s_pre_end, 'T')

        self.add_transition(s_post_start, s2_start, 'P')
        self.add_transition(s21, s22, 'T')
        self.add_transition(s21, s23, 'P')
        self.add_transition(s22, s22, 'S')
        self.add_transition(s22, s24, 'X')
        self.add_transition(s23, s23, 'T')
        self.add_transition(s23, s25, 'V')
        self.add_transition(s24, s23, 'X')
        self.add_transition(s24, s26, 'S')
        self.add_transition(s25, s24, 'P')
        self.add_transition(s25, s26, 'V')
        self.add_transition(s26, s2_end, 'E')
        self.add_transition(s2_end, s_pre_end, 'P')

        self.add_transition(s_pre_end, s_post_end, 'E')

        self.set_initial_state(s_pre_start)
