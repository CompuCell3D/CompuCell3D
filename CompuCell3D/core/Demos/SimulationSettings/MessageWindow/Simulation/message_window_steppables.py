from cc3d.core.PySteppables import *


class MessageWindowSteppable(SteppableBasePy):
    def __init__(self, frequency=100):
        SteppableBasePy.__init__(self, frequency)

        self.msg_win = None

    def start(self):
        self.msg_win = self.add_new_message_window(title='Messages')
        self.msg_win.print("INSIDE START FCN", color='red')

    def step(self, mcs):

        self.msg_win.print("step=", mcs, style=BOLD|ITALIC|UNDERLINE, color='blue')

        if not mcs % 100:
            self.msg_win.clear()
