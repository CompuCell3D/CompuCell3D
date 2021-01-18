"""
Base class for Twedit plugin
"""


class TweditPluginBase:
    def __init__(self):
        pass

    def activate(self):
        pass

    def post_activate(self, **kwds):
        pass

    def deactivate(self):
        pass

    def __initMenus(self):
        pass

    def __initActions(self):
        pass