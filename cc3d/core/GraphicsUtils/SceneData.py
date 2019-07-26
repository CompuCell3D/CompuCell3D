from collections import OrderedDict
class ActorProperties(object):
    def __init__(self):
        self.name = None
        self.color = None

class SceneData(object):
    def __init__(self):
        self.actor_data_dict = OrderedDict()
