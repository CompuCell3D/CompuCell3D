class ExtraFieldAdapter:
    def __init__(self,name='unknown', field_type=None):
        self.name = name
        self.field_type = field_type
        self.__field_ref = None


    def set_ref(self,field_ref):
        self.__field_ref = field_ref

    def get_ref(self):
        return self.__field_ref

    def __getitem__(self, item):
        return self.__field_ref.__getitem(item)

    def __setitem__(self, key, value):
        self.__field_ref.__setitem__(key, value)