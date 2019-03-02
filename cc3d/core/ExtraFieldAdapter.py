class ExtraFieldAdapter:
    def __init__(self,name='unknown', field_type=None):
        self.name = name
        self.field_type = field_type
        self.field_ref = None

    def clear(self):
        try:
            self.field_ref.clear()
        except AttributeError:
            pass

    def set_ref(self,field_ref):
        self.field_ref = field_ref

    def get_ref(self):
        return self.field_ref

    def __getitem__(self, item):
        return self.field_ref.__getitem__(item)

    def __setitem__(self, key, value):
        self.field_ref.__setitem__(key, value)