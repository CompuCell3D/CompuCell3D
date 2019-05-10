from threading import Lock
from copy import deepcopy


class SteeringParam(object):
    def __init__(self, name, val, min_val=None, max_val=None, decimal_precision=3, enum=None,
                 widget_name=None):
        self._name = name
        self._val = None
        self._type = None

        # might be used later for selecting which parameters user have changed
        self._dirty_flag = False

        self.lock = Lock()

        if val is not None:
            self.val = val

        self._min = min_val
        self._min = min_val
        self._max = max_val
        self._decimal_precision = decimal_precision
        self.enum_allowed_widgets = ['combobox', 'pull-down']
        self._allowed_widget_names = ['lineedit', 'slider', 'combobox', 'pull-down']
        if widget_name in self.enum_allowed_widgets:
            assert isinstance(enum, list) or (enum is None), 'enum argument must be a list of None'
            if enum is None:
                enum = []
            # ensure all types in the enum list are the same
            test_list = [self.val] + enum

            type_set = set((map(lambda x: type(x), test_list)))
            assert len(type_set) == 1, 'enum list elements (together with initial value) must me of the same type. ' \
                                       'Instead I got the following types: {}'.format(
                ','.join(map(lambda x: str(x), type_set)))

            self._enum = enum
            if val is not None:
                try:
                    list(map(lambda x: str(x), self.enum)).index(str(self.val))
                except ValueError:
                    self._enum = [str(self.val)] + self._enum  # prepending current value
        else:
            self._enum = None

        if widget_name is None:
            self._widget_name = 'lineedit'
        else:
            assert isinstance(widget_name, str), 'widget_name has to be a Python string or None object'
            assert widget_name.lower() in self._allowed_widget_names, \
                '{} is not supported. We support the following  widgets {}'.format(widget_name,
                                                                                   ','.join(self._allowed_widget_names))
            self._widget_name = str(widget_name.lower())

    @property
    def dirty_flag(self):
        return self._dirty_flag

    @dirty_flag.setter
    def dirty_flag(self, flag):
        self._dirty_flag = flag

    @property
    def val(self):

        with self.lock:
            tmp_val = deepcopy(self._val)
        # self.lock.release()
        return tmp_val

    @val.setter
    def val(self, val):

        with self.lock:
            self._val = val
            self._type = type(self._val)
            # print 'val.type=', self._type

        # self.lock.release()

    @property
    def enum(self):
        return self._enum

    @property
    def name(self):
        return self._name

    @property
    def widget_name(self):
        return self._widget_name

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def decimal_precision(self):
        return self._decimal_precision

    @decimal_precision.setter
    def decimal_precision(self, decimal_precision):
        self._decimal_precision = decimal_precision

    @property
    def item_type(self):
        return self._type

    def __str__(self):
        s = ''
        s += ' name: {}'.format(self.name)
        s += ' val: {}'.format(self.val)
        s += ' widget: {}'.format(self.widget_name)
        return s

    def __repr__(self):
        return self.__str__()
