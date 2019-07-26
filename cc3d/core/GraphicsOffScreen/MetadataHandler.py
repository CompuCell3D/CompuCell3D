class MetadataHandler(object):
    def __init__(self, mdata):
        self._mdata = mdata
        self._defaults = {
            'color': (255,255,255,255),
            'bool': False,
        }

    def get(self, key, default=None, data_type=None):
        """
        Fetches metadata from the metadata dict - convenience functionn that simplifies client-side code simpler
        :param key: {str}
        :param default: {object or NOne}
        :param data_type: {str} data type -
        :return: {object or raises exception}
        """
        try:
            return self._mdata[key]
        except KeyError:
            if default is not None:
                print(('WARNING: Could not find key {} in metadata . Returning provided default of {}'.format(key, default)))
                return default
            elif data_type is not None:
                try:
                    default_type_val = self._defaults[data_type]
                    print(('WARNING: Could not find key {} in metadata . Returning type default of {}'.format(key,
                                                                                                    default_type_val)))
                except KeyError:
                    raise KeyError('Unable to fetch {} metadata'.format(key))
            else:
                raise KeyError('Unable to fetch {} metadata'.format(key))
