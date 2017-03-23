(PARAMETER, TYPE, VALUE, ACTION) = range(4)

(XML_ATTR, XML_CDATA, PYTHON_GLOBAL) = range(3)
# reverse lookup
TYPE_DICT = {XML_ATTR: 'XML_ATTR', XML_CDATA: 'XML_CDATA', PYTHON_GLOBAL: 'PYTHON_GLOBAL'}

TYPE_DICT_REVERSE = {}  # dictionry comprehension does not work for 2.6 and earlier... thank you Apple
for k, v in TYPE_DICT.items():
    TYPE_DICT_REVERSE[v] = k

(FLOAT, INT, STRING) = range(3)
# reverse lookup
VALUE_TYPE_DICT = {FLOAT: 'float', INT: 'int', STRING: 'string'}
VALUE_TYPE_DICT_REVERSE = {}  # dictionry comprehension does not work for 2.6 and earlier... thank you Apple
for k, v in VALUE_TYPE_DICT.items():
    VALUE_TYPE_DICT_REVERSE[v] = k

(SCAN_FINISHED_OR_DIRECTORY_ISSUE,) = range(2, 3)
