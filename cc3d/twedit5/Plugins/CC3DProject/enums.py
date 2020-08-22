(PARAMETER,TYPE,VALUE,ACTION)=list(range(4))



(XML_ATTR,XML_CDATA,PYTHON_GLOBAL)=list(range(3))

#reverse lookup

TYPE_DICT={XML_ATTR:'XML_ATTR',XML_CDATA:'XML_CDATA',PYTHON_GLOBAL:'PYTHON_GLOBAL'}



TYPE_DICT_REVERSE={v:k for k, v in list(TYPE_DICT.items())}



(FLOAT,INT,CUSTOM)=list(range(3))

#reverse lookup

VALUE_TYPE_DICT={FLOAT:'float',INT:'int',CUSTOM:'custom'}

VALUE_TYPE_DICT_REVERSE={v:k for k, v in list(VALUE_TYPE_DICT.items())}

