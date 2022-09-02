#ifndef CC3DPYOBJECT_H
#define CC3DPYOBJECT_H

#include "CC3DPy.h"


class CC3DPyObject {

protected:

    PyObject *dict;

public:

    CC3DPyObject() : 
        dict{PyDict_New()}
    {}

    PyObject *getdict() {
        Py_INCREF(dict);
        return dict;
    }

};

#endif