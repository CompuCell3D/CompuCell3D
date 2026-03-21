//
// Created by m on 9/13/25.
//

#ifndef COMPUCELL3D_CPPNUMPYUTILS_H
#define COMPUCELL3D_CPPNUMPYUTILS_H

#include <numpy/arrayobject.h>

// Primary template: left undefined on purpose
template <typename T>
struct NumpyTypeMap;

template <>
struct NumpyTypeMap<short> {
    static const int typenum = NPY_SHORT;
};

template <>
struct NumpyTypeMap<int> {
    static const int typenum = NPY_INT;
};

template <>
struct NumpyTypeMap<long> {
    static const int typenum = NPY_LONG;
};

template <>
struct NumpyTypeMap<long long> {
    static const int typenum = NPY_LONGLONG;
};

template <>
struct NumpyTypeMap<unsigned char> {
    static const int typenum = NPY_UBYTE;
};

template <>
struct NumpyTypeMap<unsigned short> {
    static const int typenum = NPY_USHORT;
};

template <>
struct NumpyTypeMap<unsigned int> {
    static const int typenum = NPY_UINT;
};

template <>
struct NumpyTypeMap<unsigned long> {
    static const int typenum = NPY_ULONG;
};

template <>
struct NumpyTypeMap<unsigned long long> {
    static const int typenum = NPY_ULONGLONG;
};

template <>
struct NumpyTypeMap<float> {
    static const int typenum = NPY_FLOAT;
};

template <>
struct NumpyTypeMap<double> {
    static const int typenum = NPY_DOUBLE;
};

template <>
struct NumpyTypeMap<long double> {
    static const int typenum = NPY_LONGDOUBLE;
};

template <>
struct NumpyTypeMap<bool> {
    static const int typenum = NPY_BOOL;
};

// ---- char / signed char ----
template <>
struct NumpyTypeMap<char> {
    // platform dependent signedness
    static const int typenum = NPY_BYTE;
};

template <>
struct NumpyTypeMap<signed char> {
    static const int typenum = NPY_BYTE;
};

#endif // COMPUCELL3D_CPPNUMPYUTILS_H
