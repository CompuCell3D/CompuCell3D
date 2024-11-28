//==========================================================
// Modification of the 
// ndarray.h: C++ interface for easy access to numpy arrays 
//
// J. De Ridder
//==========================================================

#ifndef NDARRAY_ADAPTER_H
#define NDARRAY_ADAPTER_H

#include <vector>
// The C-struct to retrieve the ctypes structure.
// Note: Order of struct members must be the same as in Python.
//       Member names are not recognized!


// template <typename T>
// struct numpyArray
// {
//     T *data;
//     long *shape;
//     long *strides;
// };



// Traits need to used because the return type of the []-operator can be
// a subarray or an element, depending whether all the axes are exhausted.

template<typename datatype, int ndim> class NdarrayAdapter;        // forward declaration


template<typename datatype, int ndim>
struct getItemTraits
{
    typedef NdarrayAdapter<datatype, ndim-1> returnType;
};


template<typename datatype>
struct getItemTraits<datatype, 1>
{
    typedef datatype& returnType;
};



// NdarrayAdapter definition

template<typename datatype, int ndim>
class NdarrayAdapter
{
    private:
        
        long *shape;
        long *strides;
        std::vector<long> shapeVec;
        std::vector<long> stridesVec;
        
    public:
        datatype *data;
        
        NdarrayAdapter(datatype *data, long *shape, long *strides);
//         NdarrayAdapter(datatype *data, int _ndim);
        NdarrayAdapter(datatype *data=0);
        void setStrides(const std::vector<long> &_strides);
        void setShape(const std::vector<long> &_shape);
        void setData(datatype *_data);        
        NdarrayAdapter(const NdarrayAdapter<datatype, ndim>& array);
//         NdarrayAdapter(const numpyArray<datatype>& array);
        long getShape(const int axis);
        void clear();
        typename getItemTraits<datatype, ndim>::returnType operator[](unsigned long i);       
};


// NdarrayAdapter constructor
template<typename datatype, int ndim>
NdarrayAdapter<datatype, ndim>::NdarrayAdapter(datatype *data, long *shape, long *strides)
{
    this->data = data;
    this->shape = shape;
    this->strides = strides;
}


// NdarrayAdapter constructor
template<typename datatype, int ndim>
// NdarrayAdapter<datatype, ndim>::NdarrayAdapter(datatype *data, int _ndim){
    NdarrayAdapter<datatype, ndim>::NdarrayAdapter(datatype *data){    
    this->data = data;
    this->shapeVec = std::vector<long>(ndim,0);
    this->stridesVec = std::vector<long>(ndim,1);
    strides=&stridesVec[0];
    shape=&shapeVec[0];
}

// NdarrayAdapter setStrides
template<typename datatype, int ndim>
void NdarrayAdapter<datatype, ndim>::setStrides(const std::vector<long> &_strides){
       stridesVec=_strides;
       strides=&stridesVec[0];
    
}
// NdarrayAdapter setStrides
template<typename datatype, int ndim>
void NdarrayAdapter<datatype, ndim>::setShape(const std::vector<long> &_shape){
    shapeVec=_shape;
    shape=&shapeVec[0];
}

// NdarrayAdapter setData
template<typename datatype, int ndim>
void NdarrayAdapter<datatype, ndim>::setData(datatype *_data){
    data=_data;
}

// NdarrayAdapter copy constructor

template<typename datatype, int ndim>
NdarrayAdapter<datatype, ndim>::NdarrayAdapter(const NdarrayAdapter<datatype, ndim>& array)
{
    this->data = array.data;
    this->shape = array.shape;
    this->strides = array.strides;
    this->shapeVec = array.shapeVec;
    this->stridesVec = array.stridesVec;
    
}


// // NdarrayAdapter constructor from ctypes structure

// template<typename datatype, int ndim>
// NdarrayAdapter<datatype, ndim>::NdarrayAdapter(const numpyArray<datatype>& array)
// {
//     this->data = array.data;
//     this->shape = array.shape;
//     this->strides = array.strides;
// }

// NdarrayAdapter method to clear container
template<typename datatype, int ndim>
void NdarrayAdapter<datatype, ndim>::clear()
{
    long arraySize=1;
    for (int i= 0 ; i < ndim ; ++i){
        arraySize*=shape[i];
    }
    
    for (int i  ; i < arraySize ; ++i){
        data[i]=datatype();
    }
}


// NdarrayAdapter method to get length of given axis

template<typename datatype, int ndim>
long NdarrayAdapter<datatype, ndim>::getShape(const int axis)
{
    return this->shape[axis];
}



// NdarrayAdapter overloaded []-operator.
// The [i][j][k] selection is recursively replaced by i*strides[0]+j*strides[1]+k*strides[2]
// at compile time, using template meta-programming. If the axes are not exhausted, return
// a subarray, else return an element.

template<typename datatype, int ndim>
typename getItemTraits<datatype, ndim>::returnType
NdarrayAdapter<datatype, ndim>::operator[](unsigned long i)
{
    return NdarrayAdapter<datatype, ndim-1>(&this->data[i*this->strides[0]], &this->shape[1], &this->strides[1]);
}



// Template partial specialisation of NdarrayAdapter.
// For 1D NdarrayAdapters, the [] operator should return an element, not a subarray, so it needs
// to be special-cased. In principle only the operator[] method should be specialised, but
// for some reason my gcc version seems to require that then the entire class with all its 
// methods are specialised.

template<typename datatype>
class NdarrayAdapter<datatype, 1>
{
    private:
        
        long *shape;
        long *strides;
        std::vector<long> shapeVec;
        std::vector<long> stridesVec;
        
    public:
        datatype *data;
        NdarrayAdapter(datatype *data, long *shape, long *strides);
//         NdarrayAdapter(datatype *data, int _ndim);
        NdarrayAdapter(datatype *data=0);
        void setStrides(const std::vector<long> &_strides);
        void setShape(const std::vector<long> &_shape);
        void setData(datatype *_data);
//         NdarrayAdapter(datatype *data, long *shape, long *strides);
        NdarrayAdapter(const NdarrayAdapter<datatype, 1>& array);
//         NdarrayAdapter(const numpyArray<datatype>& array);
        void clear();
        long getShape(const int axis);
        typename getItemTraits<datatype, 1>::returnType operator[](unsigned long i);       
};

// NdarrayAdapter partial specialised constructor
template<typename datatype>
// NdarrayAdapter<datatype, 1>::NdarrayAdapter(datatype *data, int _ndim){
NdarrayAdapter<datatype, 1>::NdarrayAdapter(datatype *data){    
    this->data = data;
    this->shapeVec = std::vector<long>(1,0);
    this->stridesVec = std::vector<long>(1,1);
    strides=&stridesVec[0];
    shape=&shapeVec[0];
}

// NdarrayAdapter setStrides partial specialised
template<typename datatype>
void NdarrayAdapter<datatype, 1>::setStrides(const std::vector<long> &_strides){
    stridesVec=_strides;   
    strides=&stridesVec[0];
       
    
}
// NdarrayAdapter setStrides partial specialised
template<typename datatype>
void NdarrayAdapter<datatype, 1>::setShape(const std::vector<long> &_shape){
    shapeVec=_shape;
    shape=&shapeVec[0];
}

// NdarrayAdapter setDatapartial specialised
template<typename datatype>
void NdarrayAdapter<datatype, 1>::setData(datatype *_data){
    data=_data;
}


// NdarrayAdapter partial specialised constructor
template<typename datatype>
NdarrayAdapter<datatype, 1>::NdarrayAdapter(datatype *data, long *shape, long *strides)
{
    this->data = data;
    this->shape = shape;
    this->strides = strides;
}



// NdarrayAdapter partially specialised copy constructor
template<typename datatype>
NdarrayAdapter<datatype, 1>::NdarrayAdapter(const NdarrayAdapter<datatype, 1>& array)
{
    this->data = array.data;
    this->shape = array.shape;
    this->strides = array.strides;

    this->shapeVec = array.shapeVec;
    this->stridesVec = array.stridesVec;
    
}



// // NdarrayAdapter partially specialised constructor from ctypes structure

// template<typename datatype>
// NdarrayAdapter<datatype, 1>::NdarrayAdapter(const numpyArray<datatype>& array)
// {
//     this->data = array.data;
//     this->shape = array.shape;
//     this->strides = array.strides;
// }



// NdarrayAdapter method to get length of given axis

template<typename datatype>
long NdarrayAdapter<datatype, 1>::getShape(const int axis)
{
    return this->shape[axis];
}

// NdarrayAdapter method to clear container partial specialised
template<typename datatype>
void NdarrayAdapter<datatype, 1>::clear()
{
    for (int i  ; i < shape[0]; ++i){
        data[i]=datatype();
    }
}


// Partial specialised [] operator: for 1D arrays, return an element rather than a subarray 

template<typename datatype>
typename getItemTraits<datatype, 1>::returnType
NdarrayAdapter<datatype, 1>::operator[](unsigned long i)
{
    return this->data[i*this->strides[0]];
}



#endif