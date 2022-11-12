#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <vector>
#include <set>
#include "Dim3D.h"
#include "Field3DImpl.h"
#include <iostream>


namespace CompuCell3D {

    template<typename T>
    class Array3D {
    public:
        typedef std::vector <std::vector<std::vector < T>> >
        ContainerType;

        void allocateArray(const Dim3D &_dim, T &val = T());

//       operator Type&();
        ContainerType &getContainer() { return array; }

    private:
        std::vector <std::vector<std::vector < T>> >
        array;

    };

    template<typename T>
    void Array3D<T>::allocateArray(const Dim3D &_dim, T &val) {
        using namespace std;
        array.assign(_dim.x, vector < vector < T > > (_dim.y, vector<T>(_dim.z, val)));
    }


//Adapter is necessary to keep current API of the Field3D . These fields are registered in Simulator and
    template<typename T>
    class Array3DField3DAdapter : public Field3DImpl<T> {
    public:
        Array3DField3DAdapter() : Field3DImpl<T>(Dim3D(1, 1, 1), T()), array3DPtr(0), containerPtr(0) {};

        virtual ~Array3DField3DAdapter() {
            if (array3DPtr)
                delete array3DPtr;
            array3DPtr = 0;
        }

        Array3D<T> *getArray3DPtr() { return array3DPtr; }

        typename Array3D<T>::ContainerType &getContainer() { return array3DPtr->getContainer(); }

        virtual void setDim(const Dim3D theDim) {
            if (!array3DPtr) {
                array3DPtr = new Array3D<T>();
                T t;
                t = T();

                array3DPtr->allocateArray(theDim, t);
                containerPtr = &array3DPtr->getContainer();
                Field3DImpl<T>::dim = theDim;
            } else {
                delete array3DPtr;
                array3DPtr = new Array3D<T>();
                T t;
                t = T();
                array3DPtr->allocateArray(theDim, t);
                containerPtr = &array3DPtr->getContainer();
                Field3DImpl<T>::dim = theDim;
            }
        }

        virtual Dim3D getDim() const { return Field3DImpl<T>::dim; };

        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < Field3DImpl<T>::dim.x &&
                    0 <= pt.y && pt.y < Field3DImpl<T>::dim.y &&
                    0 <= pt.z && pt.z < Field3DImpl<T>::dim.z);
        }

        virtual void set(const Point3D &pt, const T value) {
            if (array3DPtr) {
                (*containerPtr)[pt.x][pt.y][pt.z] = value;
            }
        }

        virtual T get(const Point3D &pt) const {
            return (*containerPtr)[pt.x][pt.y][pt.z];

        };

        virtual T getByIndex(long _offset) const {
            return T();
        }

        virtual void setByIndex(long _offset, const T _value) {

        }

    protected:
        Array3D<T> *array3DPtr;
        //Dim3D dim; //defined already in Field3DImpl<>
        typename Array3D<T>::ContainerType *containerPtr;
    };


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Adapter is necessary to keep current API of the Field3D . 3D array is allocated as a single chunk of memory - therefore trhe name linear. this particular array acts as a proxy for Fortran Arrays
//Fortran style array  shifts all the elements by (1,1,1) vector and 
//has dimensions (m+1)x(n+1)x(q+1)  
//last dimension of the array is iterated in the innermost loop 
//NOTICE: to get precise solutions from the solver we need to use double precision calculations (floats are imprecise)
//however since player is assuming float numbers as a return values of the Field3DImpl<T> API we need to 
//make base class  Field3DImpl<float>

    class Array3DLinearFortranField3DAdapter : public Field3DImpl<float> {
    public:
        //typedef float precision_t;

        Array3DLinearFortranField3DAdapter() : Field3DImpl<float>(Dim3D(1, 1, 1), float()) {};

        Array3DLinearFortranField3DAdapter(Dim3D &_dim, float &_initVal) : Field3DImpl<float>(Dim3D(1, 1, 1), float()) {
            allocateMemory(_dim, _initVal);
        }

        virtual ~Array3DLinearFortranField3DAdapter() {}

        virtual void allocateMemory(const Dim3D theDim, float &_initVal) {
            container.clear();
            internalDim.x = theDim.x + 1;
            internalDim.y = theDim.y + 1;
            internalDim.z = theDim.z + 1;

            dim = theDim;

            container.assign((internalDim.x) * (internalDim.y) * (internalDim.z), _initVal);


        }

        virtual void setDim(const Dim3D newDim) {
            this->resizeAndShift(newDim);
        }

        virtual void resizeAndShift(const Dim3D newDim, Dim3D shiftVec = Dim3D()) {
            vector<double> tmpContainer = container;
            tmpContainer.swap(container);// swapping vector content  => copy old vec to new

            Dim3D oldInternalDim = internalDim;

            internalDim.x = newDim.x + 1;
            internalDim.y = newDim.y + 1;
            internalDim.z = newDim.z + 1;


            container.assign((internalDim.x) * (internalDim.y) * (internalDim.z),
                             0.0); //resize container and initialize it

            //copying old array to new one - we do not copy swap values or boundary conditions these are usually
            // set externally by the solver
            Point3D pt;
            Point3D ptShift;

            //when lattice is growing or shrinking
            for (pt.x = 0; pt.x < newDim.x; ++pt.x)
                for (pt.y = 0; pt.y < newDim.y; ++pt.y)
                    for (pt.z = 0; pt.z < newDim.z; ++pt.z) {

                        ptShift = pt - shiftVec;
                        if (ptShift.x >= 0 && ptShift.x < dim.x && ptShift.y >= 0 && ptShift.y < dim.y &&
                            ptShift.z >= 0 && ptShift.z < dim.z) {
                            container[pt.x + (pt.y + pt.z * internalDim.y) * internalDim.x] = tmpContainer[ptShift.x +
                                                                                                           (ptShift.y +
                                                                                                            ptShift.z *
                                                                                                            oldInternalDim.y) *
                                                                                                           oldInternalDim.x];
                        }
                    }
            //setting dim to new dim
            dim = newDim;


        }


        std::vector<double> &getContainerRef() { return container; }

        double *getContainerArrayPtr() { return &(container[0]); }

        inline unsigned int index(int _x, int _y, int _z) const {

            //start indexing from 0'th element but calculate index based on increased lattice dimmension
            return _x + (_y + _z * internalDim.y) * internalDim.x;
        }


        virtual Dim3D getDim() const { return dim; }

        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < dim.x &&
                    0 <= pt.y && pt.y < dim.y &&
                    0 <= pt.z && pt.z < dim.z);
        }

        virtual void set(const Point3D &pt, const float value) {
            container[index(pt.x, pt.y, pt.z)] = value;

        }

        virtual float get(const Point3D &pt) const {
            //TODO: we'd better check why we cast from doubles to floats here
            //Either doubles are not used, either interface is incorrect
            return container[index(pt.x, pt.y, pt.z)];

        }

        void setQuick(const Point3D &pt, const float value) {
            container[index(pt.x, pt.y, pt.z)] = value;

        }

        float getQuick(const Point3D &pt) const {
            return container[index(pt.x, pt.y, pt.z)];

        }

        void setQuick(int _x, int _y, int _z, float value) {
            container[index(_x, _y, _z)] = value;

        }

        float getQuick(int _x, int _y, int _z) const {
            return container[index(_x, _y, _z)];

        }


        virtual float getByIndex(long _offset) const {
            return float();
        }

        virtual void setByIndex(long _offset, const float _value) {

        }


    protected:
        std::vector<double> container;
        //Dim3D dim; //defined already in Field3DImpl<>
        Dim3D internalDim;
    };


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//Adapter is necessary to keep current API of the Field3D . 2D array is allocated as a single chunk of memory - therefore trhe name linear. this particular array acts as a proxy for Fortran Arrays
//Fortran style array  shifts all the elements by (1,1) vector and 
//actually it looks like indexing begins at 0,0 but we need to use increaze lattice size by 1 in index incalculation to obtain correct index value
//has dimensions (m+1)x(n+1)
//last dimension of the array is iterated in the innermost loop 
//NOTICE: to get precise solutions from the solver we need to use double precision calculations (floats are imprecise)
//however since player is assuming float numbers as a return values of the Field3DImpl<T> API we need to 
//make base class  Field3DImpl<float>


    class Array2DLinearFortranField3DAdapter : public Field3DImpl<float> {
    public:
        //typedef float precision_t;
        Array2DLinearFortranField3DAdapter() : Field3DImpl<float>(Dim3D(1, 1, 1), float()) {};

        Array2DLinearFortranField3DAdapter(Dim3D &_dim, float &_initVal) : Field3DImpl<float>(Dim3D(1, 1, 1), float()) {
            allocateMemory(_dim, _initVal);
        }

        virtual ~Array2DLinearFortranField3DAdapter() {}

        virtual void allocateMemory(const Dim3D theDim, float &_initVal) {
            container.clear();
            internalDim.x = theDim.x + 1;
            internalDim.y = theDim.y + 1;
            internalDim.z = 1;
            dim = theDim;
            dim.z = 1;

            container.assign((internalDim.x) * (internalDim.y), _initVal);

        }

        virtual void setDim(const Dim3D newDim) {
            this->resizeAndShift(newDim);
        }

        virtual void resizeAndShift(const Dim3D newDim, Dim3D shiftVec = Dim3D()) {
            vector<double> tmpContainer = container;
            tmpContainer.swap(container);// swapping vector content  => copy old vec to new

            Dim3D oldInternalDim = internalDim;

            internalDim.x = newDim.x + 1;
            internalDim.y = newDim.y + 1;
            internalDim.z = 1;


            container.assign((internalDim.x) * (internalDim.y), 0.0); //resize container and initialize it

            //copying old array to new one - we do not copy swap values or boundary conditions
            // these are usually set externally by the solver
            Point3D pt;
            Point3D ptShift;

            //when lattice is growing or shrinking
            for (pt.x = 0; pt.x < newDim.x; ++pt.x)
                for (pt.y = 0; pt.y < newDim.y; ++pt.y) {
                    ptShift = pt - shiftVec;
                    if (ptShift.x >= 0 && ptShift.x < dim.x && ptShift.y >= 0 && ptShift.y < dim.y) {


                        container[pt.x + (pt.y) * internalDim.x] = tmpContainer[ptShift.x +
                                                                                (ptShift.y) * oldInternalDim.x];

                    }
                }
            //setting dim to new dim
            dim = newDim;
            dim.z = 1;


        }

        std::vector<double> &getContainerRef() { return container; }

        double *getContainerArrayPtr() { return &(container[0]); }

        inline unsigned int index(int _x, int _y) const {

            //start indexing from 0'th element but calculate index based on increased lattice dimmension
            return _x + (_y) * internalDim.x;

        }

        virtual Dim3D getDim() const { return dim; };

        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < dim.x &&
                    0 <= pt.y && pt.y < dim.y);
        }

        virtual void set(const Point3D &pt, const float value) {
            container[index(pt.x, pt.y)] = value;

        }

        virtual float get(const Point3D &pt) const {

            return container[Array2DLinearFortranField3DAdapter::index(pt.x, pt.y)];

        };

        void setQuick(const Point3D &pt, const float value) {
            container[index(pt.x, pt.y)] = value;

        }

        virtual float getQuick(const Point3D &pt) const {
            return container[index(pt.x, pt.y)];

        }

        void setQuick(int _x, int _y, float value) {
            container[index(_x, _y)] = value;

        }

        virtual float getQuick(int _x, int _y) const {
            return container[index(_x, _y)];

        };


        virtual float getByIndex(long _offset) const {
            return float();
        }

        virtual void setByIndex(long _offset, const float _value) {

        }

    protected:
        std::vector<double> container;
        Dim3D internalDim;
    };


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    template<typename T>
    class Array3DBorders {
    public:
        typedef T ***ContainerType;

        Array3DBorders() : array(0), borderWidth(1) {
        }

        virtual ~Array3DBorders() { freeMemory(); };

        void allocateArray(const Dim3D &_dim, T &val = T());

        ContainerType &getContainer() { return array; }

        void setBorderWidth(unsigned int _borderWidth) { borderWidth = _borderWidth; }

        unsigned int getBorderWidth() { return borderWidth; }

        bool switchContainersQuick(Array3DBorders<T> &_array3D);

        Dim3D getInternalDim() { return internalDim; }

    protected:
        ContainerType array;
        unsigned int borderWidth;
        Dim3D internalDim;

        void allocateMemory(const Dim3D &_dim, T &val = T());

        void freeMemory();

    };


    template<typename T>
    void Array3DBorders<T>::allocateArray(const Dim3D &_dim, T &val) {

        internalDim = _dim;
        freeMemory();
        allocateMemory(internalDim, val);

    }

    template<typename T>
    void Array3DBorders<T>::allocateMemory(const Dim3D &_dim, T &val) {

        array = new T **[_dim.x];
        for (int i = 0; i < _dim.x; ++i) {
            array[i] = new T *[_dim.y];
        }

        for (int i = 0; i < _dim.x; ++i)
            for (int j = 0; j < _dim.y; ++j) {
                array[i][j] = new T[_dim.z];
            }

        for (int i = 0; i < _dim.x; ++i)
            for (int j = 0; j < _dim.y; ++j)
                for (int k = 0; k < _dim.z; ++k) {
                    array[i][j][k] = val;
                }

    }

#include <iostream>
template<typename T>
void Array3DBorders<T>::freeMemory() {
    using namespace std;

    if (array) {

        for (int i = 0; i < internalDim.x; ++i)
            for (int j = 0; j < internalDim.y; ++j) {

                delete[] array[i][j];
                array[i][j] = 0;
            }

        for (int i = 0; i < internalDim.x; ++i) {
            delete[] array[i];
            array[i] = 0;
        }
        delete[] array;
        array = 0;

    }


}


template<typename T>
bool Array3DBorders<T>::switchContainersQuick(Array3DBorders<T> &_switchedArray) {
    ContainerType tmpPtr;
    ContainerType &switchedArrayPtr = _switchedArray.getContainer();

    tmpPtr = array;
    array = switchedArrayPtr;
    switchedArrayPtr = tmpPtr;
    return true;
}

//Adapter is necessary to keep current API of the Field3D . These fields are registered in Simulator and 
template<typename T>
class Array3DBordersField3DAdapter : public Field3DImpl<T>, public Array3DBorders<T> {
public:
    Array3DBordersField3DAdapter() : Field3DImpl<T>(Dim3D(1, 1, 1), T()), Array3DBorders<T>() {};

    Array3DBordersField3DAdapter(Dim3D &_dim, T &_initVal) : Field3DImpl<T>(Dim3D(1, 1, 1), T()), Array3DBorders<T>() {
        this->allocateMemory(_dim, _initVal);
        Array3DBorders<T>::internalDim = _dim;
    }

    virtual ~Array3DBordersField3DAdapter() {

    };


    virtual void setDim(const Dim3D theDim) {
        T t;

        this->allocateMemory(theDim, t);
        Array3DBorders<T>::internalDim = theDim;

    }

    virtual Dim3D getDim() const {

        return Dim3D(Array3DBorders<T>::internalDim.x - 2 * Array3DBorders<T>::borderWidth,
                     Array3DBorders<T>::internalDim.y - 2 * Array3DBorders<T>::borderWidth,
                     Array3DBorders<T>::internalDim.z - 2 * Array3DBorders<T>::borderWidth);
    };

    virtual bool isValid(const Point3D &pt) const {
        return (0 <= pt.x && pt.x < Array3DBorders<T>::internalDim.x &&
                0 <= pt.y && pt.y < Array3DBorders<T>::internalDim.y &&
                0 <= pt.z && pt.z < Array3DBorders<T>::internalDim.z);
    }

    virtual void set(const Point3D &pt, const T value) {

        Array3DBorders<T>::array
        [pt.x + Array3DBorders<T>::borderWidth]
        [pt.y + Array3DBorders<T>::borderWidth]
        [pt.z + Array3DBorders<T>::borderWidth] = value;

    }


    virtual T get(const Point3D &pt) const {

        return Array3DBorders<T>::array
        [pt.x + Array3DBorders<T>::borderWidth]
        [pt.y + Array3DBorders<T>::borderWidth]
        [pt.z + Array3DBorders<T>::borderWidth];
    }

    virtual T getByIndex(long _offset) const {
        return T();
    }

    virtual void setByIndex(long _offset, const T _value) {

    }

protected:

};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T>
class Array2DBorders {
public:
    typedef T **ContainerType;

    Array2DBorders() : array(0), borderWidth(1) {
    }

    virtual ~Array2DBorders() { freeMemory(); };

    void allocateArray(const Dim3D &_dim, T &val = T());

    ContainerType &getContainer() { return array; }

    void setBorderWidth(unsigned int _borderWidth) { borderWidth = _borderWidth; }

    unsigned int getBorderWidth() { return borderWidth; }

    bool switchContainersQuick(Array2DBorders<T> &_array2D);

    Dim3D getInternalDim() { return internalDim; }

protected:
    ContainerType array;
    unsigned int borderWidth;
    Dim3D internalDim;

    void allocateMemory(const Dim3D &_dim, T &val = T());

    void freeMemory();

};


template<typename T>
void Array2DBorders<T>::allocateArray(const Dim3D &_dim, T &val) {

    internalDim = _dim;
    freeMemory();
    allocateMemory(internalDim, val);

}

template<typename T>
void Array2DBorders<T>::allocateMemory(const Dim3D &_dim, T &val) {

    array = new T *[_dim.x];
    for (int i = 0; i < _dim.x; ++i) {
        array[i] = new T[_dim.y];
    }

    for (int i = 0; i < _dim.x; ++i)
        for (int j = 0; j < _dim.y; ++j)
            array[i][j] = val;


}

#include <iostream>

    template<typename T>
    void Array2DBorders<T>::freeMemory() {
        using namespace std;

        if (array) {

            for (int i = 0; i < internalDim.x; ++i) {
                delete[] array[i];
                array[i] = 0;
            }

            delete[] array;
            array = 0;

        }


    }


    template<typename T>
    bool Array2DBorders<T>::switchContainersQuick(Array2DBorders<T> &_switchedArray) {
        ContainerType tmpPtr;
        ContainerType &switchedArrayPtr = _switchedArray.getContainer();

        tmpPtr = array;
        array = switchedArrayPtr;
        switchedArrayPtr = tmpPtr;
        return true;
    }

//Adapter is necessary to keep current API of the Field3D . These fields are registered in Simulator and 
    template<typename T>
    class Array2DBordersField3DAdapter : public Field3DImpl<T>, public Array2DBorders<T> {
    public:
        Array2DBordersField3DAdapter() : Field3DImpl<T>(Dim3D(1, 1, 1), T()), Array2DBorders<T>() {};

        Array2DBordersField3DAdapter(Dim3D &_dim, T &_initVal) : Field3DImpl<T>(Dim3D(1, 1, 1), T()),
                                                                 Array2DBorders<T>() {
            allocateMemory(_dim, _initVal);
            Array2DBorders<T>::internalDim = _dim;
        }

        virtual ~Array2DBordersField3DAdapter() {

        };

        virtual void setDim(const Dim3D theDim) {
            T t;

            allocateMemory(theDim, t);
            Array2DBorders<T>::internalDim = theDim;

        }

        virtual Dim3D getDim() const {

            return Dim3D(Array2DBorders<T>::internalDim.x - 2 * Array2DBorders<T>::borderWidth,
                         Array2DBorders<T>::internalDim.y - 2 * Array2DBorders<T>::borderWidth,
                         1);
        };

        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < Array2DBorders<T>::internalDim.x &&
                    0 <= pt.y && pt.y < Array2DBorders<T>::internalDim.y &&
                    0 == pt.z);
        }

        virtual void set(const Point3D &pt, const T value) {
            Array2DBorders<T>::array
            [pt.x + Array2DBorders<T>::borderWidth]
            [pt.y + Array2DBorders<T>::borderWidth]
                    = value;

        }

        virtual T get(const Point3D &pt) const {
            return Array2DBorders<T>::array
            [pt.x + Array2DBorders<T>::borderWidth]
            [pt.y + Array2DBorders<T>::borderWidth];
        }

        virtual T getByIndex(long _offset) const {
            return T();
        }

        virtual void setByIndex(long _offset, const T _value) {

        }

    protected:

    };




//Array3DContiguous is a special container where scratch and concentration field
// are stored next to each other to optimize read/write operations.
// this should result in significant speed up of FE solvers

    template<typename T>
    class Array3DContiguous : public Field3DImpl<T> {
    public:
        typedef T *ContainerType;

        Array3DContiguous() :
                Field3DImpl<T>(Dim3D(1, 1, 1), T()),
                arrayCont(0),
                arraySize(0),
                shiftArray(0),
                shiftSwap(1),
                borderWidth(1) {}

        Array3DContiguous(Dim3D &_dim, T &_initVal = T()) :
                Field3DImpl<T>(Dim3D(1, 1, 1), T()),
                arrayCont(0),
                arraySize(0),
                shiftArray(0),
                shiftSwap(1),
                borderWidth(1) {
            allocateArray(_dim, _initVal);
        }

        ~Array3DContiguous() {
            if (arrayCont) {
                delete[] arrayCont;
            }
            arrayCont = 0;
        }

        virtual void setDim(const Dim3D theDim);

        virtual void resizeAndShift(const Dim3D theDim, Dim3D shiftVec = Dim3D());

        void allocateArray(const Dim3D &_dim, T val = T());

        ContainerType getContainer() { return arrayCont; }

        T get(int x, int y, int z) const { return getDirect(x + borderWidth, y + borderWidth, z + borderWidth); }

        void set(int x, int y, int z, T t) { return setDirect(x + borderWidth, y + borderWidth, z + borderWidth, t); }


        T getDirect(int x, int y, int z) const {
            return arrayCont[((x + shiftArray) +
                              (((y + shiftArray) + ((2 * z + shiftArray) * internalDim.y)) * internalDim.x))];
        }

        void setDirect(int x, int y, int z, T t) {
            arrayCont[((x + shiftArray) +
                       (((y + shiftArray) + ((2 * z + shiftArray) * internalDim.y)) * internalDim.x))] = t;
        }

        T getDirectSwap(int x, int y, int z) const {
            return arrayCont[((x + shiftSwap) +
                              (((y + shiftSwap) + ((2 * z + shiftSwap) * internalDim.y)) * internalDim.x))];
        }

        void setDirectSwap(int x, int y, int z, T t) {
            arrayCont[((x + shiftSwap) +
                       (((y + shiftSwap) + ((2 * z + shiftSwap) * internalDim.y)) * internalDim.x))] = t;
        }

        //Field3D interface
        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < internalDim.x &&
                    0 <= pt.y && pt.y < internalDim.y &&
                    0 <= pt.z && pt.z < internalDim.z);
        }

        virtual void set(const Point3D &pt, const T value) {

            set(pt.x, pt.y, pt.z, value);

        }

        virtual T get(const Point3D &pt) const {

            return get(pt.x, pt.y, pt.z);
        }

        virtual T getByIndex(long _offset) const {
            return T();
        }

        virtual void setByIndex(long _offset, const T _value) {

        }


        void swapArrays();

        Dim3D getInternalDim() { return internalDim; }

        Dim3D getDim() { return Field3DImpl<T>::dim; }

        int getShiftArray() { return shiftArray; }

        int getShiftSwap() { return shiftSwap; }

        void swapQuick(Array3DContiguous &_switchedArray) {
            ContainerType tmpPtr;
            ContainerType switchedArrayPtr = _switchedArray.getContainer();

            tmpPtr = arrayCont;
            arrayCont = switchedArrayPtr;
            switchedArrayPtr = tmpPtr;

        }

    protected:
        T *arrayCont;
        Dim3D internalDim;
        int arraySize;
        int shiftSwap;
        int shiftArray;
        int borderWidth;

    };

    template<typename T>
    void Array3DContiguous<T>::allocateArray(const Dim3D &_dim, T val) {

        Field3DImpl<T>::dim = _dim;
        internalDim = Field3DImpl<T>::dim;
        internalDim.x += 3;
        internalDim.y += 3;
        internalDim.z += 3;

        if (arrayCont) {
            delete[] arrayCont;
            arrayCont = 0;
        }
        arraySize = internalDim.x * internalDim.y * 2 * internalDim.z;
        arrayCont = new T[arraySize];

        //initialization
        for (int i = 0; i < arraySize; ++i) {
            arrayCont[i] = val;
        }

    }


    template<typename T>
    void Array3DContiguous<T>::resizeAndShift(const Dim3D newDim, Dim3D shiftVec) {

        Dim3D newInternalDim = newDim;
        newInternalDim.x += 3;
        newInternalDim.y += 3;
        newInternalDim.z += 3;

        int newArraySize = newInternalDim.x * newInternalDim.y * 2 * newInternalDim.z;
        T *newArrayCont = new T[newArraySize];

        //initialization
        for (int i = 0; i < newArraySize; ++i) {
            newArrayCont[i] = T();
        }

        //copying old array to new one - we do not copy swap values or boundary conditions
        // these are usually set externally by the solver
        Point3D pt;
        Point3D ptShift;

        //when lattice is growing or shrinking
        for (pt.x = 0; pt.x < newDim.x; ++pt.x)
            for (pt.y = 0; pt.y < newDim.y; ++pt.y)
                for (pt.z = 0; pt.z < newDim.z; ++pt.z) {

                    ptShift = pt - shiftVec;
                    if (ptShift.x >= 0 && ptShift.x < Field3DImpl<T>::dim.x && ptShift.y >= 0 &&
                        ptShift.y < Field3DImpl<T>::dim.y && ptShift.z >= 0 && ptShift.z < Field3DImpl<T>::dim.z) {

                        newArrayCont[(((pt.x + borderWidth) + shiftArray) + ((((pt.y + borderWidth) + shiftArray) +
                                                                              ((2 * (pt.z + borderWidth) + shiftArray) *
                                                                               newInternalDim.y)) *
                                                                             newInternalDim.x))] = get(ptShift.x,
                                                                                                       ptShift.y,
                                                                                                       ptShift.z);

                    }
                }

        //swapping array and deallocation old one
        internalDim = newInternalDim;
        Field3DImpl<T>::dim = newDim;
        arraySize = newArraySize;

        delete[] arrayCont;

        arrayCont = newArrayCont;


    }

    template<typename T>
    void Array3DContiguous<T>::setDim(const Dim3D newDim) {
        this->resizeAndShift(newDim);
    }


    template<typename T>
    void Array3DContiguous<T>::swapArrays() {
        if (shiftSwap) {
            shiftSwap = 0;
            shiftArray = 1;
        } else {
            shiftSwap = 1;
            shiftArray = 0;

        }
    }




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


//Array3DCUDA is a special container designed for CUDA purposes
//had to fo static_cast<T>(0) instead of T() to ensure that it will compile on gcc in the SWIG generated wrapper file
    template<typename T>
    class Array3DCUDA : public Field3DImpl<T> {
    public:
        typedef T *ContainerType;

        Array3DCUDA() :
                Field3DImpl<T>(Dim3D(1, 1, 1), static_cast<T>(0)),
                arrayCont(0),
                arraySize(0),
                borderWidth(1) {}

        Array3DCUDA(Dim3D &_dim, T _initVal = static_cast<T>(0)) :
                Field3DImpl<T>(Dim3D(1, 1, 1), static_cast<T>(0)),
                arrayCont(0),
                arraySize(0),
                borderWidth(1) {
            allocateArray(_dim, _initVal);
        }

        ~Array3DCUDA() {
            if (arrayCont) {
                free(arrayCont);
            }
            arrayCont = 0;
        }

        void allocateArray(const Dim3D &_dim, T val = static_cast<T>(0));

        ContainerType getContainer() { return arrayCont; }


        T get(int x, int y, int z) const { return getDirect(x + borderWidth, y + borderWidth, z + borderWidth); }

        void set(int x, int y, int z, T t) { return setDirect(x + borderWidth, y + borderWidth, z + borderWidth, t); }


        T getDirect(int x, int y, int z) const {
            return arrayCont[z * (internalDim.x) * (internalDim.y) + y * (internalDim.x) + x];
        }

        void setDirect(int x, int y, int z, T t) {
            arrayCont[z * (internalDim.x) * (internalDim.y) + y * (internalDim.x) + x] = t;
        }

        //Field3D interface
        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < internalDim.x &&
                    0 <= pt.y && pt.y < internalDim.y &&
                    0 <= pt.z && pt.z < internalDim.z);
        }

        virtual void set(const Point3D &pt, const T value) {


            set(pt.x, pt.y, pt.z, value);

        }

        virtual T get(const Point3D &pt) const {

            return get(pt.x, pt.y, pt.z);
        }

        virtual T getByIndex(long _offset) const {
            return T();
        }

        virtual void setByIndex(long _offset, const T _value) {

        }


        void swapArrays();

        int getArraySize() { return arraySize; }

        Dim3D getInternalDim() { return internalDim; }

        Dim3D getDim() { return Field3DImpl<T>::dim; }

        void swapQuick(Array3DCUDA &_switchedArray) {
        }

    protected:
        T *arrayCont;
        Dim3D internalDim;
        int arraySize;
        int borderWidth;

    };


    template<typename T>
    void Array3DCUDA<T>::allocateArray(const Dim3D &_dim, T val) {

        Field3DImpl<T>::dim = _dim;
        internalDim = Field3DImpl<T>::dim;
        internalDim.x += 2;
        internalDim.y += 2;
        internalDim.z += 2;

        if (arrayCont) {
            free(arrayCont);
            arrayCont = 0;
        }
        arraySize = internalDim.x * internalDim.y * internalDim.z;
        arrayCont = (ContainerType) malloc(arraySize * sizeof(T));

        //initialization
        for (int i = 0; i < arraySize; ++i) {
            arrayCont[i] = val;
        }

    }

    template<typename T>
    void Array3DCUDA<T>::swapArrays() {

    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



    //Array2DContiguous is a special container where scratch and concentration field are stored
    // next to each other to optimize read/write operations. this should result in significant speed up of FE solvers
    template<typename T>
    class Array2DContiguous : public Field3DImpl<T> {
    public:
        typedef T *ContainerType;

        Array2DContiguous() :
                Field3DImpl<T>(Dim3D(1, 1, 1), T()),
                arrayCont(0),
                arraySize(0),
                shiftArray(0),
                shiftSwap(1),
                borderWidth(1) {}

        Array2DContiguous(Dim3D &_dim, T &_initVal = T()) :
                Field3DImpl<T>(Dim3D(1, 1, 1), T()),
                arrayCont(0),
                arraySize(0),
                shiftArray(0),
                shiftSwap(1),
                borderWidth(1) {
            allocateArray(_dim, _initVal);
        }

        ~Array2DContiguous() {
            if (arrayCont) {
                delete[] arrayCont;
            }
            arrayCont = 0;
        }

        void allocateArray(const Dim3D &_dim, T val = T());

        virtual void setDim(const Dim3D newDim);

        virtual void resizeAndShift(const Dim3D newDim, Dim3D shiftVec = Dim3D());

        ContainerType getContainer() { return arrayCont; }


        T get(int x, int y) const { return getDirect(x + borderWidth, y + borderWidth); }

        void set(int x, int y, T t) { return setDirect(x + borderWidth, y + borderWidth, t); }


        T getDirect(int x, int y) const {

            return arrayCont[x + shiftArray + (2 * y + shiftArray) * internalDim.x];
        }

        void setDirect(int x, int y, T t) {
            arrayCont[x + shiftArray + (2 * y + shiftArray) * internalDim.x] = t;
        }

        T getDirectSwap(int x, int y) const {
            return arrayCont[x + shiftSwap + (2 * y + shiftSwap) * internalDim.x];
        }

        void setDirectSwap(int x, int y, T t) {
            arrayCont[x + shiftSwap + (2 * y + shiftSwap) * internalDim.x] = t;
        }

        //Field3D interface
        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < internalDim.x &&
                    0 <= pt.y && pt.y < internalDim.y);
        }

        virtual void set(const Point3D &pt, const T value) {

            set(pt.x, pt.y, value);
        }

        virtual T get(const Point3D &pt) const {

            return get(pt.x, pt.y);
        }

        virtual T getByIndex(long _offset) const {
            return T();
        }

        virtual void setByIndex(long _offset, const T _value) {
        }

        void swapArrays();

        Dim3D getInternalDim() { return internalDim; }

        Dim3D getDim() { return Field3DImpl<T>::dim; }

        int getShiftArray() { return shiftArray; }

        int getShiftSwap() { return shiftSwap; }

        void swapQuick(Array2DContiguous &_switchedArray) {
            ContainerType tmpPtr;
            ContainerType switchedArrayPtr = _switchedArray.getContainer();

            tmpPtr = arrayCont;
            arrayCont = switchedArrayPtr;
            switchedArrayPtr = tmpPtr;

        }

    protected:
        T *arrayCont;
        Dim3D internalDim;
        int arraySize;
        int shiftSwap;
        int shiftArray;
        int borderWidth;

    };

    template<typename T>
    void Array2DContiguous<T>::allocateArray(const Dim3D &_dim, T val) {

        Field3DImpl<T>::dim = _dim;
        internalDim = Field3DImpl<T>::dim;
        internalDim.x += 3;
        internalDim.y += 3;
        internalDim.z = 1;


        if (arrayCont) {
            delete[] arrayCont;
            arrayCont = 0;
        }
        arraySize = internalDim.x * 2 * internalDim.y;
        arrayCont = new T[arraySize];

        //initialization
        for (int i = 0; i < arraySize; ++i) {
            arrayCont[i] = val;
        }
    }

    template<typename T>
    void Array2DContiguous<T>::swapArrays() {
        if (shiftSwap) {
            shiftSwap = 0;
            shiftArray = 1;
        } else {
            shiftSwap = 1;
            shiftArray = 0;
        }
    }


    template<typename T>
    void Array2DContiguous<T>::resizeAndShift(const Dim3D newDim, Dim3D shiftVec) {

        Dim3D newInternalDim = newDim;
        newInternalDim.x += 3;
        newInternalDim.y += 3;
        newInternalDim.z = 1;

        int newArraySize = newInternalDim.x * newInternalDim.y * 2;
        T *newArrayCont = new T[newArraySize];

        //initialization
        for (int i = 0; i < newArraySize; ++i) {
            newArrayCont[i] = T();
        }

        //copying old array to new one - we do not copy swap values or
        // boundary conditions these are usually set externally by the solver
        Point3D pt;
        Point3D ptShift;

        //when lattice is growing or shrinking
        for (pt.x = 0; pt.x < newDim.x; ++pt.x)
            for (pt.y = 0; pt.y < newDim.y; ++pt.y) {
                ptShift = pt - shiftVec;
                if (ptShift.x >= 0 && ptShift.x < Field3DImpl<T>::dim.x && ptShift.y >= 0 &&
                    ptShift.y < Field3DImpl<T>::dim.y) {
                    newArrayCont[(pt.x + borderWidth) + shiftArray +
                                 (2 * (pt.y + borderWidth) + shiftArray) * newInternalDim.x] = arrayCont[ptShift.x +
                                                                                                         borderWidth +
                                                                                                         shiftArray +
                                                                                                         (2 *
                                                                                                          (ptShift.y +
                                                                                                           borderWidth) +
                                                                                                          shiftArray) *
                                                                                                         internalDim.x];
                }
            }

        //swapping array and deallocation old one
        internalDim = newInternalDim;
        Field3DImpl<T>::dim = newDim;
        arraySize = newArraySize;

        delete[] arrayCont;

        arrayCont = newArrayCont;


    }

    template<typename T>
    void Array2DContiguous<T>::setDim(const Dim3D newDim) {
        this->resizeAndShift(newDim);
    }


    template<typename T>
    class Array3DFiPy : public Field3DImpl<T> {
    public:
        std::vector <std::vector<float>> secData;
        std::vector <std::vector<int>> doNotDiffuseData;
        typedef T *ContainerType;

        Array3DFiPy() :
                Field3DImpl<T>(Dim3D(1, 1, 1), T()),
                arrayCont(0),
                arraySize(0),
                shiftArray(0),
                shiftSwap(1),
                borderWidth(1) {}

        Array3DFiPy(Dim3D &_dim, T &_initVal = T()) :
                Field3DImpl<T>(Dim3D(1, 1, 1), T()),
                arrayCont(0),
                arraySize(0),
                shiftArray(0),
                shiftSwap(1),
                borderWidth(1) {
            allocateArray(_dim, _initVal);
        }

        ~Array3DFiPy() {
            if (arrayCont) {
                delete[] arrayCont;
            }
            arrayCont = 0;
        }

        void allocateArray(const Dim3D &_dim, T val = T());

        ContainerType getContainer() { return arrayCont; }

        T get(int x, int y, int z) const { return getDirect(x + borderWidth, y + borderWidth, z + borderWidth); }

        void set(int x, int y, int z, T t) { return setDirect(x + borderWidth, y + borderWidth, z + borderWidth, t); }

        T getDirect(int x, int y, int z) const {
            return arrayCont[((x + shiftArray) +
                              (((y + shiftArray) + ((2 * z + shiftArray) * internalDim.y)) * internalDim.x))];
        }

        void setDirect(int x, int y, int z, T t) {
            arrayCont[((x + shiftArray) +
                       (((y + shiftArray) + ((2 * z + shiftArray) * internalDim.y)) * internalDim.x))] = t;
        }

        T getDirectSwap(int x, int y, int z) const {
            return arrayCont[((x + shiftSwap) +
                              (((y + shiftSwap) + ((2 * z + shiftSwap) * internalDim.y)) * internalDim.x))];
        }

        void setDirectSwap(int x, int y, int z, T t) {
            arrayCont[((x + shiftSwap) +
                       (((y + shiftSwap) + ((2 * z + shiftSwap) * internalDim.y)) * internalDim.x))] = t;
        }

        void setDoNotDiffuseData(std::vector <std::vector<int>> &_doNotDiffuseData) {
            doNotDiffuseData = _doNotDiffuseData;
        }

        std::vector <std::vector<int>> &getDoNotDiffuseVec() {
            return doNotDiffuseData;
        }

        void storeSecData(int x, int y, int z, T t) {
            vector<float> fourVector(4);
            fourVector[0] = x;
            fourVector[1] = y;
            fourVector[2] = z;
            fourVector[3] = t;
            secData.push_back(fourVector);
        }

        std::vector <std::vector<float>> &getSecretionData() {
            return secData;
        }

        void clearSecData() {
            secData.clear();
        }

        //Field3D interface
        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < internalDim.x &&
                    0 <= pt.y && pt.y < internalDim.y &&
                    0 <= pt.z && pt.z < internalDim.z);
        }

        virtual void set(const Point3D &pt, const T value) {
            set(pt.x, pt.y, pt.z, value);

        }

        virtual T get(const Point3D &pt) const {
            return get(pt.x, pt.y, pt.z);
        }

        virtual T getByIndex(long _offset) const {
            return T();
        }

        virtual void setByIndex(long _offset, const T _value) {
        }


        void swapArrays();

        Dim3D getInternalDim() { return internalDim; }

        Dim3D getDim() { return Field3DImpl<T>::dim; }

        int getShiftArray() { return shiftArray; }

        int getShiftSwap() { return shiftSwap; }

        void swapQuick(Array3DFiPy &_switchedArray) {
            ContainerType tmpPtr;
            ContainerType switchedArrayPtr = _switchedArray.getContainer();

            tmpPtr = arrayCont;
            arrayCont = switchedArrayPtr;
            switchedArrayPtr = tmpPtr;

        }

    protected:
        T *arrayCont;
        Dim3D internalDim;
        int arraySize;
        int shiftSwap;
        int shiftArray;
        int borderWidth;

    };

    template<typename T>
    void Array3DFiPy<T>::allocateArray(const Dim3D &_dim, T val) {

        Field3DImpl<T>::dim = _dim;
        internalDim = Field3DImpl<T>::dim;
        internalDim.x += 3;
        internalDim.y += 3;
        internalDim.z += 3;

        if (arrayCont) {
            delete[] arrayCont;
            arrayCont = 0;
        }
        arraySize = internalDim.x * internalDim.y * 2 * internalDim.z;
        arrayCont = new T[arraySize];

        //initialization
        for (int i = 0; i < arraySize; ++i) {
            arrayCont[i] = val;
        }

    }

    template<typename T>
    void Array3DFiPy<T>::swapArrays() {
        if (shiftSwap) {
            shiftSwap = 0;
            shiftArray = 1;
        } else {
            shiftSwap = 1;
            shiftArray = 0;

        }
    }


};

#endif
