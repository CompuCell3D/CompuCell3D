//
// Created by m on 10/5/24.
//

#ifndef COMPUCELL3D_VECTORNUMPYARRAYWRAPPER3DIMPL_H
#define COMPUCELL3D_VECTORNUMPYARRAYWRAPPER3DIMPL_H

#include <vector>
#include <initializer_list>
#include <functional>  // For std::function


#include <vector>
#include <initializer_list>
#include <functional>  // For std::function
#include <numeric>  // For std::accumulate
#include <algorithm> // For std::copy
#include <iostream>

#include <math.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/CC3DExceptions.h>
#include "NumpyArrayWrapperImpl.h"
#include "Dim3D.h"
#include "VectorField3D.h"

template<typename T>
class Coordinates3D;


namespace CompuCell3D {


    typedef std::vector<double>::size_type array_size_t;

    template<typename T>
    class VectorNumpyArrayWrapper3DImpl : public VectorField3D<T>, public NumpyArrayWrapperImpl<T> {
    protected:
        Dim3D dim;

    public:
        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */

        VectorNumpyArrayWrapper3DImpl(const std::vector <array_size_t> &dims) :
                NumpyArrayWrapperImpl<T>(dims) {
            if (dims.size() != 4) {
                throw CC3DException("VectorNumpyArrayWrapperImpl3D must have exactly 4 dimensions!!!");
            }
            if (dims[3] != 3) {
                throw CC3DException("VectorNumpyArrayWrapperImpl3D 4 dimensions must be exactly 3 - for 3D vector!!!");
            }


            if (dims[0] == 0 && dims[1] == 0 && dims[2] == 0)
                throw CC3DException("VectorNumpyArrayWrapperImpl3D cannot have a 0 dimension!!!");
            dim.x = this->dimensions[0];
            dim.y = this->dimensions[1];
            dim.z = this->dimensions[2];
        }

        virtual ~VectorNumpyArrayWrapper3DImpl() = default;

        //Field 3D interface
        virtual void set(const Point3D &pt, const Coordinates3D<T> value) {

            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     0})] = value.x;
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     1})] = value.y;
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z), 2})] = value.z;
        };

        virtual void set(const Point3D &pt, const std::vector <T> value) {

            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     0})] = value[0];
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     1})] = value[1];
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     2})] = value[2];
        };


        virtual Coordinates3D<T> get(const Point3D &pt) const {
            return
                    Coordinates3D<T>(
                            this->array[this->index(
                                    {static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     0})],
                            this->array[this->index(
                                    {static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     1})],
                            this->array[this->index(
                                    {static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     2})]
                    );

        };

        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < this->dimensions[0] &&
                    0 <= pt.y && pt.y < this->dimensions[1] &&
                    0 <= pt.z && pt.z < this->dimensions[2]);
        }

        virtual Dim3D getDim() const { return dim; }
    };
};

//namespace CompuCell3D {
//
//
//    typedef std::vector<size_t>::size_type array_size_t;
//
//    class NumpyArrayWrapper {
//
//
//    private:
//
//        std::vector<array_size_t> dimensions;
//        std::vector<array_size_t> strides;
//        std::vector<double> array;
//
//
//    public:
//        /**
//         * @param dim The field dimensions
//         * @param initialValue The initial value of all data elements in the field.
//         */
//
//
//        NumpyArrayWrapper(const std::vector<array_size_t> &dims);
//
//
//        void setDimensions(const std::vector<array_size_t> &_dimensions) {
//            this->dimensions = _dimensions;
//        }
//        std::vector<array_size_t> computeStrides(const std::vector<array_size_t>& dims);
//
//
//        void iterateOverAxes(const std::vector<array_size_t> &dims,
//                             std::function<void(const std::vector<array_size_t> &)> functor);
//
//        void printArrayValue(const std::vector<array_size_t> &indices);
//
//        array_size_t index(const std::vector<array_size_t>& indices) const {
//            array_size_t index = 0;
//            for (size_t i = 0; i < indices.size(); ++i) {
//                index += indices[i] * strides[i];
//            }
//            return index;
//        }
//
//        array_size_t getSize() {
//            return array.size();
//
//        }
//
//        double *getPtr() {
//            return array.size() ? &array[0] : nullptr;
//        }
//
//        void printAllArrayValues();
//
//        virtual ~NumpyArrayWrapper() {}
//
//
//    };
//};


#endif
