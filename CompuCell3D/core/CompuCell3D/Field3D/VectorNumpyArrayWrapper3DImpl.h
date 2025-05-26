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

#include <cmath>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/CC3DExceptions.h>
#include "NumpyArrayWrapperImpl.h"
#include "Dim3D.h"
#include "VectorField3D.h"

# include "ndarray_adapter.h"


template<typename T>
class Coordinates3D;


namespace CompuCell3D {

    using array_size_t = size_t;
//    typedef std::vector<double>::size_type array_size_t;


    template<typename T>
    class VectorNumpyArrayWrapper3DImpl : public VectorField3D<T>, public NumpyArrayWrapperImpl<T> {
    protected:
        Dim3D dim;
        NdarrayAdapter<T, 4> ndarrayAdapter;
        std::string elementType;

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

//          initializing ndarrayAdapter
            ndarrayAdapter.setData( &(this->array[0]));

            // Convert each element from array_size_t to long explicitly
            std::vector<long> long_vector_dimensions(this->dimensions.size());
            std::transform(this->dimensions.begin(), this->dimensions.end(), long_vector_dimensions.begin(),
                           [](array_size_t val) { return static_cast<long>(val); });
            ndarrayAdapter.setShape(long_vector_dimensions);



            std::vector<long> long_vector_strides(this->dimensions.size());
            std::transform(this->strides.begin(), this->strides.end(), long_vector_strides.begin(),
                           [](array_size_t val) { return static_cast<long>(val); });

            ndarrayAdapter.setStrides(long_vector_strides);


        }

        virtual ~VectorNumpyArrayWrapper3DImpl() = default;

        std::string getElementType() const {
            if (std::is_same<T, float>::value) {
                return "float";
            } else if (std::is_same<T, double>::value) {
                return "double";
            } else if (std::is_same<T, int>::value) {
                return "int";
            } else {
                return "unknown";
            }
        }

        //Field 3D interface
        virtual void set(const Point3D &pt, const Coordinates3D<T>& value) {

            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     0})] = value.x;
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     1})] = value.y;
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z), 2})] = value.z;
        };

        virtual void set(const Point3D &pt, const std::vector <T>& value) {

            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     0})] = value[0];
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     1})] = value[1];
            this->array[this->index({static_cast<size_t>(pt.x), static_cast<size_t>(pt.y), static_cast<size_t>(pt.z),
                                     2})] = value[2];
        };


        virtual Coordinates3D<T> get(const Point3D &pt) const {
            size_t x = static_cast<size_t>(pt.x);
            size_t y = static_cast<size_t>(pt.y);
            size_t z = static_cast<size_t>(pt.z);
            return Coordinates3D<T>(
                this->array[this->index({x, y, z, 0})],
                this->array[this->index({x, y, z, 1})],
                this->array[this->index({x, y, z, 2})]
            );
        }


        virtual NdarrayAdapter<T, 4>* getNdarrayAdapter() {
            return &ndarrayAdapter;
        }

        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < this->dimensions[0] &&
                    0 <= pt.y && pt.y < this->dimensions[1] &&
                    0 <= pt.z && pt.z < this->dimensions[2]);
        }

        virtual Dim3D getDim() const { return dim; }
    };
};



#endif
