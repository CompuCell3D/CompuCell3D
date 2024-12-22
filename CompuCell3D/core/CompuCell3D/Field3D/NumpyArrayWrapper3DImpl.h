//
// Created by m on 10/5/24.
//

#ifndef COMPUCELL3D_NUMPYARRAYWRAPPER3DIMPL_H
#define COMPUCELL3D_NUMPYARRAYWRAPPER3DIMPL_H

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
#include "Field3D.h"


namespace CompuCell3D {


    typedef std::vector<double>::size_type array_size_t;

    template<typename T>
    class NumpyArrayWrapper3DImpl : public Field3D<T>, public NumpyArrayWrapperImpl<T> {
    protected:
        Dim3D dim;
        array_size_t padding;

    public:
        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */

        NumpyArrayWrapper3DImpl(const std::vector<array_size_t> &dims, array_size_t padding=0) :
                NumpyArrayWrapperImpl<T>(dims, padding),
                        padding(padding)
            {
            if (dims.size() != 3) {
                throw CC3DException("NumpyArrayWrapperImpl3D must have exactly 3 dimensions!!!");
            }
            if (dims[0] == 0 && dims[1] == 0 && dims[2] == 0)
                throw CC3DException("NumpyArrayWrapperImpl3D cannot have a 0 dimension!!!");
            dim.x = this->dimensions[0];
            dim.y = this->dimensions[1];
            dim.z = this->dimensions[2];
        }

        virtual ~NumpyArrayWrapper3DImpl() = default;

        virtual array_size_t getPadding(){return padding;}

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
        virtual void set(const Point3D &pt, const T value) {
            this->array[this->index({static_cast<size_t>(pt.x+padding), static_cast<size_t>(pt.y+padding), static_cast<size_t>(pt.z+padding)})] = value;
        };

        virtual T get(const Point3D &pt) const {
            return this->array[this->index({static_cast<size_t>(pt.x+padding), static_cast<size_t>(pt.y+padding), static_cast<size_t>(pt.z+padding)})];

        };

        virtual bool isValid(const Point3D &pt) const {
            return (0 <= pt.x && pt.x < this->dimensions[0] &&
                    0 <= pt.y && pt.y < this->dimensions[1] &&
                    0 <= pt.z && pt.z < this->dimensions[2]);
        }

        virtual Dim3D getDim() const { return dim ;}
    };
};



#endif
