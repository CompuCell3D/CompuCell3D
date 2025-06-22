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
#include "Field3DTypeBase.h"


namespace CompuCell3D {


//    typedef std::vector<double>::size_type array_size_t;
    using array_size_t = size_t;

    template<typename T>
    class NumpyArrayWrapper3DImpl : public Field3D<T>, public NumpyArrayWrapperImpl<T>, public Field3DTypeBase {
    private:
        // needed to be able to return value by reference. And return by reference is needed to avoid SWIG wrapping issues
        const std::type_index typeIndex;
    protected:
        Dim3D dim;
        array_size_t padding;

    public:
        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */

        NumpyArrayWrapper3DImpl(const std::vector<array_size_t> &dims, array_size_t padding=0) :
                typeIndex(typeid(T)),
                NumpyArrayWrapperImpl<T>(dims, padding),
                padding(padding)
            {
            if (dims.size() != 3) {
                throw CC3DException("NumpyArrayWrapperImpl3D must have exactly 3 dimensions!!!");
            }

            if (dims[0] == 0 && dims[1] == 0 && dims[2] == 0)
                throw CC3DException("NumpyArrayWrapperImpl3D cannot have a 0 dimension!!!");

            if (dims[0] == 1 || dims[1] == 1 )
                throw CC3DException("NumpyArrayWrapperImpl3D does not support 2D arrays along xz or yz planes !!!");


            dim.x = this->dimensions[0];
            dim.y = this->dimensions[1];
            dim.z = this->dimensions[2];
        }

        virtual ~NumpyArrayWrapper3DImpl() = default;

        array_size_t getPadding(){return this->padding;}
        virtual std::vector<size_t> getPaddingVec(){return this->paddingVec;}

        void displayType() const override {
            std::cout << "VectorField of type: " << typeid(T).name() << "\n";
        }
        std::string getTypeString() const override {
            return typeid(T).name(); // Return typestring
        }

        const std::type_index & getType() const override {
            return typeIndex;
        }

        std::string getNumPyTypeString() const override{
            return getNumPyType(typeid(T));
        };

        std::string getElementType() {
            static const std::unordered_map<std::string, std::string> type_map = {
                    {"char", "int8"},
                    {"signed char", "int8"},
                    {"unsigned char", "uint8"},
                    {"short", "int16"},
                    {"unsigned short", "uint16"},
                    {"int", "int32"},
                    {"unsigned int", "uint32"},
                    {"long", "int64"},
                    {"unsigned long", "uint64"},
                    {"long long", "int64"},
                    {"unsigned long long", "uint64"},
                    {"float", "float32"},
                    {"double", "float64"}
            };

            std::string type_name =
                    std::is_same<T, char>::value ? "char" :
                    std::is_same<T, signed char>::value ? "signed char" :
                    std::is_same<T, unsigned char>::value ? "unsigned char" :
                    std::is_same<T, short>::value ? "short" :
                    std::is_same<T, unsigned short>::value ? "unsigned short" :
                    std::is_same<T, int>::value ? "int" :
                    std::is_same<T, unsigned int>::value ? "unsigned int" :
                    std::is_same<T, long>::value ? "long" :
                    std::is_same<T, unsigned long>::value ? "unsigned long" :
                    std::is_same<T, long long>::value ? "long long" :
                    std::is_same<T, unsigned long long>::value ? "unsigned long long" :
                    std::is_same<T, float>::value ? "float" :
                    std::is_same<T, double>::value ? "double" :
                    "unknown";

            auto it = type_map.find(type_name);
            return it != type_map.end() ? it->second : "unknown";
        }

//        std::string getElementType() const {
//            if (std::is_same<T, float>::value) {
//                return "float";
//            } else if (std::is_same<T, double>::value) {
//                return "double";
//            } else if (std::is_same<T, int>::value) {
//                return "int";
//            } else {
//                return "unknown";
//            }
//        }

        //Field 3D interface
        virtual void set(const Point3D &pt, const T value) {
            this->array[this->index({static_cast<size_t>(pt.x+this->paddingVec[0]), static_cast<size_t>(pt.y+this->paddingVec[1]), static_cast<size_t>(pt.z+this->paddingVec[2])})] = value;
        };

        virtual T get(const Point3D &pt) const {
            return this->array[this->index({static_cast<size_t>(pt.x+this->paddingVec[0]), static_cast<size_t>(pt.y+this->paddingVec[1]), static_cast<size_t>(pt.z+this->paddingVec[2])})];

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
