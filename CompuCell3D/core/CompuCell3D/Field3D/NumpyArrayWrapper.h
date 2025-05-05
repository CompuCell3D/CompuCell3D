//
// Created by m on 10/5/24.
//

#ifndef COMPUCELL3D_NUMPYARRAYWRAPPER_H
#define COMPUCELL3D_NUMPYARRAYWRAPPER_H

#include <vector>
#include <initializer_list>
#include <functional>  // For std::function
#include <iostream>


#include <vector>
#include <initializer_list>
#include <functional>  // For std::function


namespace CompuCell3D {


    typedef std::vector<std::size_t>::size_type array_size_t;

    class NumpyArrayWrapper {


    private:

        std::vector<array_size_t> dimensions;
        std::vector<array_size_t> strides;
        std::vector<double> array;


    public:
        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */


        NumpyArrayWrapper(const std::vector<array_size_t> &dims);


        void setDimensions(const std::vector<array_size_t> &_dimensions) {
            this->dimensions = _dimensions;
        }
        std::vector<array_size_t> computeStrides(const std::vector<array_size_t>& dims);


        void iterateOverAxes(const std::vector<array_size_t> &dims,
                             std::function<void(const std::vector<array_size_t> &)> functor);

        void printArrayValue(const std::vector<array_size_t> &indices);

        array_size_t index(const std::vector<array_size_t>& indices) const {
            array_size_t index = 0;
            for (std::size_t i = 0; i < indices.size(); ++i) {
                index += indices[i] * strides[i];
            }
            return index;
        }

        size_t getSize() {
//            std::cerr<<"INTERNAL GET SIZE"<<array.size()<<std::endl;
            return array.size();

        }
//        array_size_t getSize() {
//            return array.size();
//
//        }

        double *getPtr() {
            return array.size() ? &array[0] : nullptr;
        }

        void printAllArrayValues();

        virtual ~NumpyArrayWrapper() {}


    };
};


#endif
