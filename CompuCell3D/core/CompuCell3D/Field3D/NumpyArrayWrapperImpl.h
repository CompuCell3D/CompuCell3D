//
// Created by m on 10/5/24.
//

#ifndef COMPUCELL3D_NUMPYARRAYWRAPPERIMPL_H
#define COMPUCELL3D_NUMPYARRAYWRAPPERIMPL_H

#include <vector>
#include <initializer_list>
#include <functional>  // For std::function



#include <vector>
#include <initializer_list>
#include <functional>  // For std::function
#include <numeric>  // For std::accumulate
#include <algorithm> // For std::copy
#include <iostream>
#include <iterator> // for ostream_iterator


namespace CompuCell3D {

    using array_size_t = size_t;
//    typedef std::vector<double>::size_type array_size_t;

    template <typename T>
    class NumpyArrayWrapperImpl{


    protected:
        //includes padding
        std::vector<array_size_t> dimensions;
        std::vector<array_size_t> strides;
        std::vector<T> array;
        // "user-visible" dimensions
        std::vector<array_size_t> internalDimensions;

        std::vector<array_size_t> paddingVec;

    public:
        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */


        NumpyArrayWrapperImpl(const std::vector<array_size_t> &dims, array_size_t padding=0)
        {
            this->dimensions = dims;
            // creating padding vec - initially 0
            this->paddingVec.assign(dims.size(), padding);

            // add padding
            for ( size_t i=0 ; i< this->dimensions.size() ; ++i) {
                if (this->dimensions[i] == 1){
                    // we set padding to be 0 along any dimension of length 1
                    // we only pad along "larger" dimensions
                    paddingVec[i] = 0;
                }
                dimensions[i] += 2*paddingVec[i];
            }

            strides = computeStrides(dimensions);
            std::copy(strides.begin(), strides.end(), std::ostream_iterator<int>(std::cerr, "\n"));

            size_t array_size = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<array_size_t>());

            array.assign(array_size, 0.0);
            std::cerr << "array array_size=" << array.size() << std::endl;

        }

        virtual ~NumpyArrayWrapperImpl() = default;

        void setDimensions(const std::vector<array_size_t> &_dimensions) {
            this->dimensions = _dimensions;
        }
        std::vector<array_size_t> computeStrides(const std::vector<array_size_t>& dims){
            array_size_t n = dims.size();
            std::vector<array_size_t> strides_local(n, 1);  // Initialize strides_local with 1 for the last dimension

            for (int i = n - 2; i >= 0; --i) {
                strides_local[i] = strides_local[i + 1] * dims[i + 1];
            }

            return strides_local;

        }


        void iterateOverAxes(const std::vector<array_size_t> &dims,
                             std::function<void(const std::vector<array_size_t> &)> functor){

            if (dims.empty()) {
                return;  // No dimensions to iterate over
            }

            std::vector<array_size_t> indices(dims.size(), 0);  // Initialize indices for all axes to 0

            while (true) {
                // Process the current combination of indices

                functor(indices);

                // Increment the indices starting from the last axis (rightmost dimension)
                int axis = dims.size() - 1;
                while (axis >= 0) {
                    if (++indices[axis] < dims[axis]) {
                        // If incrementing this axis doesn't overflow, break out of the loop
                        break;
                    } else {
                        // Reset this axis and carry over to the next axis
                        indices[axis] = 0;
                        axis--;
                    }
                }

                // If all axes are done (i.e., axis < 0), we are finished
                if (axis < 0) {
                    break;
                }
            }

        }

        void printArrayValue(const std::vector<array_size_t> &indices){
            for (int idx: indices) {
                std::cout << idx << " ";  // Multiply each index by the member variable
            }
            std::cout<<array[index(indices)]<<" ";
            std::cout << std::endl;
        }

        array_size_t index(const std::vector<array_size_t>& indices) const {
            array_size_t index = 0;
            for (size_t i = 0; i < indices.size(); ++i) {
                index += indices[i] * strides[i];
            }
            return index;
        }

        size_t getSize() {
//            std::cerr<<"INTERNAL GET SIZE"<<array.size()<<std::endl;
            return array.size();

        }

        T *getPtr() {
            return array.size() ? &array[0] : nullptr;
        }

        void printAllArrayValues(){

            auto lambdaFunctor = [this](const std::vector<array_size_t>& indices) {
                // Inside the lambda, call another member function
                this->printArrayValue(indices);
            };

            // Call iterateOverAxes with the lambda functor
            std::cout << "\nUsing a lambda with iterateOverAxes as a member function:\n";
            this->iterateOverAxes(dimensions, lambdaFunctor);
        }




    };
};


#endif
