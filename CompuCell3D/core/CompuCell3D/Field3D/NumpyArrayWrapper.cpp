//
// Created by m on 10/5/24.
//

#include "NumpyArrayWrapper.h"
#include <iostream>
#include <numeric>  // For std::accumulate
#include <algorithm> // For std::copy
#include <iterator>

using namespace CompuCell3D;
using namespace std;

NumpyArrayWrapper::NumpyArrayWrapper(const std::vector<array_size_t> &dims) {

    this->dimensions = dims;
    strides = computeStrides(dimensions);
    std::copy(strides.begin(), strides.end(), std::ostream_iterator<int>(std::cerr, "\n"));

    size_t array_size = std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<array_size_t>());

    array.assign(array_size, 0.0);
//    cerr << "array array_size=" << array.size() << endl;
};


std::vector<array_size_t> NumpyArrayWrapper::computeStrides(const std::vector<array_size_t>& dims) {
    array_size_t n = dims.size();
    std::vector<array_size_t> strides_local(n, 1);  // Initialize strides_local with 1 for the last dimension

    for (int i = n - 2; i >= 0; --i) {
        strides_local[i] = strides_local[i + 1] * dims[i + 1];
    }

    return strides_local;
}


void NumpyArrayWrapper::printArrayValue(const std::vector<array_size_t> &indices)  {

    for (int idx: indices) {
        cout << idx << " ";  // Multiply each index by the member variable
    }
    cout<<array[index(indices)]<<" ";
    cout << endl;
}


void NumpyArrayWrapper::iterateOverAxes(const std::vector<array_size_t> &dims,
                                        std::function<void(const std::vector<array_size_t> &)> functor) {
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

void NumpyArrayWrapper::printAllArrayValues(){

    auto lambdaFunctor = [this](const std::vector<array_size_t>& indices) {
        // Inside the lambda, call another member function
        this->printArrayValue(indices);
    };

    // Call iterateOverAxes with the lambda functor
    std::cout << "\nUsing a lambda with iterateOverAxes as a member function:\n";
    this->iterateOverAxes(dimensions, lambdaFunctor);

}

