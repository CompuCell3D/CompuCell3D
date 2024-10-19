//
// Created by m on 10/5/24.
//

#ifndef COMPUCELL3D_NUMPYARRAYWRAPPER_H
#define COMPUCELL3D_NUMPYARRAYWRAPPER_H

#include <vector>
#include <initializer_list>


namespace CompuCell3D {


    typedef std::vector<double>::size_type array_size_t;

    class NumpyArrayWrapper {

    public:
//        typedef std::vector<double>::size_type array_size_t;
    private:

        std::vector<array_size_t> dimensions;
        std::vector<double> array;


        array_size_t dim_x;
        array_size_t dim_y;

    public:
        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */

        NumpyArrayWrapper(array_size_t dim_x,array_size_t dim_y);

//        // Variadic template function to take variable number of dimensions
//        template<typename... Args>
//        void setDimensions(Args... args) {
//            dimensions = {args...};  // Initialize the vector with the provided arguments
//
//        }
        void setDimensions(const std::vector<array_size_t>& _dimensions){
            this->dimensions = _dimensions;
        }

//        const std::vector<unsigned int>& getDimensions() const {
//            return dimensions;
//        }

        array_size_t getSize(){
            return array.size();

        }
        double *getPtr() {
            return array.size() ? &array[0] : nullptr;
        }

        void printArray();

        virtual ~NumpyArrayWrapper() {}


    };
};


#endif
