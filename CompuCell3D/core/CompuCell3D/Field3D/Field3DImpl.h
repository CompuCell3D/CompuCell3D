#ifndef FIELD3DIMPL_H
#define FIELD3DIMPL_H

#include <math.h>
#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/CC3DExceptions.h>

#include "Dim3D.h"
#include "Field3D.h"


namespace CompuCell3D {
//indexing macro
#define PT2IDX(pt) (pt.x + ((pt.y + (pt.z * dim.y)) * dim.x))

    template<class T>
    class Field3D;

    /**
     * Default implementation of the Field3D interface.
     *
     */
    template<class T>
    class Field3DImpl : public Field3D<T> {
    protected:
        Dim3D dim;
        T *field;
        T initialValue;
        size_t len=0;
    public:

        /**
         * @param dim The field dimensions
         * @param initialValue The initial value of all data elements in the field.
         */
        Field3DImpl(const Dim3D dim, const T &initialValue) :
                dim(dim), field(0), initialValue(initialValue) {

            if (dim.x == 0 && dim.y == 0 && dim.z == 0) throw CC3DException("Field3D cannot have a 0 dimension!!!");

            // Check that the dimensions are not too large.
            if (log((double) dim.x) / log(2.0) + log((double) dim.y) / log(2.0) + log((double) dim.z) / log(2.0) >
                sizeof(int) * 8)
                throw CC3DException("Field3D dimensions too large!!!");

            // Allocate and initialize the field
            std::size_t len =
                static_cast<std::size_t>(dim.x) *
                static_cast<std::size_t>(dim.y) *
                static_cast<std::size_t>(dim.z);
            field = new T[len];
            for (size_t i = 0; i < len; i++)
                field[i] = initialValue;
        }

        ~Field3DImpl() override {
            if (field) {
                delete[] field;
                field = 0;
            }

        }

        void set(const Point3D &pt, const T value) override {
            if (!isValid(pt)) throw CC3DException("set() point out of range!");
            field[PT2IDX(pt)] = value;
        }

        void resizeAndShift(const Dim3D theDim, Dim3D shiftVec) override {

            const std::size_t nx = static_cast<std::size_t>(theDim.x);
            const std::size_t ny = static_cast<std::size_t>(theDim.y);
            const std::size_t nz = static_cast<std::size_t>(theDim.z);

            std::size_t len = nx * ny * nz;

            T* field2 = new T[len];

            // initialize
            for (std::size_t i = 0; i < len; ++i)
                field2[i] = initialValue;

            // copy old field
            for (std::size_t x = 0; x < nx; ++x)
                for (std::size_t y = 0; y < ny; ++y)
                    for (std::size_t z = 0; z < nz; ++z) {

                        // compute old coordinates in signed space
                        const int ox = static_cast<int>(x) - shiftVec.x;
                        const int oy = static_cast<int>(y) - shiftVec.y;
                        const int oz = static_cast<int>(z) - shiftVec.z;

                        if (ox >= 0 && ox < dim.x &&
                            oy >= 0 && oy < dim.y &&
                            oz >= 0 && oz < dim.z) {

                            field2[
                                x + (y + z * ny) * nx
                            ] = getQuick(Point3D(ox, oy, oz));
                        }
                    }

            delete[] field;
            field = field2;
            dim = theDim;

            BoundaryStrategy::getInstance()->setDim(dim);
        }




        void setDim(const Dim3D theDim) override{
            this->resizeAndShift(theDim, Dim3D());
        }

        T getQuick(const Point3D &pt) const {

            //return field[PT2IDX(pt)];
            return (isValid(pt) ? field[PT2IDX(pt)] : initialValue);
        }

        void setQuick(const Point3D &pt, const T _value) {

            field[PT2IDX(pt)] = _value;
        }


        T get(const Point3D &pt) const override{

            return (isValid(pt) ? field[PT2IDX(pt)] : initialValue);
        }

        T getByIndex(long _offset) const override{
            if (_offset < 0) {
                return initialValue;
            }

            const auto idx = static_cast<std::size_t>(_offset);
            return (idx < len) ? field[idx] : initialValue;
        }

        void setByIndex(long _offset, const T _value) override {
            if (_offset < 0) {
                return;
            }

            const auto idx = static_cast<std::size_t>(_offset);
            if (idx < len) {
                field[idx] = _value;
            }
        }


        Dim3D getDim() const override { return dim; }

        bool isValid(const Point3D &pt) const override {
            return (0 <= pt.x && pt.x < dim.x &&
                    0 <= pt.y && pt.y < dim.y &&
                    0 <= pt.z && pt.z < dim.z);
        }
    };
};
#endif
