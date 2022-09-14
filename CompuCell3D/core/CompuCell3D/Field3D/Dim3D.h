#ifndef DIM3D_H
#define DIM3D_H

#include "Point3D.h"

#include <string>

namespace CompuCell3D {

    /**
     * A 3D dimension.
     */
    class Dim3D : public Point3D {
    public:
        /// Construct a Dim3D with dimensions (0,0,0).
        Dim3D() : Point3D() {}

        Dim3D(const short x, const short y, const short z) : Point3D(x, y, z) {}

        /// Copy constructor
        Dim3D(const Dim3D &dim) : Point3D(dim) {}

        Dim3D &operator=(const Dim3D pt);

        /**
         * Add the coordinates of pt to this Point3D.
         */
        Dim3D &operator+=(const Dim3D pt);

        /**
         * Subtract the coordinates of pt to this Point3D.
         */
        Dim3D &operator-=(const Dim3D pt);

        /// Comparison operator
        bool operator==(const Dim3D pt) const { return (x == pt.x && y == pt.y && z == pt.z); }

        /// Not equal operator
        bool operator!=(const Dim3D pt) const { return !(*this == pt); }

        bool operator<(const Dim3D _rhs) const;

        short &operator[](int _idx);

        friend std::ostream &operator<<(std::ostream &stream, const Dim3D &pt);
    };

    inline Dim3D &Dim3D::operator=(const Dim3D pt) {
        x = pt.x;
        y = pt.y;
        z = pt.z;
        return *this;
    }

    inline Dim3D &Dim3D::operator+=(const Dim3D pt) {
        x += pt.x;
        y += pt.y;
        z += pt.z;
        return *this;
    }

    inline Dim3D &Dim3D::operator-=(const Dim3D pt) {
        x -= pt.x;
        y -= pt.y;
        z -= pt.z;
        return *this;
    }

    inline bool Dim3D::operator<(const Dim3D _rhs) const {
        return x < _rhs.x || (!(_rhs.x < x) && y < _rhs.y)
               || (!(_rhs.x < x) && !(_rhs.y < y) && z < _rhs.z);
    }

    inline std::ostream &operator<<(std::ostream &stream, const Dim3D &pt) {
        stream << '(' << pt.x << ',' << pt.y << ',' << pt.z << ')';
        return stream;
    }

    inline short &Dim3D::operator[](int _idx) {
        if (!_idx) {
            return x;
        } else if (_idx == 1) {
            return y;
        } else { //there is no error checking here so in case user picks index out of range we return z coordinate
            return z;
        }
    }

    /**
     * Overloads the + operator for Dim3D.
     */
    inline Dim3D operator+(const Dim3D pt1, const Dim3D pt2) {
        return Dim3D(pt1.x + pt2.x, pt1.y + pt2.y, pt1.z + pt2.z);
    }

    /**
     * Overloads the - operator for Point3D.
     */
    inline Dim3D operator-(const Dim3D pt1, const Dim3D pt2) {
        return Dim3D(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
    }

    /**
     * Overloads the operator std::string + Point3D.
     */
    inline std::string operator+(const std::string s, const Dim3D pt) {
        return s + "(" + std::to_string((int) pt.x) + "," + std::to_string((int) pt.y) + "," +
               std::to_string((int) pt.z) + ")";
    }

};
#endif
