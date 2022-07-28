#ifndef POINT3D_H
#define POINT3D_H


#include <iostream>
#include <string>

namespace CompuCell3D {

    /**
     * A 3D point.
     *
     */
    class Point3D {
    public:
        short x;
        short y;
        short z;

        /**
         * Construct a point at the origin.
         */
        Point3D() : x(0), y(0), z(0) {}

        Point3D(const short x, const short y, const short z) :
                x(x), y(y), z(z) {}

        /**
         * Copy constructor
         */
        Point3D(const Point3D &pt) : x(pt.x), y(pt.y), z(pt.z) {}

        /**
         * Assignment operator.
         */
        Point3D &operator=(const Point3D pt);

        /**
         * Add the coordinates of pt to this Point3D.
         */
        Point3D &operator+=(const Point3D pt);

        /**
         * Subtract the coordinates of pt to this Point3D.
         */
        Point3D &operator-=(const Point3D pt);

        /// Comparison operator
        bool operator==(const Point3D pt) const { return (x == pt.x && y == pt.y && z == pt.z); }

        /// Not equal operator
        bool operator!=(const Point3D pt) const { return !(*this == pt); }

        bool operator<(const Point3D _rhs) const;

        friend std::ostream &operator<<(std::ostream &stream, const Point3D &pt);
    };

    inline Point3D &Point3D::operator=(const Point3D pt) {
        x = pt.x;
        y = pt.y;
        z = pt.z;
        return *this;
    }

    inline Point3D &Point3D::operator+=(const Point3D pt) {
        x += pt.x;
        y += pt.y;
        z += pt.z;
        return *this;
    }

    inline Point3D &Point3D::operator-=(const Point3D pt) {
        x -= pt.x;
        y -= pt.y;
        z -= pt.z;
        return *this;
    }

    inline bool Point3D::operator<(const Point3D _rhs) const {
        return x < _rhs.x || (!(_rhs.x < x) && y < _rhs.y)
               || (!(_rhs.x < x) && !(_rhs.y < y) && z < _rhs.z);
    }

    /**
     * Print a Point3D to a std::ostream.
     * The format is (x,y,z).
     */
    inline std::ostream &operator<<(std::ostream &stream, const Point3D &pt) {
        stream << '(' << pt.x << ',' << pt.y << ',' << pt.z << ')';
        return stream;
    }

    /**
     * Overloads the + operator for Point3D.
     */
    inline Point3D operator+(const Point3D pt1, const Point3D pt2) {
        return Point3D(pt1.x + pt2.x, pt1.y + pt2.y, pt1.z + pt2.z);
    }

    /**
     * Overloads the - operator for Point3D.
     */
    inline Point3D operator-(const Point3D pt1, const Point3D pt2) {
        return Point3D(pt1.x - pt2.x, pt1.y - pt2.y, pt1.z - pt2.z);
    }

    /**
     * Overloads the operator std::string + Point3D.
     */
    inline std::string operator+(const std::string s, const Point3D pt) {
        return s + "(" + std::to_string((int) pt.x) + "," + std::to_string((int) pt.y) + "," +
               std::to_string((int) pt.z) + ")";
    }
};
#endif
