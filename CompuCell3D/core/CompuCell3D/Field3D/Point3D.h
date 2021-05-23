/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef POINT3D_H
#define POINT3D_H

#include <BasicUtils/BasicString.h>

#include <iostream>

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
    Point3D();

    Point3D(const short x, const short y, const short z);

    /** 
     * Copy constructor
     */
    Point3D(const Point3D &pt);

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
    bool operator==(const Point3D pt) const;

    /// Not equal operator
    bool operator!=(const Point3D pt) const;
    
    bool operator<(const Point3D _rhs) const;
    
    friend std::ostream &operator<<(std::ostream &stream, const Point3D &pt);

    // Python support

#ifdef SWIGPYTHON

    PyObject* to_tuple() { return PyTuple_Pack(3, PyLong_FromLong(x), PyLong_FromLong(y), PyLong_FromLong(z)); }

#endif // SWIGPYTHON
  };

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
    return s + "(" + BasicString(pt.x) + "," + BasicString(pt.y) + "," +
      BasicString(pt.z) + ")";
  }
};
#endif
