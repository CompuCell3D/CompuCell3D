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

#ifndef DIM3D_H
#define DIM3D_H

#include "Point3D.h"

//#include <XMLCereal/XMLSerializable.h>

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

    Dim3D &operator=(const Dim3D pt) {
      x = pt.x;
      y = pt.y;
      z = pt.z;
      return *this;
    }

    /** 
     * Add the coordinates of pt to this Point3D.
     */
    Dim3D &operator+=(const Dim3D pt) {
      x += pt.x;
      y += pt.y;
      z += pt.z;
      return *this;
    }

    /** 
     * Subtract the coordinates of pt to this Point3D.
     */
    Dim3D &operator-=(const Dim3D pt) {
      x -= pt.x;
      y -= pt.y;
      z -= pt.z;
      return *this;
    }
        
    /// Comparison operator
    bool operator==(const Dim3D pt) const {
      return (x == pt.x && y == pt.y && z == pt.z);
    }
    /// Not equal operator
    bool operator!=(const Dim3D pt) const {
		
      return !(*this==pt);
    }
    
   bool operator<(const Dim3D  _rhs) const{
      return x < _rhs.x || (!(_rhs.x < x)&& y < _rhs.y)
			||(!(_rhs.x < x)&& !(_rhs.y <y )&& z < _rhs.z);
   }
   short & operator[](int _idx){
	   if(!_idx){
			return x;
	   }else if(_idx==1){
			return y;
	   }else { //there is no error checking here so in case user picks index out of range we return z coordinate
			return z;
	   }
   }

   friend std::ostream &operator<<(std::ostream &stream, const Dim3D &pt);
  };

    inline std::ostream &operator<<(std::ostream &stream, const Dim3D &pt) {
    stream << '(' << pt.x << ',' << pt.y << ',' << pt.z << ')';
    return stream;
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
    return s + "(" + BasicString(pt.x) + "," + BasicString(pt.y) + "," +
      BasicString(pt.z) + ")";
  }

};
#endif
