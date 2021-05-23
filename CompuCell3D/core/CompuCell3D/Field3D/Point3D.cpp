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

#include "Point3D.h"
using namespace CompuCell3D;

#include <BasicUtils/BasicString.h>

Point3D::Point3D() : x(0), y(0), z(0) {}

Point3D::Point3D(const short x, const short y, const short z) :
    x(x), y(y), z(z) {}

Point3D::Point3D(const Point3D &pt) : x(pt.x), y(pt.y), z(pt.z) {}

Point3D &Point3D::operator=(const Point3D pt) {
    x = pt.x;
    y = pt.y;
    z = pt.z;
    return *this;
}

Point3D &Point3D::operator+=(const Point3D pt) {
    x += pt.x;
    y += pt.y;
    z += pt.z;
    return *this;
}

Point3D &Point3D::operator-=(const Point3D pt) {
    x -= pt.x;
    y -= pt.y;
    z -= pt.z;
    return *this;
}

bool Point3D::operator==(const Point3D pt) const {
    return (x == pt.x && y == pt.y && z == pt.z);
}

bool Point3D::operator!=(const Point3D pt) const {
    
    return !(*this==pt);
}
    
bool Point3D::operator<(const Point3D  _rhs) const{
    return x < _rhs.x || (!(_rhs.x < x)&& y < _rhs.y)
        ||(!(_rhs.x < x)&& !(_rhs.y <y )&& z < _rhs.z);
}
