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

#include "Dim3D.h"
using namespace CompuCell3D;

Dim3D::Dim3D() : Point3D() {}

Dim3D::Dim3D(const short x, const short y, const short z) : Point3D(x, y, z) {}

Dim3D::Dim3D(const Dim3D &dim) : Point3D(dim) {}  

Dim3D &Dim3D::operator=(const Dim3D pt) {
    x = pt.x;
    y = pt.y;
    z = pt.z;
    return *this;
}

Dim3D &Dim3D::operator+=(const Dim3D pt) {
    x += pt.x;
    y += pt.y;
    z += pt.z;
    return *this;
}

Dim3D &Dim3D::operator-=(const Dim3D pt) {
    x -= pt.x;
    y -= pt.y;
    z -= pt.z;
    return *this;
}
    
bool Dim3D::operator==(const Dim3D pt) const {
    return (x == pt.x && y == pt.y && z == pt.z);
}

bool Dim3D::operator!=(const Dim3D pt) const {
    
    return !(*this==pt);
}

bool Dim3D::operator<(const Dim3D  _rhs) const{
    return x < _rhs.x || (!(_rhs.x < x)&& y < _rhs.y)
        ||(!(_rhs.x < x)&& !(_rhs.y <y )&& z < _rhs.z);
}
short & Dim3D::operator[](int _idx){
    if(!_idx){
        return x;
    }else if(_idx==1){
        return y;
    }else { //there is no error checking here so in case user picks index out of range we return z coordinate
        return z;
    }
}
