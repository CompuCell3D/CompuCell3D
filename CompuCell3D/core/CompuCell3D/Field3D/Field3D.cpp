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

#include "Field3D.h"
using namespace CompuCell3D;
template<>
const char Field3D<int>::typeStr[3] = " i";
template<>
const char Field3D<unsigned int>::typeStr[3] = "ui";
template<>
const char Field3D<double>::typeStr[3] = " d";
template<>
const char Field3D<float>::typeStr[3] = " f";
template<>
const char Field3D<char>::typeStr[3] = " c";
template<>
const char Field3D<unsigned char>::typeStr[3] = "uc";
template<>
const char Field3D<long>::typeStr[3] = " l";
template<>
const char Field3D<unsigned long>::typeStr[3] = "ul";
