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

#ifndef _MABOSS_CONFIG_H
#define _MABOSS_CONFIG_H

#define HAS_UNORDERED_MAP
#include <unordered_map>
#define STATE_MAP std::unordered_map
#define HASH_STRUCT hash

#ifndef MAXNODES
#define MAXNODES 64
#endif

#ifdef _WIN32

#ifndef _SC_CLK_TCK
#define _SC_CLK_TCK 2

#include <exception>
#include <iostream>
#include <time.h>

long sysconf(int _name);

#endif // _SC_CLK_TCK

#endif // _WIN32

#endif // _MABOSS_CONFIG_H