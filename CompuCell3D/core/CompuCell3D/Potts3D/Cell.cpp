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

#include "Cell.h"

using namespace CompuCell3D;

CellG::CellG() :
	volume(0),
	targetVolume(0.0),
	lambdaVolume(0.0),
	surface(0),
	targetSurface(0.0),
	lambdaSurface(0.0),
	clusterSurface(0.0),
	targetClusterSurface(0.0),
	lambdaClusterSurface(0.0),
	type(0),
	xCM(0), yCM(0), zCM(0),
	xCOM(0), yCOM(0), zCOM(0),
	xCOMPrev(0), yCOMPrev(0), zCOMPrev(0),
	iXX(0), iXY(0), iXZ(0), iYY(0), iYZ(0), iZZ(0),
	lX(0.0),
	lY(0.0),
	lZ(0.0),
	lambdaVecX(0.0),
	lambdaVecY(0.0),
	lambdaVecZ(0.0),
	flag(0),
	id(0),
	clusterId(0),
	fluctAmpl(-1.0),
	lambdaMotility(0.0),
	biasVecX(1.0),
	biasVecY(0.0),
	biasVecZ(0.0),
	connectivityOn(false),
	extraAttribPtr(0),
	pyAttrib(0)
	//test_biasV


{
	DerivedProperty<CellG, float, &CellG::getPressure> pressure(this);
	DerivedProperty<CellG, float, &CellG::getSurfaceTension> surfaceTension(this);
	DerivedProperty<CellG, float, &CellG::getClusterSurfaceTension> clusterSurfaceTension(this);
}

float CellG::getPressure() { return 2.0 * lambdaVolume * (volume - targetVolume); }

float CellG::getSurfaceTension() { return 2.0 * lambdaSurface * (surface - targetSurface); }

float CellG::getClusterSurfaceTension() { return 2.0 * lambdaClusterSurface * (clusterSurface - targetClusterSurface); }
