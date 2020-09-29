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

#include "FocalPointPlasticityTracker.h"

using namespace CompuCell3D;

FocalPointPlasticityTrackerData::FocalPointPlasticityTrackerData(const FocalPointPlasticityLinkTrackerData& fppltd) {
	lambdaDistance = fppltd.lambdaDistance;
	targetDistance = fppltd.targetDistance;
	maxDistance = fppltd.maxDistance;
	activationEnergy = fppltd.activationEnergy;
	maxNumberOfJunctions = fppltd.maxNumberOfJunctions;
	neighborOrder = fppltd.neighborOrder;
	anchor = fppltd.anchor;
	anchorId = fppltd.anchorId;
	anchorPoint = fppltd.anchorPoint;
	initMCS = fppltd.initMCS;
}

FocalPointPlasticityTrackerData& FocalPointPlasticityTrackerData::operator=(const FocalPointPlasticityLinkTrackerData& fppltd) {
	return FocalPointPlasticityTrackerData(fppltd);
}

FocalPointPlasticityLinkTrackerData::FocalPointPlasticityLinkTrackerData(const FocalPointPlasticityTrackerData& fpptd)
{
	lambdaDistance = fpptd.lambdaDistance;
	targetDistance = fpptd.targetDistance;
	maxDistance = fpptd.maxDistance;
	activationEnergy = fpptd.activationEnergy;
	maxNumberOfJunctions = fpptd.maxNumberOfJunctions;
	neighborOrder = fpptd.neighborOrder;
	anchor = fpptd.anchor;
	anchorId = fpptd.anchorId;
	anchorPoint = fpptd.anchorPoint;
	initMCS = fpptd.initMCS;
}

FocalPointPlasticityLinkTrackerData& FocalPointPlasticityLinkTrackerData::operator=(const FocalPointPlasticityTrackerData& fpptd) {
	return FocalPointPlasticityLinkTrackerData(fpptd);
}