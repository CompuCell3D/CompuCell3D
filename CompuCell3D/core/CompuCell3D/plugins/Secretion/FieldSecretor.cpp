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





#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
#include <CompuCell3D/plugins/PixelTracker/BoundaryPixelTrackerPlugin.h>
#include <CompuCell3D/plugins/PixelTracker/BoundaryPixelTracker.h>


#include <iostream>
using namespace CompuCell3D;
using namespace std;


#include "FieldSecretor.h"

FieldSecretor::FieldSecretor():
concentrationFieldPtr(0),
boundaryPixelTrackerPlugin(0),
pixelTrackerPlugin(0),
boundaryStrategy(0),
maxNeighborIndex(0),
cellFieldG(0)
{}

FieldSecretor::~FieldSecretor()
{}

bool FieldSecretor::secreteInsideCell(CellG * _cell, float _amount){
	if (!pixelTrackerPlugin){
		return false;
	}
	BasicClassAccessor<PixelTracker> *pixelTrackerAccessorPtr=pixelTrackerPlugin->getPixelTrackerAccessorPtr();
	set<PixelTrackerData > & pixelSetRef=pixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;
	for (set<PixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){		

		concentrationFieldPtr->set(sitr->pixel,concentrationFieldPtr->get(sitr->pixel)+_amount);

	}

	return true;
}

bool FieldSecretor::secreteInsideCellAtBoundary(CellG * _cell, float _amount){

	if (!boundaryPixelTrackerPlugin){
		return false;
	}

	BasicClassAccessor<BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr=boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

	std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;


	for (set<BoundaryPixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){		

		concentrationFieldPtr->set(sitr->pixel,concentrationFieldPtr->get(sitr->pixel)+_amount);

	}

	return true;

}

bool FieldSecretor::secreteOutsideCellAtBoundary(CellG * _cell, float _amount){

	if (!boundaryPixelTrackerPlugin){
		return false;
	}

	BasicClassAccessor<BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr=boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

	std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

	Point3D nPt;

	CellG *nCell=0;

	Neighbor neighbor;

	set<FieldSecretorPixelData> visitedPixels;

	for (set<BoundaryPixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){		


		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(sitr->pixel),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}
			nPt=neighbor.pt;
			nCell = cellFieldG->get(neighbor.pt);
			if (nCell!=_cell && visitedPixels.find(FieldSecretorPixelData(neighbor.pt))==visitedPixels.end()){

				concentrationFieldPtr->set(nPt,concentrationFieldPtr->get(nPt)+_amount);
				visitedPixels.insert(FieldSecretorPixelData(nPt));

			}

		}		

	}

	return true;


}

bool FieldSecretor::secreteInsideCellAtCOM(CellG * _cell, float _amount){
	Point3D pt((int)round(_cell->xCM/_cell->volume),(int)round(_cell->yCM/_cell->volume),(int)round(_cell->zCM/_cell->volume));

	concentrationFieldPtr->set(pt,concentrationFieldPtr->get(pt)+_amount);
	return true;

}

bool FieldSecretor::uptakeInsideCell(CellG * _cell, float _maxUptake, float _relativeUptake){

	if (!boundaryPixelTrackerPlugin){
		return false;
	}

	BasicClassAccessor<BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr=boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

	std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

	float currentConcentration;
	for (set<BoundaryPixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){		
		currentConcentration=concentrationFieldPtr->get(sitr->pixel);
		if(currentConcentration*_relativeUptake>_maxUptake){
			concentrationFieldPtr->set(sitr->pixel,currentConcentration-_maxUptake);
		}else{
			concentrationFieldPtr->set(sitr->pixel,currentConcentration-currentConcentration*_relativeUptake);
		}				

	}



	return true;
}

bool FieldSecretor::uptakeInsideCellAtBoundary(CellG * _cell, float _maxUptake, float _relativeUptake){

	if (!boundaryPixelTrackerPlugin){
		return false;
	}

	BasicClassAccessor<BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr=boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

	std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

	float currentConcentration;
	for (set<BoundaryPixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){		

		currentConcentration=concentrationFieldPtr->get(sitr->pixel);
		if(currentConcentration*_relativeUptake>_maxUptake){
			concentrationFieldPtr->set(sitr->pixel,currentConcentration-_maxUptake);
		}else{
			concentrationFieldPtr->set(sitr->pixel,currentConcentration-currentConcentration*_relativeUptake);
		}				

	}

	return true;

}

bool FieldSecretor::uptakeOutsideCellAtBoundary(CellG * _cell, float _maxUptake, float _relativeUptake){

	if (!boundaryPixelTrackerPlugin){
		return false;
	}

	BasicClassAccessor<BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr=boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

	std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

	Point3D nPt;

	CellG *nCell=0;

	Neighbor neighbor;

	set<FieldSecretorPixelData> visitedPixels;

	float currentConcentration;

	for (set<BoundaryPixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){		


		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(sitr->pixel),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}
			nPt=neighbor.pt;
			nCell = cellFieldG->get(neighbor.pt);
			if (nCell!=_cell && visitedPixels.find(FieldSecretorPixelData(neighbor.pt))==visitedPixels.end()){

				currentConcentration=concentrationFieldPtr->get(nPt);
				if(currentConcentration*_relativeUptake>_maxUptake){
					concentrationFieldPtr->set(nPt,currentConcentration-_maxUptake);
				}else{
					concentrationFieldPtr->set(nPt,currentConcentration-currentConcentration*_relativeUptake);
				}				

				
				visitedPixels.insert(FieldSecretorPixelData(nPt));

			}

		}		

	}

	return true;

}

bool FieldSecretor::uptakeInsideCellAtCOM(CellG * _cell, float _maxUptake, float _relativeUptake){
	Point3D pt((int)round(_cell->xCM/_cell->volume),(int)round(_cell->yCM/_cell->volume),(int)round(_cell->zCM/_cell->volume));

	float currentConcentration=concentrationFieldPtr->get(pt);

	
	if(currentConcentration*_relativeUptake>_maxUptake){
		concentrationFieldPtr->set(pt,currentConcentration-_maxUptake);
	}else{
		concentrationFieldPtr->set(pt,currentConcentration-currentConcentration*_relativeUptake);
	}				

	return true;

}
