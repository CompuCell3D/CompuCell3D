#include <CompuCell3D/CC3D.h>



// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/plugins/PixelTracker/PixelTrackerPlugin.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTrackerPlugin.h>
#include <CompuCell3D/plugins/BoundaryPixelTracker/BoundaryPixelTracker.h>



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

bool FieldSecretor::secreteInsideCellAtBoundaryOnContactWith(CellG * _cell, float _amount,const std::vector<unsigned char> & _onContactVec){
	
	set<unsigned char> onContactSet(_onContactVec.begin(),_onContactVec.end());

	if (!boundaryPixelTrackerPlugin){
		return false;
	}

	BasicClassAccessor<BoundaryPixelTracker> *boundaryPixelTrackerAccessorPtr=boundaryPixelTrackerPlugin->getBoundaryPixelTrackerAccessorPtr();

	std::set<BoundaryPixelTrackerData > & pixelSetRef=boundaryPixelTrackerAccessorPtr->get(_cell->extraAttribPtr)->pixelSet;

	Point3D nPt;

	CellG *nCell=0;

	Neighbor neighbor;

	for (set<BoundaryPixelTrackerData>::iterator sitr=pixelSetRef.begin() ; sitr!=pixelSetRef.end(); ++sitr){	
		
		for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
			neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(sitr->pixel),nIdx);
			if(!neighbor.distance){
				//if distance is 0 then the neighbor returned is invalid
				continue;
			}

			nPt=neighbor.pt;
			nCell = cellFieldG->get(neighbor.pt);
			if(nCell!=_cell && !nCell && onContactSet.find(0)!=onContactSet.end()){
				//user requested secrete on contact with medium and we found medium pixel
				concentrationFieldPtr->set(sitr->pixel,concentrationFieldPtr->get(sitr->pixel)+_amount);
				break; //after secreting do not try to secrete more
			}
			
			if (nCell!=_cell && nCell && onContactSet.find(nCell->type)!=onContactSet.end() )
			{
				//user requested secretion on contact with cell type whose pixel we have just found 
				concentrationFieldPtr->set(sitr->pixel,concentrationFieldPtr->get(sitr->pixel)+_amount);
				break;//after secreting do not try to secrete more
			}

		}

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


bool FieldSecretor::secreteOutsideCellAtBoundaryOnContactWith(CellG * _cell, float _amount, const std::vector<unsigned char> & _onContactVec){

	set<unsigned char> onContactSet(_onContactVec.begin(),_onContactVec.end());

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
				if (!nCell && onContactSet.find(0)!=onContactSet.end()){
					//checking if the unvisited pixel belongs to Medium and if Medium is a  cell type listed in the onContactSet
					concentrationFieldPtr->set(nPt,concentrationFieldPtr->get(nPt)+_amount);
					visitedPixels.insert(FieldSecretorPixelData(nPt));
				}

				if (nCell && onContactSet.find(nCell->type)!=onContactSet.end()){
					//checking if the unvisited pixel belongs to a  cell type listed in the onContactSet
					concentrationFieldPtr->set(nPt,concentrationFieldPtr->get(nPt)+_amount);
					visitedPixels.insert(FieldSecretorPixelData(nPt));
				}

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
