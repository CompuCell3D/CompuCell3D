#include "CellBoundaryTrackerPlugin.h"
#include <CompuCell3D/Simulator.h>
#include <CompuCell3D/Potts3D/Potts3D.h>
#include <CompuCell3D/Field3D/AdjacentNeighbor.h>
#include <CompuCell3D/plugins/Volume/VolumePlugin.h>
#include <CompuCell3D/plugins/Volume/VolumeEnergy.h>

using namespace CompuCell3D;

#include <XMLCereal/XMLPullParser.h>
#include <XMLCereal/XMLSerializer.h>

#include <iostream>
#include <cmath>
#include <Logger/CC3DLogger.h>

using namespace std;

CellBoundaryTrackerPlugin::CellBoundaryTrackerPlugin() :
        cellFieldG(0),
        periodicX(false),
        periodicY(false),
        periodicZ(false) {}

CellBoundaryTrackerPlugin::~CellBoundaryTrackerPlugin() {

}

void CellBoundaryTrackerPlugin::init(Simulator *_simulator) {
  CC3D_Log(LOG_DEBUG) << "INITIALIZING CELL BOUNDARYTRACKER PLUGIN";
    simulator = _simulator;
    Potts3D *potts = simulator->getPotts();
    cellFieldG = potts->getCellFieldG();


    ///will register CellBoundaryTracker here
    BasicClassAccessorBase *cellBTAPtr = &cellBoundaryTrackerAccessor;
    ///************************************************************************************************
    ///REMARK. HAVE TO USE THE SAME BASIC CLASS ACCESSOR INSTANCE THAT WAS USED TO REGISTER WITH FACTORY
    ///************************************************************************************************
    potts->getCellFactoryGroupPtr()->registerClass(cellBTAPtr);
ergyFunction(surfaceEnergy);

    potts->registerCellGChangeWatcher(this);

    fieldDim = cellFieldG->getDim();

    adjNeighbor = AdjacentNeighbor(fieldDim);
    if (potts->getBoundaryXName() == "Periodic") {
        adjNeighbor.setPeriodicX();
        periodicX = true;
    }
    if (potts->getBoundaryYName() == "Periodic") {
        adjNeighbor.setPeriodicY();
        periodicY = true;
    }
    if (potts->getBoundaryZName() == "Periodic") {
        adjNeighbor.setPeriodicZ();
        periodicZ = true;
    }


    maxIndex = adjNeighbor.getField3DIndex().getMaxIndex();


}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CellBoundaryTrackerPlugin::field3DChange(const Point3D &pt, CellG *newCell,
                                              CellG *oldCell) {



    if (newCell ==
        oldCell) {//happens during multiple calls to se fcn on the same pixel woth current cell - should be avoided
        return;
    }

    vector <Point3D> const &adjNeighborOffsetsVec_inn = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
    unsigned int neighborSize = adjNeighborOffsetsVec_inn.size();

    const vector <Point3D> &adjFace2FaceNeighborOffsetsVec_inn = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
    unsigned int face2FaceNeighborSize = adjFace2FaceNeighborOffsetsVec_inn.size();


    const Field3DIndex &field3DIndex = adjNeighbor.getField3DIndex();

    long currentPtIndex = 0;
    long adjNeighborIndex = 0;
    long adjFace2FaceNeighborIndex = 0;

    CellG *currentCellPtr = 0;
    CellG *adjCellPtr = 0;

    unsigned int token = 0;
    double distance;
    int oldDiff = 0;
    int newDiff = 0;
    Point3D n;
    Point3D ptAdj;
    CellG *nCell = 0;


    set <BoundaryData> *oldCellBoundarySetPtr = 0;
    set <BoundaryData> *newCellBoundarySetPtr = 0;
    set<BoundaryData>::iterator BdSitr;
    set<BoundaryData>::iterator BdSitrN;
    set<BoundaryData>::iterator InsSitr;///position of the element after insertion
    pair<set<BoundaryData>::iterator, bool> InsSitrOKPair;

    set <NeighborSurfaceData> *oldCellNeighborSurfaceDataSetPtr = 0;
    set <NeighborSurfaceData> *newCellNeighborSurfaceDataSetPtr = 0;
    pair<set<NeighborSurfaceData>::iterator, bool> set_NSD_itr_OK_Pair;
    set<NeighborSurfaceData>::iterator set_NSD_itr;


    if (newCell) {
        newCellBoundarySetPtr = &cellBoundaryTrackerAccessor.get(newCell->extraAttribPtr)->boundary;
        newCellNeighborSurfaceDataSetPtr = &cellBoundaryTrackerAccessor.get(newCell->extraAttribPtr)->cellNeighbors;
    }

    if (oldCell) {
        oldCellBoundarySetPtr = &cellBoundaryTrackerAccessor.get(oldCell->extraAttribPtr)->boundary;
        oldCellNeighborSurfaceDataSetPtr = &cellBoundaryTrackerAccessor.get(oldCell->extraAttribPtr)->cellNeighbors;
    }


    currentPtIndex = field3DIndex.index(pt);
    currentCellPtr = cellFieldG->getByIndex(currentPtIndex);

    if (oldCell) {
        ///When there was a cell at this pixel we will have to update all its neighbours (update counters) and remove it from the boundary
//       currentPtIndex=field3DIndex.index(pt);
//       currentCellPtr=cellFieldG->getByIndex(currentPtIndex);
      CC3D_Log(LOG_TRACE) << "currentPtIndex="<<currentPtIndex;
        BdSitr = oldCellBoundarySetPtr->find(BoundaryData(currentPtIndex));

        if (BdSitr != oldCellBoundarySetPtr->end()) {
            //Now will have to erase current Point from the boundary

            oldCellBoundarySetPtr->erase(BdSitr);

///RESTORE IT LATER! it is important
/*         if(BdSitr->numberOfForeignNeighbors>26){
             CC3D_Log(LOG_TRACE) << "BUG, there can be no more that 26 foreign Neighbors";
            exit(0);
         }*/
      }/*else{
         CC3D_Log(LOG_TRACE) << " PROBLEM: old cell does not belong to the boundary";
         CC3D_Log(LOG_TRACE) << "pt="<<pt;
         CC3D_Log(LOG_TRACE) << "old="<<oldCell<<" new="<<newCell;
         CC3D_Log(LOG_TRACE) << "flipNeighbor="<<simulator->getPotts()->getFlipNeighbor();
         CC3D_Log(LOG_TRACE) << "face2FaceNeighborSize="<<face2FaceNeighborSize<<" neighborSize="<<neighborSize;
         exit(0);
      }*/

        //BasicClassGroup *adjCellPtrAlt;
        for (int i = 0; i < neighborSize; ++i) {
            ptAdj = pt;
            ptAdj += adjNeighborOffsetsVec_inn[i];

            adjNeighborIndex = field3DIndex.index(ptAdj);
            //adjNeighborIndex=currentPtIndex+adjNeighborOffsetsVec[i];

            //if(!(adjNeighborIndex<0 || adjNeighborIndex > maxIndex ) ){
            if (cellFieldG->isValid(ptAdj)) {


                adjCellPtr = cellFieldG->get(ptAdj);
                //adjCellPtr=cellFieldG->get(ptAdj);

                if (adjCellPtr == oldCell) {///modifying boundary data for old cell

                    BdSitr = oldCellBoundarySetPtr->find(
                            BoundaryData(adjNeighborIndex)); ///check if a neighbor belongs to boundary of old cell
                    if (BdSitr != oldCellBoundarySetPtr->end()) {///if it does
                        BdSitr->incrementNumberOfForeignNeighbors(*BdSitr);
                    } else { ///if it does not belong to the boundary we will insert it to the boundary now as it has contact with foreign neighbor
                        oldCellBoundarySetPtr->insert(BoundaryData(adjNeighborIndex,
                                                                   1)); ///newly inserted boundary point has 1 foreign neighbor
                    }

                    //continue;///do another iteration
                }

            }
        }


        /// Now will adjust common surface area with cell neighbors
        long temp_index;
        for (int i = 0; i < face2FaceNeighborSize; ++i) {
            ptAdj = pt;
            ptAdj += adjFace2FaceNeighborOffsetsVec_inn[i];

            //adjFace2FaceNeighborIndex = currentPtIndex + adjFace2FaceNeighborOffsetsVec[i];

         //if(!(adjFace2FaceNeighborIndex<0 || adjFace2FaceNeighborIndex > maxIndex ) ){
            CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";

//          if(!cellFieldG->isValid(ptAdj)){
   //          CC3D_Log(LOG_TRACE) << "ERROR - something is wrong with adjFace2FaceNeighborOffsetsVec_inn";
//             temp_index=field3DIndex.index(ptAdj);
//             CC3D_Log(LOG_TRACE) << "temp_index="<<temp_index<<" adjFace2FaceNeighborIndex="<<adjFace2FaceNeighborIndex;
//             CC3D_Log(LOG_TRACE) << "maxIndex="<<maxIndex;
//             CC3D_Log(LOG_TRACE) << "pt="<<pt<<" ptAdj="<<ptAdj;
//             //exit(0);
//          }
            if (cellFieldG->isValid(ptAdj)) {

               //adjCellPtr=cellFieldG->getByIndex(adjFace2FaceNeighborIndex);
               adjCellPtr=cellFieldG->get(ptAdj);
   
               if( adjCellPtr != oldCell ){ /// will decrement commSurfArea with all face 2 face neighbors
               // CC3D_Log(LOG_TRACE) << "adjCellPtr="<<adjCellPtr<<" oldCell="<<oldCell;
                  CC3D_Log(LOG_TRACE) << "ptAdj="<<ptAdj<<" pt="<<pt;
                    set_NSD_itr = oldCellNeighborSurfaceDataSetPtr->find(NeighborSurfaceData(adjCellPtr));
                    if (set_NSD_itr != oldCellNeighborSurfaceDataSetPtr->end()) {
                        set_NSD_itr->decrementCommonSurfaceArea(*set_NSD_itr); ///decrement commonSurfArea with adj cell
                        if (set_NSD_itr->OKToRemove()) ///if commSurfArea reaches 0 I remove this entry from cell neighbor set
                            oldCellNeighborSurfaceDataSetPtr->erase(set_NSD_itr);

                    } else {
                        CC3D_Log(LOG_DEBUG) << "Could not find cell address in the boundary - set of cellNeighbors is corrupted. Exiting ...";
                        exit(0);
                    }


                    if (adjCellPtr) { ///now process common area for adj cell provided it is not the oldCell
                        set <NeighborSurfaceData> &set_NSD_ref = cellBoundaryTrackerAccessor.get(
                                adjCellPtr->extraAttribPtr)->cellNeighbors;
                        set<NeighborSurfaceData>::iterator sitr;
                        sitr = set_NSD_ref.find(oldCell);
                        if (sitr != set_NSD_ref.end()) {
                            sitr->decrementCommonSurfaceArea(*sitr); ///decrement common area
                            if (sitr->OKToRemove()) ///if commSurfArea reaches 0 I remove this entry from cell neighbor set
                                set_NSD_ref.erase(sitr);

                        }
                    }
                }
            }


        }
    }

    if (newCell) {

        ///When there was a cell at this pixel we will have to update all its neighbours (update counters) and remove it from the boundary
//       currentCellPtr=cellFieldG->getByIndex(currentPtIndex);      
//       currentCellPtr=cellFieldG->getByIndex(currentPtIndex);
      // CC3D_Log(LOG_TRACE) << "\t\t new Cell section pt="<<pt<<" newCell="<<newCell;
      // CC3D_Log(LOG_TRACE) <<"newCellBoundarySet size="<<newCellBoundarySetPtr->size();
/*      if(newCellBoundarySetPtr->size())
         CC3D_Log(LOG_TRACE) <<" preinitial newCellBoundarySetPtr->begin()="<<newCellBoundarySetPtr->begin()->pixelIndex;
      else{
         CC3D_Log(LOG_TRACE) <<"\t\t SORRY EMPTY SET";
      }   */

        BdSitrN = newCellBoundarySetPtr->find(BoundaryData(currentPtIndex));
///RESTORE IT LATER _ IMPORTANT SANITY TEST      
//       if(BdSitrN!=newCellBoundarySetPtr->end()){
            CC3D_Log(LOG_TRACE)<< "OOOPS trying to insert a point:"<<currentPtIndex<<" into a boundary that already is there -  in the NEW CELL BONDARY";
            CC3D_Log(LOG_TRACE)<<"This is allowed only if lattice is 1 x 1 x 1. If this is not the case notify developer about bug";
//          exit(0);
//       }else{
               CC3D_Log(LOG_TRACE)<<"Could not find element BoundaryData(currentPtIndex)="<<currentPtIndex;
//       }



        InsSitrOKPair = (newCellBoundarySetPtr->insert(
                BoundaryData(currentPtIndex, 0)));  ///every new pixel in the cell must be added to the
        ///boundary

        InsSitr = InsSitrOKPair.first;

        for (int i = 0; i < neighborSize; ++i) {

            ptAdj = pt;
            ptAdj += adjNeighborOffsetsVec_inn[i];

            adjNeighborIndex = field3DIndex.index(ptAdj);

            if (cellFieldG->isValid(ptAdj)) {


                adjCellPtr = cellFieldG->get(ptAdj);

                if (adjCellPtr == newCell) {//modifying boundary data for new cell
                    BdSitrN = newCellBoundarySetPtr->find(
                            BoundaryData(adjNeighborIndex)); //check if a neighbor belongs to boundary
                    if (BdSitrN != newCellBoundarySetPtr->end()) {//if it does
                        BdSitrN->decrementNumberOfForeignNeighbors(*BdSitrN);
                        if (BdSitrN->OKToRemove()) {
                            newCellBoundarySetPtr->erase(BdSitrN);
                     CC3D_Log(LOG_TRACE)<<"\t\t\tafter erasing newCellBoundarySet size="<<newCellBoundarySetPtr->size();
                        }
                    }

                } else {//increment number of foreign neighbors for the newly added pixel
                    InsSitr->incrementNumberOfForeignNeighbors(*InsSitr);
                }

            } else { // if a pixel touches lattice boundary we treat adjacent pixel (the one that does not belong to boundary)
                // as a foreign neighbor . Otherwise this would cause pixel touching lattice boundary (e.g. pixel in the corner) to be removed
                //to early from the boundary. The only place where pixel touching boundary can be removed is in the if(oldCell) section
                ///have to add  special case when there are periodic boundary conditions set on - then simple test  cellFieldG->isValid(ptAdj
                ///may be misleading: i.e. cellFieldG->isValid(ptAdj can be false but a point still may not be at the boundary
                //if(isTouchingLatticeBoundary(pt,ptAdj))
                InsSitr->incrementNumberOfForeignNeighbors(*InsSitr);
            }


        }
        if (InsSitr->numberOfForeignNeighbors == 0) {//in case new pixel ends up fully inside the cell
            newCellBoundarySetPtr->erase(InsSitr);
        }

        /// Now will adjust common surface area with cell neighbors
        for (int i = 0; i < face2FaceNeighborSize; ++i) {
            ptAdj = pt;
            ptAdj += adjFace2FaceNeighborOffsetsVec_inn[i];
            adjFace2FaceNeighborIndex = field3DIndex.index(ptAdj);
            //adjFace2FaceNeighborIndex = currentPtIndex + adjFace2FaceNeighborOffsetsVec[i];

            //if(!(adjFace2FaceNeighborIndex<0 || adjFace2FaceNeighborIndex > maxIndex ) ){
            CC3D_Log(LOG_TRACE) <<"adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";
            if (cellFieldG->isValid(ptAdj)) {

                //adjCellPtr=cellFieldG->getByIndex(adjFace2FaceNeighborIndex);
                adjCellPtr = cellFieldG->get(ptAdj);

                if (adjCellPtr !=
                    newCell) { ///if adjCellPtr denotes foreign cell we increase common area and insert set entry if necessary
            CC3D_Log(LOG_TRACE) << "inserting adjCellPtr="<<adjCellPtr <<" ptAdj="<<ptAdj<<" into newCell="<<newCell<<" pt="<<pt;
                    set_NSD_itr_OK_Pair = newCellNeighborSurfaceDataSetPtr->insert(
                            NeighborSurfaceData(adjCellPtr));/// OK to insert even if
                    ///duplicate, in such a case an iterator to existing NeighborSurfaceData(adjCellPtr) obj is returned

                    set_NSD_itr = set_NSD_itr_OK_Pair.first;
                    set_NSD_itr->incrementCommonSurfaceArea(*set_NSD_itr); ///increment commonSurfArea with adj cell

                    if (adjCellPtr) { ///now process common area for adj cell
                        set <NeighborSurfaceData> &set_NSD_ref = cellBoundaryTrackerAccessor.get(
                                adjCellPtr->extraAttribPtr)->cellNeighbors;
                        pair<set<NeighborSurfaceData>::iterator, bool> sitr_OK_pair = set_NSD_ref.insert(newCell);
                        set<NeighborSurfaceData>::iterator sitr = sitr_OK_pair.first;
                        sitr->incrementCommonSurfaceArea(
                                *sitr); ///increment commonSurfArea of adj cell with current cell
                    }

                }

            }


        }
    }


    if (!oldCell) { ///this special case is required in updating common Surface Area with medium
        ///in this case we update surface of adjCell only (we do not update medium's neighbors list or its contact surfaces)

        /// Now will adjust common surface area with cell neighbors
        for (int i = 0; i < face2FaceNeighborSize; ++i) {
            ptAdj = pt;
            ptAdj += adjFace2FaceNeighborOffsetsVec_inn[i];

            adjFace2FaceNeighborIndex = field3DIndex.index(ptAdj);

            //adjFace2FaceNeighborIndex = currentPtIndex + adjFace2FaceNeighborOffsetsVec[i];

            //if(!(adjFace2FaceNeighborIndex<0 || adjFace2FaceNeighborIndex > maxIndex ) ){
            CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";
            if (cellFieldG->isValid(ptAdj)) {

                //adjCellPtr=cellFieldG->getByIndex(adjFace2FaceNeighborIndex);
                adjCellPtr = cellFieldG->get(ptAdj);

                if (adjCellPtr != oldCell &&
                    !(ptAdj == pt)) { /// will decrement commSurfArea with all face 2 face neighbors
            CC3D_Log(LOG_TRACE) << "!old cell section  adjCellPtr="<<adjCellPtr <<" ptAdj="<<ptAdj<<"  oldCell="<<oldCell<<" pt="<<pt;
                    if (adjCellPtr) { ///now process common area for adj cell provided it is not the oldCell
                        set <NeighborSurfaceData> &set_NSD_ref = cellBoundaryTrackerAccessor.get(
                                adjCellPtr->extraAttribPtr)->cellNeighbors;
                        set<NeighborSurfaceData>::iterator sitr;
                        sitr = set_NSD_ref.find(oldCell);
                        if (sitr != set_NSD_ref.end()) {
                            sitr->decrementCommonSurfaceArea(*sitr); ///decrement common area
                            if (sitr->OKToRemove()) { ///if commSurfArea reaches 0 I remove this entry from cell neighbor set
                                set_NSD_ref.erase(sitr);
                        CC3D_Log(LOG_TRACE) << "removing from boundary";

                            }

                        }
                    }
                }
            }

        }


    }

    if (!newCell) {  ///this special case is required in updating common Surface Area with medium
        ///in this case we update surface of adjCell only (we do not update medium's neighbors list or its contact surfaces)

        for (int i = 0; i < face2FaceNeighborSize; ++i) {
            ptAdj = pt;
            ptAdj += adjFace2FaceNeighborOffsetsVec_inn[i];

            //adjFace2FaceNeighborIndex = currentPtIndex + adjFace2FaceNeighborOffsetsVec[i];

            //if(!(adjFace2FaceNeighborIndex<0 || adjFace2FaceNeighborIndex > maxIndex ) ){
            CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";
            if (cellFieldG->isValid(ptAdj)) {


                //adjCellPtr=cellFieldG->getByIndex(adjFace2FaceNeighborIndex);
                adjCellPtr = cellFieldG->get(ptAdj);

                if (adjCellPtr !=
                    newCell) { ///if adjCellPtr denotes foreign cell we increase common area and insert set entry if necessary

                    if (adjCellPtr) { ///now process common area of adj cell with medium in this case
                        set <NeighborSurfaceData> &set_NSD_ref = cellBoundaryTrackerAccessor.get(
                                adjCellPtr->extraAttribPtr)->cellNeighbors;
                        pair<set<NeighborSurfaceData>::iterator, bool> sitr_OK_pair = set_NSD_ref.insert(newCell);
                        set<NeighborSurfaceData>::iterator sitr = sitr_OK_pair.first;
                        sitr->incrementCommonSurfaceArea(
                                *sitr); ///increment commonSurfArea of adj cell with current cell
                    }

                }

            }


        }

    }

//       if(newCellBoundarySetPtr->size())
//          CC3D_Log(LOG_TRACE) << " final newCellBoundarySetPtr->begin()="<<newCellBoundarySetPtr->begin()->pixelIndex;
//       else
//          CC3D_Log(LOG_TRACE) << "SORRY THE SET IN EMPTY";


    ///temporarily for testing purposes I set
    //cellFieldG->setByIndex(currentPtIndex,oldCell);

    ++changeCounter;
    int interval;
    if (changeCounter > 1) {
        interval = 100000;
    } else {
        interval = 100000;
    }
    if (!(changeCounter % interval)) {
        CC3D_Log(LOG_DEBUG) << "OLD CELL ADR: "<<oldCell<<" NEW CELL ADR: "<<newCell;
        CC3D_Log(LOG_DEBUG) << "ChangeCounter:"<<changeCounter;
        testLatticeSanityFull();

    }

    //cellFieldG->setByIndex(currentPtIndex,newCell);

}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CellBoundaryTrackerPlugin::initializeBoundaries() {
   CC3D_Log(LOG_TRACE) << "****************************** INITIALIZE BOUNDARIES NOW *****************************";
// 
//    Dim3D fieldDim=cellFieldG->getDim();
//    //AdjacentNeighbor adjNeighbor(fieldDim);
// 
// 
// 
// 
// 
// 
//    ///now will walk over entire lattice and will initialize cell boundaries
// 
//    /// Because blob does not occupy entire lattice we will walk over this region where blob
//    /// is located. Will implement it later
//    //int size = gap + width;
//    //Dim3D itDim=getBlobDimensions(fieldDim,size);
// 
//    const Field3DIndex & field3DIndex=adjNeighbor.getField3DIndex();
//    Point3D pt(0,0,0);
//    Point3D ptAdj;
// 
// //    const vector<long> & adjNeighborOffsetsVec=adjNeighbor.getAdjNeighborOffsetVec();
// //    int neighborSize=adjNeighborOffsetsVec.size();
// //
// //    const vector<long> & adjFace2FaceNeighborOffsetsVec=adjNeighbor.getAdjFace2FaceNeighborOffsetVec();
// //    int face2FaceNeighborSize=adjFace2FaceNeighborOffsetsVec.size();
// 
// 
// 
//    long currentPtIndex=0;
//    long adjNeighborIndex=0;
//    long adjFace2FaceNeighborIndex=0;
//    pair<set<BoundaryData>::iterator,bool > set_BD_itr_OK_Pair;
//    pair<set<NeighborSurfaceData>::iterator,bool > set_NSD_itr_OK_Pair;
//    set<BoundaryData>::iterator set_BD_itr;
//    set<NeighborSurfaceData>::iterator set_NSD_itr;
// 
// /*   Dim3D maxIndexDim = fieldDim; // to get coordinates of the "furthes from the origin" point of the field
//                                  // we need to subtract 1 from every coordinate of the field dimentsion
//                                  // otherwise we will get max_index which is way too large and we will produce segfault
//    maxIndexDim.x-=1;
//    maxIndexDim.y-=1;
//    maxIndexDim.z-=1;
//    long maxIndex=field3DIndex.index(maxIndexDim);*/
      CC3D_Log(LOG_TRACE) << " MAX INDEX CALCULATED = "<<maxIndex;
      CC3D_Log(LOG_TRACE) << " FIELD SIZE CALCULATED MANUALLY : "<<fieldDim.x*fieldDim.y*fieldDim.z;
// 
// 
//    ///Now will have to get access to the pointers stored in cellFieldG from Potts3D
// 
// 
//    CellG * currentCellPtr;
//    CellG * adjCellPtr;
// 
//    ///Lattice pass -  set boundary points of the cells an the counter of foreign neighbors for each boundary point
// 
//    for(int z=0 ; z < fieldDim.z ; ++z)
//       for(int y=0 ; y < fieldDim.y ; ++y)
//          for(int x=0 ; x < fieldDim.x ; ++x){
// 
//             pt.x=x;
//             pt.y=y;
//             pt.z=z;
// 
//             currentPtIndex=field3DIndex.index(pt);
//             //currentCellPtr=cellFieldG->getByIndex(currentPtIndex);
//             currentCellPtr=cellFieldG->get(pt);
// 
//             //const vector<Point3D> & adjNeighborOffsetsVec=adjNeighbor.getAdjNeighborOffsetVec(pt);
//             const vector<Point3D> & adjNeighborOffsetsVec=adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
//             int neighborSize=adjNeighborOffsetsVec.size();
// 
//             const vector<Point3D> & adjFace2FaceNeighborOffsetsVec=adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
//             int face2FaceNeighborSize=adjFace2FaceNeighborOffsetsVec.size();
// 
// 
//             if(!currentCellPtr)
//                continue; //skip the loop if the current latice site does not belong to any cell
// 
// 
//             set<BoundaryData> & set_ref=cellBoundaryTrackerAccessor.get(currentCellPtr->extraAttribPtr)->boundary; //get a reference to boundary
//                                                                                                    //set of a   current cell
// 
//             set<NeighborSurfaceData> & set_NSD_ref=cellBoundaryTrackerAccessor.get(currentCellPtr->extraAttribPtr)->cellNeighbors;
// 
//             for(int i = 0 ; i < neighborSize ; ++i){
//                ptAdj=pt;
//                ptAdj+=adjNeighborOffsetsVec[i];
//                adjNeighborIndex=field3DIndex.index(ptAdj);
//                //adjNeighborIndex=currentPtIndex+adjNeighborOffsetsVec[i];
// 
//                //if(!(adjNeighborIndex<0 || adjNeighborIndex > maxIndex ) ){
   //                CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";
//                if(cellFieldG->isValid(ptAdj)){
// 
//                   //adjCellPtr=cellFieldG->getByIndex(adjNeighborIndex);
//                   adjCellPtr=cellFieldG->get(ptAdj);
//                   if( adjCellPtr != currentCellPtr ){ //even if the neighbor ptr is zero currentPtr should be inserted
// 
//                      set_BD_itr_OK_Pair=set_ref.insert(BoundaryData(currentPtIndex));
//                      set_BD_itr=set_BD_itr_OK_Pair.first;
//                      set_BD_itr->incrementNumberOfForeignNeighbors(*set_BD_itr);
// 
//                   }
// 
//                }
//             }
//             /// adding face 2 face neighbors
//             for(int i = 0 ; i < face2FaceNeighborSize ; ++i){
// 
//                ptAdj=pt;
//                ptAdj+=adjFace2FaceNeighborOffsetsVec[i];
//                adjFace2FaceNeighborIndex=field3DIndex.index(ptAdj);
//                //adjFace2FaceNeighborIndex=currentPtIndex+adjFace2FaceNeighborOffsetsVec[i];
// 
//                //if(!(adjFace2FaceNeighborIndex<0 || adjFace2FaceNeighborIndex > maxIndex ) ){
   //                CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";
//                if(cellFieldG->isValid(ptAdj)){
// 
//                   adjCellPtr=cellFieldG->get(ptAdj);
//                   //adjCellPtr=cellFieldG->getByIndex(adjFace2FaceNeighborIndex);
// 
//                   if( adjCellPtr != currentCellPtr ){ //even if the neighbor ptr is zero currentPtr should be inserted
// 
//                      set_NSD_itr_OK_Pair=set_NSD_ref.insert(NeighborSurfaceData(adjCellPtr));
//                      set_NSD_itr=set_NSD_itr_OK_Pair.first;
//                      set_NSD_itr->incrementCommonSurfaceArea(*set_NSD_itr);
// 
//                   }
// 
//                }
// 
//             }
// 
// 
//          }
// 
// 
//          ///Now will test boundary sanity
//          testLatticeSanityFull();
//          CC3D_Log(LOG_TRACE) << "SANITY TEST PASSED";
// 
// 
//          ///After Initialization was done turnon the flag that it is OK to start using field3DChange function
//          watchingAllowed=true;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///This function checks if boundaries are OK

void CellBoundaryTrackerPlugin::testLatticeSanity() {

    set < CellG * > set_of_visited_cells;

    Dim3D fieldDim = cellFieldG->getDim();
    //AdjacentNeighbor adjNeighbor(fieldDim);

//   vector<long> const & neighborVecRef=adjNeighbor.getAdjNeighborOffsetVec();

    const Field3DIndex &field3DIndex = adjNeighbor.getField3DIndex();
    Point3D pt(0, 0, 0);
    Point3D ptAdj;
    Point3D ptBoundary;

//    const vector<long> & adjNeighborOffsetsVec=adjNeighbor.getAdjNeighborOffsetVec();
//    int neighborSize=adjNeighborOffsetsVec.size();
// 
//    const vector<long> & adjFace2FaceNeighborOffsetsVec=adjNeighbor.getAdjFace2FaceNeighborOffsetVec();
//    int face2FaceNeighborSize=adjFace2FaceNeighborOffsetsVec.size();


    long currentPtIndex = 0;
    long adjNeighborIndex = 0;
    long adjFace2FaceNeighborIndex = 0;



    ///Now will have to get access to the pointers stored in cellFieldG from Potts3D


    CellG *currentCellPtr;
    CellG *adjCellPtr;
    int localNeighborCounter = 0;
    int localCommonSurfaceArea = 0;
    set<BoundaryData>::iterator bdsitr;
    set<NeighborSurfaceData>::iterator nsdsitr;

    bool isInBoundary = false;
    bool inInNeighborSurfaceData = false;

    for (int z = 0; z < fieldDim.z; ++z)
        for (int y = 0; y < fieldDim.y; ++y)
            for (int x = 0; x < fieldDim.x; ++x) {

                pt.x = x;
                pt.y = y;
                pt.z = z;

                currentPtIndex = field3DIndex.index(pt);
                //currentCellPtr=cellFieldG->getByIndex(currentPtIndex);
                currentCellPtr = cellFieldG->get(pt);
                //const vector<Point3D> & adjNeighborOffsetsVec=adjNeighbor.getAdjNeighborOffsetVec(pt);
                const vector <Point3D> &adjNeighborOffsetsVec = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
                int neighborSize = adjNeighborOffsetsVec.size();

                const vector <Point3D> &adjFace2FaceNeighborOffsetsVec = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(
                        pt);
                int face2FaceNeighborSize = adjFace2FaceNeighborOffsetsVec.size();


                if (!currentCellPtr)
                    continue; //skip the loop if the current latice site does not belong to any cell

                ///OK I have  a reference to the boundary set
                set <BoundaryData> &set_ref = cellBoundaryTrackerAccessor.get(
                        currentCellPtr->extraAttribPtr)->boundary; //get a reference to boundary
                //set of a   current cell



                /// reset local variables
                localNeighborCounter = 0;
                isInBoundary = false;

                ///counting foreign neighbors of current pixel
                for (int i = 0; i < neighborSize; ++i) {
                    ptAdj = pt;
                    ptAdj += adjNeighborOffsetsVec[i];
                    adjNeighborIndex = field3DIndex.index(ptAdj);
                    //adjNeighborIndex=currentPtIndex+adjNeighborOffsetsVec[i];

                    //if(!(adjNeighborIndex<0 || adjNeighborIndex > maxIndex ) ){
                    if (cellFieldG->isValid(ptAdj)) {
                        CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";


                        //adjCellPtr=cellFieldG->getByIndex(adjNeighborIndex);
                        adjCellPtr = cellFieldG->get(ptAdj);

                        if (adjCellPtr !=
                            currentCellPtr) { //even if the neighbor ptr is zero currentPtr should be inserted
                            ++localNeighborCounter;
                            ///isInBoundary flag is set;
                            isInBoundary = true;
                        }

                    }
                }


                ///reseting local variables
                localCommonSurfaceArea = 0;
                inInNeighborSurfaceData = false;

                set <NeighborSurfaceData> &set_NSD_ref = cellBoundaryTrackerAccessor.get(
                        currentCellPtr->extraAttribPtr)->cellNeighbors;

                if (set_of_visited_cells.find(currentCellPtr) != set_of_visited_cells.end()) {
                    continue; ///cell has been already checked
                } else {
                    set_of_visited_cells.insert(currentCellPtr);
                    long indexBoundary;

                    set <NeighborSurfaceData> set_NSD_local;
                    pair<set<NeighborSurfaceData>::iterator, bool> set_NSD_itr_OK_Pair;
                    set<NeighborSurfaceData>::iterator set_NSD_itr;

                    for (set<BoundaryData>::iterator sitr = set_ref.begin();
                         sitr != set_ref.end(); ++sitr) {///loop over cell boundary points

                        indexBoundary = sitr->pixelIndex;
                        ptBoundary = field3DIndex.index2Point(indexBoundary);
                        ///CAUTION: you have to get a reference to adjFace2FaceNeighborOffsetsVec based on ptBoundary not based on pt!
                        const vector <Point3D> &adjFace2FaceNeighborOffsetsVec = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(
                                ptBoundary);
                        int face2FaceNeighborSize = adjFace2FaceNeighborOffsetsVec.size();

                        for (int i = 0;
                             i < face2FaceNeighborSize; ++i) {///loop over current pixel's face2face neighbors
                            ptAdj = ptBoundary;
                            ptAdj += adjFace2FaceNeighborOffsetsVec[i];
                            adjFace2FaceNeighborIndex = field3DIndex.index(ptAdj);
                            //adjFace2FaceNeighborIndex=indexBoundary+adjFace2FaceNeighborOffsetsVec[i];

                            //if(!(adjFace2FaceNeighborIndex<0 || adjFace2FaceNeighborIndex > maxIndex ) ){
                            // CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";
                            if (cellFieldG->isValid(ptAdj)) {

                                //adjCellPtr=cellFieldG->getByIndex(adjFace2FaceNeighborIndex);
                                adjCellPtr = cellFieldG->get(ptAdj);

                                if (adjCellPtr !=
                                    currentCellPtr) { //even if the neighbor ptr is zero currentPtr should be inserted

                                    set_NSD_itr_OK_Pair = set_NSD_local.insert(NeighborSurfaceData(adjCellPtr));
                                    set_NSD_itr = set_NSD_itr_OK_Pair.first;
                                    set_NSD_itr->incrementCommonSurfaceArea(*set_NSD_itr);


                                }

                            }

                        }


               }
               // CC3D_Log(LOG_TRACE) << "Checking cell:" << currentCellPtr;

                    if (set_NSD_local.size() != set_NSD_ref.size()) {
                        CC3D_Log(LOG_DEBUG) << "Sets have different sizes - orig:"<<set_NSD_ref.size()<<" local:"<<set_NSD_local.size()<<
                  " . Exining...";
                        exit(0);

                    }


                    set<NeighborSurfaceData>::iterator sitr_local = set_NSD_local.begin();
                    for (set<NeighborSurfaceData>::iterator sitr = set_NSD_ref.begin();
                         sitr != set_NSD_ref.end(); ++sitr) {


                        if (sitr_local->neighborAddress != sitr->neighborAddress) {
                            CC3D_Log(LOG_DEBUG) << "Neighbor addresses - orig: "<<sitr->neighborAddress<<" local:"<<sitr_local->neighborAddress<<
                     " do not match. Exiting ";
                            exit(0);
                        }
                        if (sitr_local->commonSurfaceArea != sitr->commonSurfaceArea) {
                            CC3D_Log(LOG_DEBUG) << "Neighbor commonSurfaceArea - orig: "<<sitr->commonSurfaceArea<<" local:"<<sitr_local->commonSurfaceArea<<
                     " do not match. Exiting ";
                            exit(0);
                        }
                        CC3D_Log(LOG_TRACE) << "neighbor:"<<sitr->neighborAddress<<" commonSurfaceArea:"<<sitr->commonSurfaceArea;

                        ++sitr_local;
                    }


                }

                ///Got everything I need to know about this pixel
                ///Now do sanity check
                if (isInBoundary) {
                    bdsitr = set_ref.find(BoundaryData(currentPtIndex));

                    if (bdsitr == set_ref.end()) {
                        CC3D_Log(LOG_DEBUG) << "Requested pixel:"<<currentPtIndex<< " was not found in the boundary set. Boundary corrupted. Exiting...";
                        CC3D_Log(LOG_DEBUG) << "Tried address "<<currentCellPtr;
                        exit(0);
                    } else if (bdsitr->numberOfForeignNeighbors != localNeighborCounter) {
                        CC3D_Log(LOG_DEBUG) << "Different Number of foreign neighbors - orig="<<bdsitr->numberOfForeignNeighbors
                  <<" instant init="<<localNeighborCounter;
                        CC3D_Log(LOG_DEBUG) << "Tried address "<<currentCellPtr;
                        exit(0);
                    } else if ((bdsitr->numberOfForeignNeighbors) > 26) {
                        CC3D_Log(LOG_DEBUG) << "Number of fN's is "<<bdsitr->numberOfForeignNeighbors<<" but allowed value is 26 ";
                        exit(0);

                    } else {
                  CC3D_Log(LOG_TRACE) << "OK,pixel:"<<currentPtIndex<<" number of foreign neighbors - orig="<<bdsitr->numberOfForeignNeighbors
                  <<" instant init="<<localNeighborCounter;

                    }
                }





         }
     CC3D_Log(LOG_DEBUG) << "LATTICE IS SANE!!!!!";

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double distance(double x1, double y1, double z1, double x2, double y2, double z2) {
    return sqrt(
            (x1 - x2) * (x1 - x2) +
            (y1 - y2) * (y1 - y2) +
            (z1 - z2) * (z1 - z2)
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void CellBoundaryTrackerPlugin::readXML(XMLPullParser &in) {
    //surfaceEnergy->readXML(in);
}

void CellBoundaryTrackerPlugin::writeXML(XMLSerializer &out) {
    //surfaceEnergy->writeXML(out);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CellBoundaryTrackerPlugin::testLatticeSanityFull() {

    set < CellG * > set_of_visited_cells;

    Dim3D fieldDim = cellFieldG->getDim();
    //AdjacentNeighbor adjNeighbor(fieldDim);

//   vector<long> const & neighborVecRef=adjNeighbor.getAdjNeighborOffsetVec();

    const Field3DIndex &field3DIndex = adjNeighbor.getField3DIndex();
    Point3D pt(0, 0, 0);
    Point3D ptAdj;
    Point3D ptBoundary;

//    const vector<long> & adjNeighborOffsetsVec=adjNeighbor.getAdjNeighborOffsetVec();
//    int neighborSize=adjNeighborOffsetsVec.size();
// 
//    const vector<long> & adjFace2FaceNeighborOffsetsVec=adjNeighbor.getAdjFace2FaceNeighborOffsetVec();
//    int face2FaceNeighborSize=adjFace2FaceNeighborOffsetsVec.size();


    long currentPtIndex = 0;
    long adjNeighborIndex = 0;
    long adjFace2FaceNeighborIndex = 0;



    ///Now will have to get access to the pointers stored in cellFieldG from Potts3D


    CellG *currentCellPtr;
    CellG *adjCellPtr;
    int localNeighborCounter = 0;
    int localCommonSurfaceArea = 0;
    set<BoundaryData>::iterator bdsitr;
    set<NeighborSurfaceData>::iterator nsdsitr;

    bool isInBoundary = false;
    bool inInNeighborSurfaceData = false;

    map < CellG * , set < BoundaryData > > mapCellBoundaryData;
    map < CellG * , set < BoundaryData > > ::iterator
    mitr;

    /// check boundaries of each cell - will loop over each lattice point and check if point belongs to boundary
    for (int z = 0; z < fieldDim.z; ++z)
        for (int y = 0; y < fieldDim.y; ++y)
            for (int x = 0; x < fieldDim.x; ++x) {
                pt.x = x;
                pt.y = y;
                pt.z = z;
                currentPtIndex = field3DIndex.index(pt);
                //currentCellPtr=cellFieldG->getByIndex(currentPtIndex);
                currentCellPtr = cellFieldG->get(pt);
                if (!currentCellPtr)
                    continue; //skip the loop if the current latice site does not belong to any cell


                //curent cell is different than medium
                if (isBoundaryPixel(pt)) {
                    mitr = mapCellBoundaryData.find(currentCellPtr);

                    if (mitr != mapCellBoundaryData.end()) {
                        mitr->second.insert(BoundaryData(currentPtIndex));
                    } else {
                        set <BoundaryData> tmpSet;
                        tmpSet.insert(BoundaryData(currentPtIndex));
                        mapCellBoundaryData.insert(make_pair(currentCellPtr, tmpSet));
                    }
                }

            }

    //Now check if boundaryData, just initlilized, matches what is in the cell attributes
/*   int cellsChecked=0;
   for(mitr = mapCellBoundaryData.begin() ; mitr != mapCellBoundaryData.end() ; ++mitr){
      CellG * tmpCellPtr = mitr->first;
      //just initialized
      set<BoundaryData> & tmpSetBoundaryData = mitr->second;
      //from cell attribute
      set<BoundaryData> & cellSetBoundaryData = cellBoundaryTrackerAccessor.get(tmpCellPtr->extraAttribPtr)->boundary ;

      if(tmpSetBoundaryData != cellSetBoundaryData){
         CC3D_Log(LOG_TRACE) << " Cell Address: "<<tmpCellPtr;
         CC3D_Log(LOG_TRACE) << " Cell volume: "<<tmpCellPtr->volume;
         CC3D_Log(LOG_TRACE) << " Cell surface: "<<tmpCellPtr->surface;
         CC3D_Log(LOG_TRACE) << " Cell COM: "
         <<tmpCellPtr->xCM/(float)tmpCellPtr->volume<<" "
         <<tmpCellPtr->yCM/(float)tmpCellPtr->volume<<" "
         <<tmpCellPtr->zCM/(float)tmpCellPtr->volume;
         CC3D_Log(LOG_TRACE) << "Boundary sets do not match";
         CC3D_Log(LOG_TRACE) << " original set size="<<cellSetBoundaryData.size();
         CC3D_Log(LOG_TRACE) << " check set size="<<tmpSetBoundaryData.size();
         CC3D_Log(LOG_TRACE) << "checked "<<cellsChecked<<" cells";
         CC3D_Log(LOG_TRACE) << "ORIG:************************";
         for(set<BoundaryData>::iterator itr = cellSetBoundaryData.begin() ; itr != cellSetBoundaryData.end() ; ++itr){
            CC3D_Log(LOG_TRACE) << "neighbor index="<<itr->pixelIndex<<" number of pix neighbors="<<itr->numberOfForeignNeighbors;
         }
         CC3D_Log(LOG_TRACE) << "CHECK:************************";
         for(set<BoundaryData>::iterator itr = tmpSetBoundaryData.begin() ; itr != tmpSetBoundaryData.end() ; ++itr){
            CC3D_Log(LOG_TRACE) << "neighbor index="<<itr->pixelIndex<<" number of pix neighbors="<<itr->numberOfForeignNeighbors;
         }

         exit(0);
      }
      ++cellsChecked;
   }*/

    for (int z = 0; z < fieldDim.z; ++z)
        for (int y = 0; y < fieldDim.y; ++y)
            for (int x = 0; x < fieldDim.x; ++x) {

                pt.x = x;
                pt.y = y;
                pt.z = z;

                currentPtIndex = field3DIndex.index(pt);
                //currentCellPtr=cellFieldG->getByIndex(currentPtIndex);
                currentCellPtr = cellFieldG->get(pt);
                //const vector<Point3D> & adjNeighborOffsetsVec=adjNeighbor.getAdjNeighborOffsetVec(pt);
                const vector <Point3D> &adjNeighborOffsetsVec = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
                int neighborSize = adjNeighborOffsetsVec.size();

                const vector <Point3D> &adjFace2FaceNeighborOffsetsVec = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(
                        pt);
                int face2FaceNeighborSize = adjFace2FaceNeighborOffsetsVec.size();


                if (!currentCellPtr)
                    continue; //skip the loop if the current latice site does not belong to any cell

                ///OK I have  a reference to the boundary set
                set <BoundaryData> &set_ref = cellBoundaryTrackerAccessor.get(
                        currentCellPtr->extraAttribPtr)->boundary; //get a reference to boundary
                //set of a   current cell



                /// reset local variables
                localNeighborCounter = 0;
                isInBoundary = false;

                ///counting foreign neighbors of current pixel
                for (int i = 0; i < neighborSize; ++i) {
                    ptAdj = pt;
                    ptAdj += adjNeighborOffsetsVec[i];
                    adjNeighborIndex = field3DIndex.index(ptAdj);
                    //adjNeighborIndex=currentPtIndex+adjNeighborOffsetsVec[i];

                    //if(!(adjNeighborIndex<0 || adjNeighborIndex > maxIndex ) ){
                    if (cellFieldG->isValid(ptAdj)) {
                        CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";


                        //adjCellPtr=cellFieldG->getByIndex(adjNeighborIndex);
                        adjCellPtr = cellFieldG->get(ptAdj);

                        if (adjCellPtr !=
                            currentCellPtr) { //even if the neighbor ptr is zero currentPtr should be inserted
                            ++localNeighborCounter;
                            ///isInBoundary flag is set;
                            isInBoundary = true;
                        }

                    } else {//if cell touches border it is in the boundary and we count it as a neighbor
                        //if(isTouchingLatticeBoundary(pt,ptAdj)){
                        ++localNeighborCounter;
                        ///isInBoundary flag is set;
                        isInBoundary = true;
                        //}

                    }
                }


                ///reseting local variables
                localCommonSurfaceArea = 0;
                inInNeighborSurfaceData = false;

                set <NeighborSurfaceData> &set_NSD_ref = cellBoundaryTrackerAccessor.get(
                        currentCellPtr->extraAttribPtr)->cellNeighbors;

                if (set_of_visited_cells.find(currentCellPtr) != set_of_visited_cells.end()) {
                    continue; ///cell has been already checked
                } else {
                    set_of_visited_cells.insert(currentCellPtr);
                    long indexBoundary;

                    set <NeighborSurfaceData> set_NSD_local;
                    pair<set<NeighborSurfaceData>::iterator, bool> set_NSD_itr_OK_Pair;
                    set<NeighborSurfaceData>::iterator set_NSD_itr;

                    for (set<BoundaryData>::iterator sitr = set_ref.begin();
                         sitr != set_ref.end(); ++sitr) {///loop over cell boundary points

                        indexBoundary = sitr->pixelIndex;
                        ptBoundary = field3DIndex.index2Point(indexBoundary);
                        ///CAUTION: you have to get a reference to adjFace2FaceNeighborOffsetsVec based on ptBoundary not based on pt!
                        const vector <Point3D> &adjFace2FaceNeighborOffsetsVec = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(
                                ptBoundary);
                        int face2FaceNeighborSize = adjFace2FaceNeighborOffsetsVec.size();

                        for (int i = 0;
                             i < face2FaceNeighborSize; ++i) {///loop over current pixel's face2face neighbors
                            ptAdj = ptBoundary;
                            ptAdj += adjFace2FaceNeighborOffsetsVec[i];
                            adjFace2FaceNeighborIndex = field3DIndex.index(ptAdj);
                            //adjFace2FaceNeighborIndex=indexBoundary+adjFace2FaceNeighborOffsetsVec[i];

                            //if(!(adjFace2FaceNeighborIndex<0 || adjFace2FaceNeighborIndex > maxIndex ) ){
                            // CC3D_Log(LOG_TRACE) << "adjNeighborIndex="<<adjNeighborIndex<<" OUT OF THE LATTICE";
                            if (cellFieldG->isValid(ptAdj)) {

                                //adjCellPtr=cellFieldG->getByIndex(adjFace2FaceNeighborIndex);
                                adjCellPtr = cellFieldG->get(ptAdj);

                                if (adjCellPtr !=
                                    currentCellPtr) { //even if the neighbor ptr is zero currentPtr should be inserted

                                    set_NSD_itr_OK_Pair = set_NSD_local.insert(NeighborSurfaceData(adjCellPtr));
                                    set_NSD_itr = set_NSD_itr_OK_Pair.first;
                                    set_NSD_itr->incrementCommonSurfaceArea(*set_NSD_itr);


                                }

                            }

                        }


               }
               CC3D_Log(LOG_TRACE) << "Checking cell:" << currentCellPtr;

                    if (set_NSD_local.size() != set_NSD_ref.size()) {
                        CC3D_Log(LOG_DEBUG) << "Sets have different sizes - orig:"<<set_NSD_ref.size()<<" local:"<<set_NSD_local.size()<<
                  " . Exining...";
                        exit(0);

                    }


                    set<NeighborSurfaceData>::iterator sitr_local = set_NSD_local.begin();
                    for (set<NeighborSurfaceData>::iterator sitr = set_NSD_ref.begin();
                         sitr != set_NSD_ref.end(); ++sitr) {


                        if (sitr_local->neighborAddress != sitr->neighborAddress) {
                            CC3D_Log(LOG_DEBUG) << "Neighbor addresses - orig: "<<sitr->neighborAddress<<" local:"<<sitr_local->neighborAddress<<
                     " do not match. Exiting ";
                            exit(0);
                        }
                        if (sitr_local->commonSurfaceArea != sitr->commonSurfaceArea) {
                            CC3D_Log(LOG_DEBUG) << "Neighbor commonSurfaceArea - orig: "<<sitr->commonSurfaceArea<<" local:"<<sitr_local->commonSurfaceArea<<
                     " do not match. Exiting ";
                            exit(0);
                        }
                        CC3D_Log(LOG_TRACE) << "neighbor:"<<sitr->neighborAddress<<" commonSurfaceArea:"<<sitr->commonSurfaceArea;

                        ++sitr_local;
                    }


                }

                ///Got everything I need to know about this pixel
                ///Now do sanity check
                if (isInBoundary) {
                    bdsitr = set_ref.find(BoundaryData(currentPtIndex));

                    if (bdsitr == set_ref.end()) {
                        CC3D_Log(LOG_DEBUG) << "Requested pixel:"<<currentPtIndex<< " was not found in the boundary set. Boundary corrupted. Exiting...";
                        CC3D_Log(LOG_DEBUG) << "Tried address "<<currentCellPtr;
                        exit(0);
                    } else if (bdsitr->numberOfForeignNeighbors != localNeighborCounter) {
                        CC3D_Log(LOG_DEBUG) << "Different Number of foreign neighbors - orig="<<bdsitr->numberOfForeignNeighbors
                  <<" instant init="<<localNeighborCounter;
                        CC3D_Log(LOG_DEBUG) << "Tried address "<<currentCellPtr;
                        exit(0);
                    } else if ((bdsitr->numberOfForeignNeighbors) > 26) {
                        CC3D_Log(LOG_DEBUG) << "Number of fN's is "<<bdsitr->numberOfForeignNeighbors<<" but allowed value is 26 ";
                        exit(0);

                    } else {
                  CC3D_Log(LOG_TRACE) << "OK,pixel:"<<currentPtIndex<<" number of foreign neighbors - orig="<<bdsitr->numberOfForeignNeighbors
                  <<" instant init="<<localNeighborCounter;

                    }
                }





         }
     CC3D_Log(LOG_DEBUG) << "LATTICE IS SANE!!!!!";

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool CellBoundaryTrackerPlugin::isBoundaryPixel(Point3D pt) {

    const vector <Point3D> &adjNeighborOffsetsVec = adjNeighbor.getAdjFace2FaceNeighborOffsetVec(pt);
    CellG *currentCellPtr = cellFieldG->get(pt);;
    CellG *adjCellPtr;


    Point3D ptAdj;
    for (int i = 0; i < adjNeighborOffsetsVec.size(); ++i) {
        ptAdj = pt;
        ptAdj += adjNeighborOffsetsVec[i];
        //adjNeighborIndex=field3DIndex.index(ptAdj);

        if (cellFieldG->isValid(ptAdj)) {
            adjCellPtr = cellFieldG->get(ptAdj);

            if (adjCellPtr != currentCellPtr) {
            CC3D_Log(LOG_TRACE) << "pt="<<pt<<" ON";
            CC3D_Log(LOG_TRACE) << "ptAdj="<<ptAdj<<" adr="<<adjCellPtr;
                return true;
            }
        } else {//means pixel is at lattice border - thus belongs to cell boundary
      CC3D_Log(LOG_TRACE) << "pt="<<pt<<" ON not in the field";
CC3D_Log(LOG_TRACE) << "not in the field ptAdj";
            return true;
            if (isTouchingLatticeBoundary(pt, ptAdj))
                return true;
            else
                continue;
            //return true;
        }
    }
    CC3D_Log(LOG_TRACE) << "pt="<<pt<<" OFF";
    return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///temporary solution only - will write general version later
bool CellBoundaryTrackerPlugin::isTouchingLatticeBoundary(Point3D pt, Point3D ptAdj) {


    ptAdj -= pt; //vector from pt to ptAdj: by construction it will have only component along x xor y xor z

    //vector from pt to ptAdj has only x component
    if (ptAdj.x != 0 && periodicX) {

        return false;
    } else if (ptAdj.x != 0 && !periodicX) {

        return true;

    }

    //vector from pt to ptAdj has only y component
    if (ptAdj.y != 0 && periodicY) {

        return false;
    } else if (ptAdj.y != 0 && !periodicY) {

        return true;

    }
    //vector from pt to ptAdj has only z component
    if (ptAdj.z != 0 && periodicZ) {

        return false;
    } else if (ptAdj.z != 0 && !periodicZ) {

        return true;

    }

}
