#include <CompuCell3D/CC3D.h>

using namespace CompuCell3D;

using namespace std;


#include "ObjInitializer.h"
#include <Logger/CC3DLogger.h>

// ----------------------------------------------------------------------
// class constructor without file name parameter:
// ----------------------------------------------------------------------
ObjInitializer::ObjInitializer() :
// initializer list, assigning initial values to members of the class object:
        potts(0), gObjFileName("") {}

// ----------------------------------------------------------------------
// class constructor with file name parameter:
// ----------------------------------------------------------------------
ObjInitializer::ObjInitializer(string filename) :
// initializer list, assigning initial values to members of the class object:
        potts(0), gObjFileName(filename) {}


// ----------------------------------------------------------------------
// initialize the ObjInitializer steppable:
// ----------------------------------------------------------------------
void ObjInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

    bool pluginAlreadyRegisteredFlag;

    fprintf(stderr, "0 ----------------------------------------------------------------------\n");
    fprintf(stderr, "0 ----------------------------------------------------------------------\n");
    fprintf(stderr, "    void ObjInitializer::init() \n");
    fprintf(stderr, "0 ----------------------------------------------------------------------\n");
    fprintf(stderr, "0 ----------------------------------------------------------------------\n");

    // If the resolution operator "::" is placed between the class name
    //    and the data member belonging to the class,
    //    then the data name belonging to the particular class is referenced:
    Plugin *plugin = Simulator::pluginManager.get("VolumeTracker", &pluginAlreadyRegisteredFlag);
    fprintf(stderr, "1 -ObjInitializer::init()-----------------------------------------------\n");
    // <-- this will load the VolumeTracker plugin, if it is not already loaded
    if (!pluginAlreadyRegisteredFlag) {
        fprintf(stderr, "1b -ObjInitializer::init()----------------------------------------------\n");
        plugin->init(simulator);
    }

    fprintf(stderr, "2 -ObjInitializer::init()-----------------------------------------------\n");
    gObjFileName = _xmlData->getFirstElement("ObjName")->getText();
    fprintf(stderr, "2 -ObjInitializer::init()-----------------------------------------------\n");


    fprintf(stderr, "3 -ObjInitializer::init()-----------------------------------------------\n");
    std::string basePath = simulator->getBasePath();

    fprintf(stderr, "4 -ObjInitializer::init()-----------------------------------------------\n");
    if (basePath != "") {
        fprintf(stderr, "4b -ObjInitializer::init()----------------------------------------------\n");
        gObjFileName = basePath + "/" + gObjFileName;
    }
    fprintf(stderr, "5 -ObjInitializer::init()-----------------------------------------------\n");

    potts = simulator->getPotts();
    fprintf(stderr, "6 -ObjInitializer::init()-----------------------------------------------\n");

} // end of void ObjInitializer::init(Simulator *simulator, CC3DXMLElement *_xmlData)



// ----------------------------------------------------------------------
// void ObjInitializer::start()
//   this function is called only the first time... hopefully!
// ----------------------------------------------------------------------
void ObjInitializer::start() {
    long lSpin;                          // lSpin is the cell ID integer value
    // long lClusterId;
    std::string lCellTypeString;         // lCellTypeString is the cell Type string value
    std::string lFirstParsedString;
    std::string lSecondParsedString;
    std::string lLineString;
    int xLow, xHigh, yLow, yHigh, zLow, zHigh;
    Point3D lCellPoint3D;
    CellG *lCellG;
    bool lEndReachedInObjParsing;        // boolean set to true to stop parsing the input file

    // lSpinMap is used to check if a cell of the same spin
    //   (here it means a cell with the same integer ID) is listed twice:
    //   the lSpinMap uses "long" as keys and "Point3D" as stored values:
    std::map<long, Point3D> lSpinMap;

    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "    void ObjInitializer::start()   \n");
    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "----------------------------------------------------------------------\n");

    //
    // try to open the OBJ file pointed by gObjFileName:
    //
    CC3D_Log(LOG_DEBUG) << "ppdPtr->gObjFileName=" << gObjFileName;
    std::ifstream lObjFileStream(gObjFileName.c_str(), ios::in);
    CC3D_Log(LOG_DEBUG) << "ObjInitializer::start() ----- opened pid file";
    if (!lObjFileStream.good())
        throw CC3DException(string("ObjInitializer::start() ----- Could not open [" + gObjFileName +
                                   "] ....make sure it exists in the correct directory."));

    //
    // prepare a watchable field 3D as from the potts's  getCellFieldG() :
    //
    WatchableField3D < CellG * > *lCellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    if (!lCellFieldG) throw CC3DException("ObjInitializer::start() ----- initField() Cell field cannot be null!");

    //
    // get the x,y,z dimensions of the 3D field of cells:
    //
    Dim3D lDimensions = lCellFieldG->getDim();
    CC3D_Log(LOG_DEBUG) << "ObjInitializer::start() ----- Potts Dimensions are set as: " << lDimensions;

    //
    // TODO: what is this TypeTransition doing here, what's its purpose?
    //
    TypeTransition *typeTransitionPtr = potts->getTypeTransition();
    CC3D_Log(LOG_DEBUG) <<"ObjInitializer::start() ----- typeTransitionPtr=" << typeTransitionPtr;

    //
    // read one single text line from lObjFileStream into a lLineString:
    //
    getline(lObjFileStream, lLineString);

    //
    // process the lLineString by placing it into a new istringstream named "lObjIStringStream":
    //
    istringstream lObjIStringStream(lLineString);

    //
    // split the lObjStringStream into separate strings for parsing and converting to numeric data :
    //    the first two elements in a generic line of a PIF file are: 
    //    n = an integer, specifying the cell ID
    //    s = a text string (no spaces), specifying the cell Type
    lObjIStringStream >> lFirstParsedString >> lSecondParsedString;
    CC3D_Log(LOG_DEBUG) << "ObjInitializer::start() ----- First: " << lFirstParsedString << " Second: " << lSecondParsedString << "\n";


    // store the cell ID integer and call it "lSpin" :
    int tmp = atoi(lFirstParsedString.c_str());
    lSpin = tmp;
    // store the cell type string:
    lCellTypeString = lSecondParsedString;
    // CC3D_Log(LOG_DEBUG) << "lSpin: " << lSpin << " lCellTypeString: : " << lCellTypeString <<
    //     " xLow: " << xLow;

    //
    // now parse 6 integers: xLow, xHigh, yLow, yHigh, zLow, zHigh
    //   and check that they all are within field boundaries, and that they are consistent (always low <= high)
    //


// -----    
// -----    
// -----    
// -----    
// -----    
//
//  from here onwards, all the code before "while" is the same as the code after the "while.
//    therefore, check that all is set the same up to here!
//

    lEndReachedInObjParsing = false;

    while (lEndReachedInObjParsing == false) {

        lObjIStringStream >> xLow;
        if (!(xLow >= 0 && xLow < lDimensions.x))
            throw CC3DException(string("OBJ reader: xLow out of bounds : \n") + lLineString);
        lObjIStringStream >> xHigh;
        if (!(xHigh >= 0 && xHigh < lDimensions.x))
            throw CC3DException(string("OBJ reader: xHigh out of bounds : \n") + lLineString);
        if (xHigh < xLow) throw CC3DException(string("OBJ reader: xHigh is smaller than xLow : \n") + lLineString);
        //
        lObjIStringStream >> yLow;
        if (!(yLow >= 0 && yLow < lDimensions.y))
            throw CC3DException(string("OBJ reader: yLow out of bounds : \n") + lLineString);
        lObjIStringStream >> yHigh;
        if (!(yHigh >= 0 && yHigh < lDimensions.y))
            throw CC3DException(string("OBJ reader: yHigh out of bounds : \n") + lLineString);
        if (yHigh < yLow) throw CC3DException(string("OBJ reader: yHigh is smaller than yLow : \n") + lLineString);
        //
        lObjIStringStream >> zLow;
        if (!(zLow >= 0 && zLow < lDimensions.z))
            throw CC3DException(string("OBJ reader: zLow out of bounds : \n") + lLineString);
        lObjIStringStream >> zHigh;
        if (!(zHigh >= 0 && zHigh < lDimensions.z))
            throw CC3DException(string("OBJ reader: zHigh out of bounds : \n") + lLineString);
        if (zHigh < zLow) throw CC3DException(string("OBJ reader: zHigh is smaller than xLow : \n") + lLineString);

        if (lSpinMap.count(lSpin) != 0) // Spin (i.e. current cell ID) already listed
        {
            for (lCellPoint3D.z = zLow; lCellPoint3D.z <= zHigh; lCellPoint3D.z++)
                for (lCellPoint3D.y = yLow; lCellPoint3D.y <= yHigh; lCellPoint3D.y++)
                    for (lCellPoint3D.x = xLow; lCellPoint3D.x <= xHigh; lCellPoint3D.x++) {

                        // the distinction between "new lSpin" and "old lSpin" could be taken out (and in front) of this x/y/z loop!
                        lCellFieldG->set(lCellPoint3D, lCellFieldG->get(lSpinMap[lSpin]));

                        potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                        //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                        // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                        //inventory unless you call steppers(VolumeTrackerPlugin) explicitely

                    }
        } else // First time for this spin (i.e. current cell ID), we need to create a new cell
        {
            // add a new cell ID (i.e. lSpin) to the lSpinMap:
            lSpinMap[lSpin] = Point3D(xLow, yLow, zLow);
            // create a new CellG element:
            lCellG = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow), lSpin);

            potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
            //It is necessary to do it this way because steppers are called only when we are performing pixel copies
            // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
            //inventory unless you call steppers(VolumeTrackerPlugin) explicitely

            for (lCellPoint3D.z = zLow; lCellPoint3D.z <= zHigh; lCellPoint3D.z++)
                for (lCellPoint3D.y = yLow; lCellPoint3D.y <= yHigh; lCellPoint3D.y++)
                    for (lCellPoint3D.x = xLow; lCellPoint3D.x <= xHigh; lCellPoint3D.x++) {

                        lCellFieldG->set(lCellPoint3D, lCellG);
                        potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
                        //It is necessary to do it this way because steppers are called only when we are performing pixel copies
                        // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
                        //inventory unless you call steppers(VolumeTrackerPlugin) explicitely

                    }
    
            typeTransitionPtr->setType( lCellG, potts->getAutomaton()->getTypeId(lCellTypeString));
            // CC3D_Log(LOG_DEBUG) << "1. Cell Type from lCellG->:  " << (int)lCellG->type << "\n";
            // CC3D_Log(LOG_DEBUG) << "getline(lObjIStringStream,lLineString): " << getline(lObjIStringStream,lLineString);
            // CC3D_Log(LOG_DEBUG) << "getline(objfline,lLineString): " <<  getline(lObjFileStream,lLineString);
        } // end of  if (lSpinMap.count(lSpin) != 0) // Spin (i.e. current cell ID) already listed


        //
        // here completed the parsing of one PIF line, get another line from the input file stream:
        //
        fprintf(stderr, "7 -ObjInitializer::start()----------------------------------------------\n");

//        printf "    getline(lObjFileStream, lLineString)
        fprintf(stderr, "7 -ObjInitializer::start()----------------------------------------------\n");
        if (getline(lObjFileStream, lLineString)) {
            // there is more data available from the input file, keep parsing:
            lEndReachedInObjParsing = false;
            // get another stream of tokens from the lLineString:
            istringstream lObjIStringStream(lLineString);
            // parse lSpin and lCellTypeString from the input stream of tokens,
            //   for the next loop:
            lObjIStringStream >> lSpin >> lCellTypeString;
        } else {
            // no more input file lines, let's end the loop:
            lEndReachedInObjParsing = true;
        }

    } // end of  while( lEndReachedInObjParsing == false )


// *** the following commented-out part needs to be checked out before removing it completely: ***


// 
// 
//     while( getline(lObjFileStream,lLineString)  ) {
// 
//         istringstream lObjIStringStream(lLineString);
//         lObjIStringStream >> lSpin >> lCellTypeString >> xLow;
// CC3D_Log(LOG_DEBUG) <<  "lSpin: " << lSpin << " lCellTypeString: : " << lCellTypeString <<
//         //     " xLow: " << xLow;
//         ASSERT_OR_THROW(string("OBJ reader: xLow out of bounds : \n") + lLineString, xLow >= 0 && xLow < lDimensions.x);
//         lObjIStringStream >> xHigh;
//         ASSERT_OR_THROW(string("OBJ reader: xHigh out of bounds : \n") + lLineString, xHigh >= 0 && xHigh < lDimensions.x);
//         ASSERT_OR_THROW(string("OBJ reader: xHigh is smaller than xLow : \n") + lLineString, xHigh >= xLow); 
//         lObjIStringStream >> yLow;
//         ASSERT_OR_THROW(string("OBJ reader: yLow out of bounds : \n") + lLineString, yLow >= 0 && yLow < lDimensions.y);
//         lObjIStringStream >> yHigh;   
//         ASSERT_OR_THROW(string("OBJ reader: yHigh out of bounds : \n") + lLineString, yHigh >= 0 && yHigh < lDimensions.y);
//         ASSERT_OR_THROW(string("OBJ reader: yHigh is smaller than yLow : \n") + lLineString, yHigh >= yLow);
//         lObjIStringStream >> zLow;
//         ASSERT_OR_THROW(string("OBJ reader: zLow out of bounds : \n") + lLineString, zLow >= 0 && zLow < lDimensions.z);
//         lObjIStringStream >> zHigh;
//         ASSERT_OR_THROW(string("OBJ reader: zHigh out of bounds: \n ") + lLineString, zHigh >= 0 && zHigh < lDimensions.z);
//         ASSERT_OR_THROW(string("OBJ reader: zHigh is smaller than xLow: \n") + lLineString, zHigh >= zLow);
//         
//         if (lSpinMap.count(lSpin) != 0) // Spin multiply listed
//         {
//             for (lCellPoint3D.z = zLow; lCellPoint3D.z <= zHigh; lCellPoint3D.z++)
//                 for (lCellPoint3D.y = yLow; lCellPoint3D.y <= yHigh; lCellPoint3D.y++)
//                     for (lCellPoint3D.x = xLow; lCellPoint3D.x <= xHigh; lCellPoint3D.x++){
//                         lCellFieldG->set(lCellPoint3D, lCellFieldG->get(lSpinMap[lSpin]));
//                         potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
//                         //It is necessary to do it this way because steppers are called only when we are performing pixel copies
//                         // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
//                         //inventory unless you call steppers(VolumeTrackerPlugin) explicitely
//                         
//                     }
// CC3D_Log(LOG_DEBUG) << "2. Cell Type from lCellG->:  " << (int)lCellG->type << "\n";
//             
//         }
//         else // First time for this spin, we need to create a new cell
//         {
//             lSpinMap[lSpin] = Point3D(xLow, yLow, zLow);
//             lCellG = potts->createCellGSpecifiedIds(Point3D(xLow, yLow, zLow),lSpin);
//             potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
//             //It is necessary to do it this way because steppers are called only when we are performing pixel copies
//             // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
//             //inventory unless you call steppers(VolumeTrackerPlugin) explicitely
//             
//             for (lCellPoint3D.z = zLow; lCellPoint3D.z <= zHigh; lCellPoint3D.z++)
//                 for (lCellPoint3D.y = yLow; lCellPoint3D.y <= yHigh; lCellPoint3D.y++)
//                     for (lCellPoint3D.x = xLow; lCellPoint3D.x <= xHigh; lCellPoint3D.x++){
//                         lCellFieldG->set(lCellPoint3D, lCellG);
//                         potts->runSteppers(); //used to ensure that VolumeTracker Plugin step fcn gets called every time we do something to the fields
//                         //It is necessary to do it this way because steppers are called only when we are performing pixel copies
//                         // but if we initialize steppers are not called thus is you overwrite a cell here it will not get removed from
//                         //inventory unless you call steppers(VolumeTrackerPlugin) explicitely
//                         
//                     }
//             
//             typeTransitionPtr->setType( lCellG, potts->getAutomaton()->getTypeId(lCellTypeString));
// CC3D_Log(LOG_DEBUG) << "2. Cell Type from lCellG->:  " << (int)lCellG->type << "\n";
//             
//         } // end of if (lSpinMap.count(lSpin) != 0) // Spin multiply listed
//         
//     }

    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "    end of void ObjInitializer::start() \n");
    fprintf(stderr, "----------------------------------------------------------------------\n");
    fprintf(stderr, "----------------------------------------------------------------------\n");

} // end of ObjInitializer::start()
