

#include <CompuCell3D/CC3D.h>
#include <CompuCell3D/plugins/CellType/CellTypePlugin.h>

using namespace CompuCell3D;
using namespace std;


#include "PIFDumper.h"

PIFDumper::PIFDumper() :
        potts(0), pifFileExtension("piff") {}

PIFDumper::PIFDumper(string filename) :
        potts(0), pifFileExtension("piff") {}

void PIFDumper::init(Simulator *simulator, CC3DXMLElement *_xmlData) {


    potts = simulator->getPotts();

    ostringstream numStream;
    string numString;

    numStream << simulator->getNumSteps();;

    numString = numStream.str();

    numDigits = numString.size();
    typePlug = (CellTypePlugin * )(Simulator::pluginManager.get("CellType"));

    simulator->registerSteerableObject(this);

    update(_xmlData, true);
}

void PIFDumper::step(const unsigned int currentStep) {

    ostringstream fullNameStr;
    fullNameStr << pifname;
    fullNameStr.width(numDigits);
    fullNameStr.fill('0');
    fullNameStr << currentStep << "." << pifFileExtension;
    std::string output_directory = simulator->getOutputDirectory();
    std::string abs_path(fullNameStr.str());

    if (output_directory.size()) {
        abs_path = output_directory + "/" + fullNameStr.str();
    }


    ofstream pif(abs_path.c_str());
    if (!pif.is_open()) throw CC3DException("Could not open file: " + abs_path + " for writing");


    WatchableField3D < CellG * > *cellFieldG = (WatchableField3D < CellG * > *)
    potts->getCellFieldG();
    Dim3D dim = cellFieldG->getDim();
    Point3D pt;
    CellG *cell;

    for (int x = 0; x < dim.x; ++x)
        for (int y = 0; y < dim.y; ++y)
            for (int z = 0; z < dim.z; ++z) {
                pt.x = x;
                pt.y = y;
                pt.z = z;
                cell = cellFieldG->get(pt);
                if (cell) {
                    pif << cell->id << "\t";
                    pif << typePlug->getTypeName(cell->type) << "\t";
                    pif << pt.x << "\t" << pt.x << "\t";
                    pif << pt.y << "\t" << pt.y << "\t";
                    pif << pt.z << "\t" << pt.z << "\t";
                    pif << endl;
                }
            }
}


void PIFDumper::start() {


}


void PIFDumper::update(CC3DXMLElement *_xmlData, bool _fullInitFlag) {

    //Check how this is handled

    pifname = _xmlData->getFirstElement("PIFName")->getText();


    if (_xmlData->findElement("PIFFileExtension"))
        pifFileExtension = _xmlData->getFirstElement("PIFFileExtension")->getText();

    //frequency=1;
}

std::string PIFDumper::toString() {
    return "PIFDumper";

}


std::string PIFDumper::steerableName() {
    return toString();

}

