#include "CC3DTestSimulator.h"

#include <CompuCell3D/Plugin.h>

#include <sstream>
#include <stdexcept>

using namespace CompuCell3D;

CC3DTestSimulator::CC3DTestSimulator() : simulator(new Simulator()) {}

CC3DTestSimulator::~CC3DTestSimulator() = default;

CC3DXMLElement *CC3DTestSimulator::createElement(const std::string &name,
                                                 const std::vector<std::pair<std::string, std::string>> &attributes,
                                                 const std::string &cdata) {
    std::map<std::string, std::string> attrMap(attributes.begin(), attributes.end());
    ownedElements.emplace_back(new CC3DXMLElement(name, attrMap, cdata));
    return ownedElements.back().get();
}

std::string CC3DTestSimulator::toString(int value) {
    return std::to_string(value);
}

std::string CC3DTestSimulator::toString(unsigned int value) {
    return std::to_string(value);
}

std::string CC3DTestSimulator::toString(double value) {
    std::ostringstream out;
    out << value;
    return out.str();
}

CC3DXMLElement *CC3DTestSimulator::addPottsData(const Dim3D &dim, unsigned int neighborOrder, double temperature,
                                                unsigned int steps, const std::string &latticeType,
                                                const std::string &dimensionType) {
    CC3DXMLElement *pottsData = createElement("Potts");
    CC3DXMLElement *dimensions = pottsData->attachElement("Dimensions");
    dimensions->attachAttribute("x", toString(dim.x));
    dimensions->attachAttribute("y", toString(dim.y));
    dimensions->attachAttribute("z", toString(dim.z));
    pottsData->attachElement("Temperature", toString(temperature));
    pottsData->attachElement("Steps", toString(steps));
    pottsData->attachElement("NeighborOrder", toString(neighborOrder));
    if (!latticeType.empty()) {
        pottsData->attachElement("LatticeType", latticeType);
    }
    if (!dimensionType.empty()) {
        pottsData->attachElement("DimensionType", dimensionType);
    }

    simulator->ps.addPottsDataCC3D(pottsData);
    return pottsData;
}

CC3DXMLElement *CC3DTestSimulator::addPluginData(const std::string &name) {
    CC3DXMLElement *pluginData = createElement("Plugin");
    pluginData->attachAttribute("Name", name);
    simulator->ps.addPluginDataCC3D(pluginData);
    return pluginData;
}

CC3DXMLElement *CC3DTestSimulator::addCellTypePluginData(
        const std::vector<std::pair<std::string, unsigned char>> &types) {
    CC3DXMLElement *pluginData = addPluginData("CellType");

    for (const auto &typeSpec : types) {
        CC3DXMLElement *cellType = pluginData->attachElement("CellType");
        cellType->attachAttribute("TypeName", typeSpec.first);
        cellType->attachAttribute("TypeId", toString(static_cast<unsigned int>(typeSpec.second)));
    }

    return pluginData;
}

void CC3DTestSimulator::initializeSimulator() {
    simulator->initializeCC3D();
    simulator->extraInit();
}

CellG *CC3DTestSimulator::createCell(const Point3D &seed, unsigned char type,
                                     const std::vector<Point3D> &additionalPixels) {
    CellG *cell = simulator->getPotts()->createCellG(seed);
    cell->type = type;

    for (const auto &pixel : additionalPixels) {
        if (pixel == seed) {
            continue;
        }
        getCellField()->set(pixel, cell);
    }

    return cell;
}
