#include "YamlConformanceLoader.h"

#include <CompuCell3D/Field3D/WatchableField3D.h>
#include <CompuCell3D/plugins/Volume/VolumePlugin.h>

#include <cmath>
#include <exception>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

namespace {
    Point3D toPoint3D(const ConformancePoint &point) {
        return Point3D(point.x, point.y, point.z);
    }

    std::vector<std::pair<std::string, unsigned char>> buildTypeTable(const ConformanceCase &testCase) {
        std::vector<std::pair<std::string, unsigned char>> typeTable;
        typeTable.emplace_back(testCase.mediumType, 0);

        std::map<std::string, unsigned char> assignedIds;
        assignedIds[testCase.mediumType] = 0;
        unsigned char nextId = 1;

        for (const auto &cell: testCase.cells) {
            if (assignedIds.find(cell.type) != assignedIds.end()) {
                continue;
            }
            assignedIds[cell.type] = nextId;
            typeTable.emplace_back(cell.type, nextId);
            ++nextId;
        }

        return typeTable;
    }

    std::map<std::string, unsigned char> makeTypeIdMap(const std::vector<std::pair<std::string, unsigned char>> &typeTable) {
        std::map<std::string, unsigned char> typeIdMap;
        for (const auto &entry: typeTable) {
            typeIdMap[entry.first] = entry.second;
        }
        return typeIdMap;
    }

    bool almostEqual(double lhs, double rhs, double tolerance) {
        return std::fabs(lhs - rhs) <= tolerance;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: cc3d_conformance_runner <case.yaml>" << std::endl;
        return 2;
    }

    try {
        const ConformanceCase testCase = loadConformanceCase(argv[1]);

        CC3DTestSimulator testSim;
        testSim.addPottsData(
                Dim3D(testCase.lattice.dim.x, testCase.lattice.dim.y, testCase.lattice.dim.z),
                testCase.lattice.neighborhoodOrder
        );

        const auto typeTable = buildTypeTable(testCase);
        const auto typeIdMap = makeTypeIdMap(typeTable);
        testSim.addCellTypePluginData(typeTable);
        testSim.addPluginData("Volume");
        testSim.initializeSimulator();

        VolumePlugin *volumePlugin = testSim.getPlugin<VolumePlugin>("Volume");
        if (!volumePlugin) {
            throw std::runtime_error("failed to initialize Volume plugin");
        }

        for (const auto &cellSpec: testCase.cells) {
            const auto typeIdItr = typeIdMap.find(cellSpec.type);
            if (typeIdItr == typeIdMap.end()) {
                throw std::runtime_error("unknown cell type '" + cellSpec.type + "'");
            }

            std::vector<Point3D> additionalPixels;
            additionalPixels.reserve(cellSpec.pixels.size() > 0 ? cellSpec.pixels.size() - 1 : 0);
            for (std::size_t i = 1; i < cellSpec.pixels.size(); ++i) {
                additionalPixels.push_back(toPoint3D(cellSpec.pixels[i]));
            }

            CellG *cell = testSim.createCell(toPoint3D(cellSpec.pixels.front()), typeIdItr->second, additionalPixels);
            cell->targetVolume = static_cast<float>(cellSpec.targetVolume);
            cell->lambdaVolume = static_cast<float>(cellSpec.lambdaVolume);
        }

        WatchableField3D<CellG *> *cellField = testSim.getCellField();
        for (const auto &query: testCase.volumeQueries) {
            const Point3D sourcePoint = toPoint3D(query.source);
            const Point3D targetPoint = toPoint3D(query.target);
            CellG *newCell = cellField->get(sourcePoint);
            CellG *oldCell = cellField->get(targetPoint);

            const double actual = volumePlugin->changeEnergy(targetPoint, newCell, oldCell);
            if (!almostEqual(actual, query.expected.value, query.expected.tolerance)) {
                std::cerr
                        << testCase.name
                        << " query " << query.id
                        << " expected " << query.expected.value
                        << " +/- " << query.expected.tolerance
                        << " but got " << actual
                        << std::endl;
                return 1;
            }
        }

        std::cout << "PASS " << testCase.name << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "cc3d_conformance_runner: " << e.what() << std::endl;
        return 1;
    }
}
