#include <gtest/gtest.h>

#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/plugins/Contact/ContactPlugin.h>

#include <map>
#include <string>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

namespace {
    using ContactKey = std::pair<unsigned char, unsigned char>;

    ContactKey makeKey(unsigned char lhs, unsigned char rhs) {
        return lhs < rhs ? ContactKey(lhs, rhs) : ContactKey(rhs, lhs);
    }

    double contactEnergy(const std::map<ContactKey, double> &contactTable, const CellG *lhs, const CellG *rhs) {
        const unsigned char lhsType = lhs ? lhs->type : 0;
        const unsigned char rhsType = rhs ? rhs->type : 0;
        return contactTable.at(makeKey(lhsType, rhsType));
    }

    double expectedContactEnergy(WatchableField3D<CellG *> *cellField,
                                 const std::map<ContactKey, double> &contactTable,
                                 const Point3D &pt,
                                 const CellG *newCell,
                                 const CellG *oldCell,
                                 unsigned int neighborOrder) {
        BoundaryStrategy *boundaryStrategy = BoundaryStrategy::getInstance();
        const unsigned int maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);

        double energy = 0.0;
        for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
            Neighbor neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
            if (!neighbor.distance) {
                continue;
            }

            CellG *nCell = cellField->get(neighbor.pt);
            if (nCell != oldCell) {
                if (nCell != nullptr && oldCell != nullptr) {
                    if (nCell->clusterId != oldCell->clusterId) {
                        energy -= contactEnergy(contactTable, oldCell, nCell);
                    }
                } else {
                    energy -= contactEnergy(contactTable, oldCell, nCell);
                }
            }

            if (nCell != newCell) {
                if (newCell != nullptr && nCell != nullptr) {
                    if (newCell->clusterId != nCell->clusterId) {
                        energy += contactEnergy(contactTable, newCell, nCell);
                    }
                } else {
                    energy += contactEnergy(contactTable, newCell, nCell);
                }
            }
        }

        return energy;
    }

    std::map<ContactKey, double> addContactPlugin(CC3DTestSimulator &testSim, unsigned int neighborOrder) {
        CC3DXMLElement *contactData = testSim.addPluginData("Contact");
        contactData->attachElement("NeighborOrder", std::to_string(neighborOrder));

        struct EnergySpec {
            const char *type1;
            const char *type2;
            double value;
        };

        const EnergySpec specs[] = {
                {"Medium", "Medium", 0.0},
                {"Medium", "A",      12.0},
                {"Medium", "B",      8.0},
                {"A",      "A",      2.0},
                {"A",      "B",      16.0},
                {"B",      "B",      4.0},
        };

        std::map<ContactKey, double> contactTable;
        for (const auto &spec: specs) {
            CC3DXMLElement *energy = contactData->attachElement("Energy", std::to_string(spec.value));
            energy->attachAttribute("Type1", spec.type1);
            energy->attachAttribute("Type2", spec.type2);

            const unsigned char lhsType =
                    std::string(spec.type1) == "Medium" ? 0 : (std::string(spec.type1) == "A" ? 1 : 2);
            const unsigned char rhsType =
                    std::string(spec.type2) == "Medium" ? 0 : (std::string(spec.type2) == "A" ? 1 : 2);
            contactTable[makeKey(lhsType, rhsType)] = spec.value;
        }

        return contactTable;
    }

    void configureHexSimulator(CC3DTestSimulator &testSim, const Dim3D &dim, unsigned int neighborOrder,
                               std::map<ContactKey, double> &contactTable) {
        testSim.addPottsData(dim, neighborOrder, 10.0, 1, "Hexagonal");
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        contactTable = addContactPlugin(testSim, neighborOrder);
        testSim.initializeSimulator();
    }
}

TEST(ContactPluginHex2DTest, ComputesEnergyAcrossNeighborOrders) {
    const unsigned int orders[] = {1, 2, 3, 4};

    for (unsigned int neighborOrder: orders) {
        SCOPED_TRACE(::testing::Message() << "neighborOrder=" << neighborOrder);

        CC3DTestSimulator testSim;
        std::map<ContactKey, double> contactTable;
        configureHexSimulator(testSim, Dim3D(20, 20, 1), neighborOrder, contactTable);

        ContactPlugin *contactPlugin = testSim.getPlugin<ContactPlugin>("Contact");
        ASSERT_NE(contactPlugin, nullptr);

        CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1, {
                Point3D(10, 11, 0),
                Point3D(11, 10, 0),
                Point3D(11, 11, 0)
        });
        CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2, {
                Point3D(13, 11, 0),
                Point3D(14, 10, 0)
        });

        const Point3D copyPt(13, 10, 0);
        const double expected = expectedContactEnergy(testSim.getCellField(), contactTable, copyPt, newCell, oldCell,
                                                      neighborOrder);

        EXPECT_DOUBLE_EQ(contactPlugin->changeEnergy(copyPt, newCell, oldCell), expected);
    }
}

TEST(ContactPluginHex3DTest, ComputesEnergyAcrossNeighborOrders) {
    const unsigned int orders[] = {1, 2, 3, 4};

    for (unsigned int neighborOrder: orders) {
        SCOPED_TRACE(::testing::Message() << "neighborOrder=" << neighborOrder);

        CC3DTestSimulator testSim;
        std::map<ContactKey, double> contactTable;
        configureHexSimulator(testSim, Dim3D(20, 20, 18), neighborOrder, contactTable);

        ContactPlugin *contactPlugin = testSim.getPlugin<ContactPlugin>("Contact");
        ASSERT_NE(contactPlugin, nullptr);

        CellG *newCell = testSim.createCell(Point3D(10, 10, 9), 1, {
                Point3D(10, 10, 10),
                Point3D(11, 10, 9),
                Point3D(10, 11, 9)
        });
        CellG *oldCell = testSim.createCell(Point3D(13, 10, 9), 2, {
                Point3D(13, 10, 10),
                Point3D(14, 10, 9)
        });

        const Point3D copyPt(13, 10, 9);
        const double expected = expectedContactEnergy(testSim.getCellField(), contactTable, copyPt, newCell, oldCell,
                                                      neighborOrder);

        EXPECT_DOUBLE_EQ(contactPlugin->changeEnergy(copyPt, newCell, oldCell), expected);
    }
}
