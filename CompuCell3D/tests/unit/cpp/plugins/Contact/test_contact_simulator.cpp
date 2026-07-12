#include <gtest/gtest.h>

#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/plugins/Contact/ContactPlugin.h>

#include <map>
#include <string>
#include <tuple>

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

    void configureSquareSimulator(CC3DTestSimulator &testSim, const Dim3D &dim, unsigned int neighborOrder,
                                  std::map<ContactKey, double> &contactTable) {
        testSim.addPottsData(dim, neighborOrder);
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        contactTable = addContactPlugin(testSim, neighborOrder);
        testSim.initializeSimulator();
    }
}

TEST(ContactPluginSquare2DTest, ComputesEnergyAcrossNeighborOrders) {
    const unsigned int orders[] = {1, 2, 3, 4};

    for (unsigned int neighborOrder: orders) {
        SCOPED_TRACE(::testing::Message() << "neighborOrder=" << neighborOrder);

        CC3DTestSimulator testSim;
        std::map<ContactKey, double> contactTable;
        configureSquareSimulator(testSim, Dim3D(20, 20, 1), neighborOrder, contactTable);

        ContactPlugin *contactPlugin = testSim.getPlugin<ContactPlugin>("Contact");
        ASSERT_NE(contactPlugin, nullptr);

        CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
                Point3D(1, 2, 0),
                Point3D(2, 1, 0),
                Point3D(2, 2, 0)
        });
        CellG *oldCell = testSim.createCell(Point3D(5, 1, 0), 2, {
                Point3D(5, 2, 0),
                Point3D(6, 1, 0)
        });

        const Point3D copyPt(5, 1, 0);
        const double expected = expectedContactEnergy(testSim.getCellField(), contactTable, copyPt, newCell, oldCell,
                                                      neighborOrder);

        EXPECT_DOUBLE_EQ(contactPlugin->changeEnergy(copyPt, newCell, oldCell), expected);
    }
}

TEST(ContactPluginSquare2DTest, HandlesGainFromMediumAcrossNeighborOrders) {
    const unsigned int orders[] = {1, 2, 3, 4};

    for (unsigned int neighborOrder: orders) {
        SCOPED_TRACE(::testing::Message() << "neighborOrder=" << neighborOrder);

        CC3DTestSimulator testSim;
        std::map<ContactKey, double> contactTable;
        configureSquareSimulator(testSim, Dim3D(20, 20, 1), neighborOrder, contactTable);

        ContactPlugin *contactPlugin = testSim.getPlugin<ContactPlugin>("Contact");
        ASSERT_NE(contactPlugin, nullptr);

        CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
                Point3D(1, 2, 0),
                Point3D(2, 1, 0),
                Point3D(2, 2, 0)
        });

        const Point3D copyPt(3, 1, 0);
        const double expected = expectedContactEnergy(testSim.getCellField(), contactTable, copyPt, newCell, nullptr,
                                                      neighborOrder);

        EXPECT_DOUBLE_EQ(contactPlugin->changeEnergy(copyPt, newCell, nullptr), expected);
    }
}

TEST(ContactPluginSquare2DTest, HandlesLossToMediumAcrossNeighborOrders) {
    const unsigned int orders[] = {1, 2, 3, 4};

    for (unsigned int neighborOrder: orders) {
        SCOPED_TRACE(::testing::Message() << "neighborOrder=" << neighborOrder);

        CC3DTestSimulator testSim;
        std::map<ContactKey, double> contactTable;
        configureSquareSimulator(testSim, Dim3D(20, 20, 1), neighborOrder, contactTable);

        ContactPlugin *contactPlugin = testSim.getPlugin<ContactPlugin>("Contact");
        ASSERT_NE(contactPlugin, nullptr);

        CellG *oldCell = testSim.createCell(Point3D(5, 1, 0), 2, {
                Point3D(5, 2, 0),
                Point3D(4, 1, 0)
        });

        const Point3D copyPt(5, 1, 0);
        const double expected = expectedContactEnergy(testSim.getCellField(), contactTable, copyPt, nullptr, oldCell,
                                                      neighborOrder);

        EXPECT_DOUBLE_EQ(contactPlugin->changeEnergy(copyPt, nullptr, oldCell), expected);
    }
}

TEST(ContactPluginSquare3DTest, ComputesEnergyAcrossNeighborOrders) {
    const unsigned int orders[] = {1, 2, 3, 4};

    for (unsigned int neighborOrder: orders) {
        SCOPED_TRACE(::testing::Message() << "neighborOrder=" << neighborOrder);

        CC3DTestSimulator testSim;
        std::map<ContactKey, double> contactTable;
        configureSquareSimulator(testSim, Dim3D(20, 20, 20), neighborOrder, contactTable);

        ContactPlugin *contactPlugin = testSim.getPlugin<ContactPlugin>("Contact");
        ASSERT_NE(contactPlugin, nullptr);

        CellG *newCell = testSim.createCell(Point3D(1, 1, 1), 1, {
                Point3D(1, 1, 2),
                Point3D(1, 2, 1),
                Point3D(2, 1, 1)
        });
        CellG *oldCell = testSim.createCell(Point3D(5, 1, 1), 2, {
                Point3D(5, 1, 2),
                Point3D(6, 1, 1)
        });

        const Point3D copyPt(5, 1, 1);
        const double expected = expectedContactEnergy(testSim.getCellField(), contactTable, copyPt, newCell, oldCell,
                                                      neighborOrder);

        EXPECT_DOUBLE_EQ(contactPlugin->changeEnergy(copyPt, newCell, oldCell), expected);
    }
}
