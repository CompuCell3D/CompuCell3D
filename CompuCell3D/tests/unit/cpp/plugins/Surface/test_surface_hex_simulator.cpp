#include <gtest/gtest.h>

#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/plugins/Surface/SurfacePlugin.h>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

namespace {
    void configureHexSurfaceSimulator(CC3DTestSimulator &testSim, const Dim3D &dim) {
        testSim.addPottsData(dim, 1, 10.0, 1, "Hexagonal");
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        testSim.addPluginData("Surface");
        testSim.initializeSimulator();
    }

    CellG *occupantAt(WatchableField3D<CellG *> *cellField, const Point3D &queryPt, const Point3D &copyPt, CellG *newCell) {
        return queryPt == copyPt ? newCell : cellField->get(queryPt);
    }

    double recomputeSurfaceAfterCopy(WatchableField3D<CellG *> *cellField,
                                     CellG *cell,
                                     const Point3D &copyPt,
                                     CellG *newCell,
                                     unsigned int neighborOrder = 1) {
        if (!cell) {
            return 0.0;
        }

        BoundaryStrategy *boundaryStrategy = BoundaryStrategy::getInstance();
        const double surfaceMF = boundaryStrategy->getLatticeMultiplicativeFactors().surfaceMF;
        const unsigned int maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);
        const Dim3D dim = cellField->getDim();

        double surface = 0.0;
        for (int x = 0; x < dim.x; ++x) {
            for (int y = 0; y < dim.y; ++y) {
                for (int z = 0; z < dim.z; ++z) {
                    Point3D pt(x, y, z);
                    if (occupantAt(cellField, pt, copyPt, newCell) != cell) {
                        continue;
                    }

                    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
                        Neighbor neighbor = boundaryStrategy->getNeighborDirect(pt, nIdx);
                        if (!neighbor.distance) {
                            continue;
                        }
                        if (occupantAt(cellField, neighbor.pt, copyPt, newCell) != cell) {
                            surface += surfaceMF;
                        }
                    }
                }
            }
        }

        return surface;
    }

    double diffEnergy(double lambdaSurface, double targetSurface, double surface, double diff) {
        return lambdaSurface * (diff * diff + 2.0 * diff * (surface - std::fabs(targetSurface)));
    }
}

class HexSurfacePluginSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        configureHexSurfaceSimulator(testSim, Dim3D(20, 20, 1));
        surfacePlugin = testSim.getPlugin<SurfacePlugin>("Surface");
        ASSERT_NE(surfacePlugin, nullptr);
    }

    CC3DTestSimulator testSim;
    SurfacePlugin *surfacePlugin = nullptr;
};

TEST_F(HexSurfacePluginSimulatorTest, ReturnsZeroWhenCopyDoesNotChangeCellIdentity) {
    // Copy source and target belong to the same cell, so the surface term must not change.
    CellG *cell = testSim.createCell(Point3D(10, 10, 0), 1, {Point3D(11, 10, 0)});
    cell->targetSurface = static_cast<float>(cell->surface);
    cell->lambdaSurface = 2.0f;

    EXPECT_NEAR(surfacePlugin->changeEnergy(Point3D(10, 10, 0), cell, cell), 0.0, 1.0e-12);
}

TEST_F(HexSurfacePluginSimulatorTest, ComputesEnergyWhenOneCellGainsAndOneCellLosesPixel) {
    // Cell A copies into a pixel occupied by Cell B on the 2D hex lattice.
    CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1, {
            Point3D(10, 11, 0),
            Point3D(11, 10, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2, {
            Point3D(13, 11, 0)
    });

    newCell->targetSurface = static_cast<float>(newCell->surface + 3.0);
    newCell->lambdaSurface = 2.0f;
    oldCell->targetSurface = static_cast<float>(oldCell->surface - 2.0);
    oldCell->lambdaSurface = 1.5f;

    const Point3D copyPt(13, 10, 0);
    const double newAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), newCell, copyPt, newCell);
    const double oldAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), oldCell, copyPt, newCell);
    const double expectedNew = diffEnergy(newCell->lambdaSurface, newCell->targetSurface, newCell->surface,
                                          newAfter - newCell->surface);
    const double expectedOld = diffEnergy(oldCell->lambdaSurface, oldCell->targetSurface, oldCell->surface,
                                          oldAfter - oldCell->surface);

    EXPECT_NEAR(surfacePlugin->changeEnergy(copyPt, newCell, oldCell), expectedNew + expectedOld, 1.0e-12);
}

TEST_F(HexSurfacePluginSimulatorTest, HandlesGainFromMedium) {
    // A multi-pixel cell expands into medium on the 2D hex lattice.
    CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1, {
            Point3D(10, 11, 0),
            Point3D(11, 10, 0)
    });
    newCell->targetSurface = static_cast<float>(newCell->surface + 4.0);
    newCell->lambdaSurface = 1.25f;

    const Point3D copyPt(12, 10, 0);
    const double newAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), newCell, copyPt, newCell);
    const double expected = diffEnergy(newCell->lambdaSurface, newCell->targetSurface, newCell->surface,
                                       newAfter - newCell->surface);

    EXPECT_NEAR(surfacePlugin->changeEnergy(copyPt, newCell, nullptr), expected, 1.0e-12);
}

TEST_F(HexSurfacePluginSimulatorTest, HandlesSinglePixelCellGainFromMedium) {
    // A one-pixel cell expands into medium on the 2D hex lattice.
    CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1);
    newCell->targetSurface = static_cast<float>(newCell->surface);
    newCell->lambdaSurface = 1.25f;

    const Point3D copyPt(11, 10, 0);
    const double newAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), newCell, copyPt, newCell);
    const double expected = diffEnergy(newCell->lambdaSurface, newCell->targetSurface, newCell->surface,
                                       newAfter - newCell->surface);

    EXPECT_NEAR(surfacePlugin->changeEnergy(copyPt, newCell, nullptr), expected, 1.0e-12);
}

TEST_F(HexSurfacePluginSimulatorTest, HandlesLossToMedium) {
    // Medium overwrites one pixel of a multi-pixel cell on the 2D hex lattice.
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2, {
            Point3D(13, 11, 0),
            Point3D(12, 10, 0)
    });
    oldCell->targetSurface = static_cast<float>(oldCell->surface - 3.0);
    oldCell->lambdaSurface = 0.5f;

    const Point3D copyPt(13, 10, 0);
    const double oldAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), oldCell, copyPt, nullptr);
    const double expected = diffEnergy(oldCell->lambdaSurface, oldCell->targetSurface, oldCell->surface,
                                       oldAfter - oldCell->surface);

    EXPECT_NEAR(surfacePlugin->changeEnergy(copyPt, nullptr, oldCell), expected, 1.0e-12);
}

TEST_F(HexSurfacePluginSimulatorTest, HandlesSinglePixelCellLossToMedium) {
    // A one-pixel cell is overwritten by medium on the 2D hex lattice.
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2);
    oldCell->targetSurface = static_cast<float>(oldCell->surface - 1.0);
    oldCell->lambdaSurface = 0.5f;

    const Point3D copyPt(13, 10, 0);
    const double oldAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), oldCell, copyPt, nullptr);
    const double expected = diffEnergy(oldCell->lambdaSurface, oldCell->targetSurface, oldCell->surface,
                                       oldAfter - oldCell->surface);

    EXPECT_NEAR(surfacePlugin->changeEnergy(copyPt, nullptr, oldCell), expected, 1.0e-12);
}

TEST(HexSurfacePluginSimulator3DTest, ComputesEnergyWhenOneCellGainsAndOneCellLosesVoxel) {
    CC3DTestSimulator testSim;
    configureHexSurfaceSimulator(testSim, Dim3D(20, 20, 18));

    SurfacePlugin *surfacePlugin = testSim.getPlugin<SurfacePlugin>("Surface");
    ASSERT_NE(surfacePlugin, nullptr);

    CellG *newCell = testSim.createCell(Point3D(10, 10, 9), 1, {
            Point3D(10, 10, 10),
            Point3D(11, 10, 9)
    });
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 9), 2, {
            Point3D(13, 10, 10)
    });

    newCell->targetSurface = static_cast<float>(newCell->surface + 2.0);
    newCell->lambdaSurface = 1.75f;
    oldCell->targetSurface = static_cast<float>(oldCell->surface - 1.0);
    oldCell->lambdaSurface = 0.75f;

    const Point3D copyPt(13, 10, 9);
    const double newAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), newCell, copyPt, newCell);
    const double oldAfter = recomputeSurfaceAfterCopy(testSim.getCellField(), oldCell, copyPt, newCell);
    const double expectedNew = diffEnergy(newCell->lambdaSurface, newCell->targetSurface, newCell->surface,
                                          newAfter - newCell->surface);
    const double expectedOld = diffEnergy(oldCell->lambdaSurface, oldCell->targetSurface, oldCell->surface,
                                          oldAfter - oldCell->surface);

    EXPECT_NEAR(surfacePlugin->changeEnergy(copyPt, newCell, oldCell), expectedNew + expectedOld, 1.0e-12);
}
