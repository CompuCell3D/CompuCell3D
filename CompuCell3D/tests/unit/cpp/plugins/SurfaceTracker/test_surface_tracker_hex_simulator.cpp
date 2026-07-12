#include <gtest/gtest.h>

#include <CompuCell3D/Boundary/BoundaryStrategy.h>
#include <CompuCell3D/plugins/SurfaceTracker/SurfaceTrackerPlugin.h>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

namespace {
    double surfaceMF() {
        return BoundaryStrategy::getInstance()->getLatticeMultiplicativeFactors().surfaceMF;
    }

    double recomputeSurface(WatchableField3D<CellG *> *cellField, CellG *cell, unsigned int neighborOrder) {
        BoundaryStrategy *boundaryStrategy = BoundaryStrategy::getInstance();
        const double mf = boundaryStrategy->getLatticeMultiplicativeFactors().surfaceMF;
        const unsigned int maxNeighborIndex = boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(neighborOrder);
        const Dim3D dim = cellField->getDim();

        double surface = 0.0;
        for (int x = 0; x < dim.x; ++x) {
            for (int y = 0; y < dim.y; ++y) {
                for (int z = 0; z < dim.z; ++z) {
                    const Point3D pt(x, y, z);
                    if (cellField->get(pt) != cell) {
                        continue;
                    }

                    for (unsigned int nIdx = 0; nIdx <= maxNeighborIndex; ++nIdx) {
                        Neighbor neighbor = boundaryStrategy->getNeighborDirect(const_cast<Point3D &>(pt), nIdx);
                        if (!neighbor.distance) {
                            continue;
                        }
                        if (cellField->get(neighbor.pt) != cell) {
                            surface += mf;
                        }
                    }
                }
            }
        }

        return surface;
    }

    void expectTrackedSurface(WatchableField3D<CellG *> *cellField, CellG *cell, unsigned int neighborOrder) {
        ASSERT_NE(cell, nullptr);
        EXPECT_NEAR(cell->surface, recomputeSurface(cellField, cell, neighborOrder), 1.0e-12);
    }

    void configureHexSimulator(CC3DTestSimulator &testSim, const Dim3D &dim, unsigned int neighborOrder) {
        testSim.addPottsData(dim, neighborOrder, 10.0, 1, "Hexagonal");
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        CC3DXMLElement *surfaceTrackerData = testSim.addPluginData("SurfaceTracker");
        surfaceTrackerData->attachElement("NeighborOrder", std::to_string(neighborOrder));
        testSim.initializeSimulator();
    }
}

TEST(HexSurfaceTrackerPluginShapeTest, TracksSinglePixelSurfaceIn2D) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 1);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *cell = testSim.createCell(Point3D(10, 10, 0), 1);

    EXPECT_DOUBLE_EQ(cell->surface, 6.0 * surfaceMF());
    expectTrackedSurface(testSim.getCellField(), cell, 1);
}

TEST(HexSurfaceTrackerPluginShapeTest, TracksTwoPixelSurfaceIn2D) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 1);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *cell = testSim.createCell(Point3D(10, 10, 0), 1, {Point3D(11, 10, 0)});

    EXPECT_DOUBLE_EQ(cell->surface, 10.0 * surfaceMF());
    expectTrackedSurface(testSim.getCellField(), cell, 1);
}

TEST(HexSurfaceTrackerPluginShapeTest, TracksTwoPixelSurfaceIn2DNeighborOrder2) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 2);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *cell = testSim.createCell(Point3D(10, 10, 0), 1, {Point3D(11, 10, 0)});

    expectTrackedSurface(testSim.getCellField(), cell, 2);
}

TEST(HexSurfaceTrackerPluginShapeTest, TracksTwoPixelSurfaceIn2DNeighborOrder3) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 3);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *cell = testSim.createCell(Point3D(10, 10, 0), 1, {Point3D(11, 10, 0)});

    expectTrackedSurface(testSim.getCellField(), cell, 3);
}

TEST(HexSurfaceTrackerPluginShapeTest, TracksSingleVoxelSurfaceIn3D) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 18), 1);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *cell = testSim.createCell(Point3D(10, 10, 9), 1);

    EXPECT_DOUBLE_EQ(cell->surface, 12.0 * surfaceMF());
    expectTrackedSurface(testSim.getCellField(), cell, 1);
}

TEST(HexSurfaceTrackerPluginShapeTest, TracksTwoVoxelSurfaceIn3D) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 18), 1);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *cell = testSim.createCell(Point3D(10, 10, 9), 1, {Point3D(11, 10, 9)});

    EXPECT_DOUBLE_EQ(cell->surface, 22.0 * surfaceMF());
    expectTrackedSurface(testSim.getCellField(), cell, 1);
}

TEST(HexSurfaceTrackerPluginSimulatorTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixel2D) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 1);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1, {
            Point3D(10, 11, 0),
            Point3D(11, 10, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2, {
            Point3D(13, 11, 0)
    });

    expectTrackedSurface(testSim.getCellField(), newCell, 1);
    expectTrackedSurface(testSim.getCellField(), oldCell, 1);

    testSim.getCellField()->set(Point3D(13, 10, 0), newCell);

    expectTrackedSurface(testSim.getCellField(), newCell, 1);
    expectTrackedSurface(testSim.getCellField(), oldCell, 1);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(13, 10, 0)), newCell);
}

TEST(HexSurfaceTrackerPluginSimulatorTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesVoxel3D) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 18), 1);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *newCell = testSim.createCell(Point3D(10, 10, 9), 1, {
            Point3D(10, 10, 10),
            Point3D(11, 10, 9)
    });
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 9), 2, {
            Point3D(13, 10, 10)
    });

    expectTrackedSurface(testSim.getCellField(), newCell, 1);
    expectTrackedSurface(testSim.getCellField(), oldCell, 1);

    testSim.getCellField()->set(Point3D(13, 10, 9), newCell);

    expectTrackedSurface(testSim.getCellField(), newCell, 1);
    expectTrackedSurface(testSim.getCellField(), oldCell, 1);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(13, 10, 9)), newCell);
}

TEST(HexSurfaceTrackerPluginHigherOrderTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixelNeighborOrder2) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 2);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1, {
            Point3D(10, 11, 0),
            Point3D(11, 10, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2, {
            Point3D(13, 11, 0)
    });

    expectTrackedSurface(testSim.getCellField(), newCell, 2);
    expectTrackedSurface(testSim.getCellField(), oldCell, 2);

    testSim.getCellField()->set(Point3D(13, 10, 0), newCell);

    expectTrackedSurface(testSim.getCellField(), newCell, 2);
    expectTrackedSurface(testSim.getCellField(), oldCell, 2);
}

TEST(HexSurfaceTrackerPluginHigherOrderTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixelNeighborOrder3) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 3);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1, {
            Point3D(10, 11, 0),
            Point3D(11, 10, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2, {
            Point3D(13, 11, 0)
    });

    expectTrackedSurface(testSim.getCellField(), newCell, 3);
    expectTrackedSurface(testSim.getCellField(), oldCell, 3);

    testSim.getCellField()->set(Point3D(13, 10, 0), newCell);

    expectTrackedSurface(testSim.getCellField(), newCell, 3);
    expectTrackedSurface(testSim.getCellField(), oldCell, 3);
}

TEST(HexSurfaceTrackerPluginHigherOrderTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixelNeighborOrder4) {
    CC3DTestSimulator testSim;
    configureHexSimulator(testSim, Dim3D(20, 20, 1), 4);

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *newCell = testSim.createCell(Point3D(10, 10, 0), 1, {
            Point3D(10, 11, 0),
            Point3D(11, 10, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(13, 10, 0), 2, {
            Point3D(13, 11, 0)
    });

    expectTrackedSurface(testSim.getCellField(), newCell, 4);
    expectTrackedSurface(testSim.getCellField(), oldCell, 4);

    testSim.getCellField()->set(Point3D(13, 10, 0), newCell);

    expectTrackedSurface(testSim.getCellField(), newCell, 4);
    expectTrackedSurface(testSim.getCellField(), oldCell, 4);
}
