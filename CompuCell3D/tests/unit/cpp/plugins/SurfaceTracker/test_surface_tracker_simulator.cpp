#include <gtest/gtest.h>

#include <CompuCell3D/plugins/SurfaceTracker/SurfaceTrackerPlugin.h>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

class SurfaceTrackerPluginSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        testSim.addPottsData(Dim3D(20, 20, 1));
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        testSim.addPluginData("SurfaceTracker");
        testSim.initializeSimulator();
        surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
        ASSERT_NE(surfaceTrackerPlugin, nullptr);
    }

    CC3DTestSimulator testSim;
    SurfaceTrackerPlugin *surfaceTrackerPlugin = nullptr;
};

namespace {
    void runHigherOrderSurfaceTrackerScenario(unsigned int neighborOrder,
                                              double expectedNewSurfaceBefore,
                                              double expectedOldSurfaceBefore,
                                              double expectedNewSurfaceAfter,
                                              double expectedOldSurfaceAfter) {
        CC3DTestSimulator testSim;
        testSim.addPottsData(Dim3D(20, 20, 1));
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        CC3DXMLElement *surfaceTrackerData = testSim.addPluginData("SurfaceTracker");
        surfaceTrackerData->attachElement("NeighborOrder", std::to_string(neighborOrder));
        testSim.initializeSimulator();

        SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
        ASSERT_NE(surfaceTrackerPlugin, nullptr);

        CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
                Point3D(1, 2, 0),
                Point3D(2, 1, 0)
        });
        CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2, {
                Point3D(4, 2, 0)
        });

        ASSERT_DOUBLE_EQ(newCell->surface, expectedNewSurfaceBefore);
        ASSERT_DOUBLE_EQ(oldCell->surface, expectedOldSurfaceBefore);

        testSim.getCellField()->set(Point3D(4, 1, 0), newCell);

        EXPECT_DOUBLE_EQ(newCell->surface, expectedNewSurfaceAfter);
        EXPECT_DOUBLE_EQ(oldCell->surface, expectedOldSurfaceAfter);
        EXPECT_EQ(testSim.getCellField()->get(Point3D(4, 1, 0)), newCell);
    }

    void runIsolatedSurfaceShapeScenario(const Dim3D &dim,
                                         unsigned int neighborOrder,
                                         const std::vector<Point3D> &pixels,
                                         double expectedSurface) {
        CC3DTestSimulator testSim;
        testSim.addPottsData(dim, neighborOrder);
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                      });
        CC3DXMLElement *surfaceTrackerData = testSim.addPluginData("SurfaceTracker");
        surfaceTrackerData->attachElement("NeighborOrder", std::to_string(neighborOrder));
        testSim.initializeSimulator();

        SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
        ASSERT_NE(surfaceTrackerPlugin, nullptr);
        ASSERT_FALSE(pixels.empty());

        std::vector<Point3D> additionalPixels;
        for (std::size_t i = 1; i < pixels.size(); ++i) {
            additionalPixels.push_back(pixels[i]);
        }

        CellG *cell = testSim.createCell(pixels.front(), 1, additionalPixels);
        EXPECT_DOUBLE_EQ(cell->surface, expectedSurface);
    }
}

TEST_F(SurfaceTrackerPluginSimulatorTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixel) {
    // Cell A overwrites one pixel of Cell B. With neighbor order 1 on the square lattice,
    // A gains 4 surface units and B loses 2 surface units immediately.
    CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
            Point3D(1, 2, 0),
            Point3D(2, 1, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2, {
            Point3D(4, 2, 0)
    });

    ASSERT_DOUBLE_EQ(newCell->surface, 8.0);
    ASSERT_DOUBLE_EQ(oldCell->surface, 6.0);

    testSim.getCellField()->set(Point3D(4, 1, 0), newCell);

    EXPECT_DOUBLE_EQ(newCell->surface, 12.0);
    EXPECT_DOUBLE_EQ(oldCell->surface, 4.0);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(4, 1, 0)), newCell);
}

TEST(SurfaceTrackerPluginSimulator3DTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesVoxel) {
    // In 3D with neighbor order 1, a newly occupied isolated voxel contributes 6 faces to the
    // gaining cell, while removing one voxel from a two-voxel cell reduces that cell surface by 4.
    CC3DTestSimulator testSim;
    testSim.addPottsData(Dim3D(20, 20, 20));
    testSim.addCellTypePluginData({
                                          {"Medium", 0},
                                          {"A",      1},
                                          {"B",      2},
                                  });
    testSim.addPluginData("SurfaceTracker");
    testSim.initializeSimulator();

    SurfaceTrackerPlugin *surfaceTrackerPlugin = testSim.getPlugin<SurfaceTrackerPlugin>("SurfaceTracker");
    ASSERT_NE(surfaceTrackerPlugin, nullptr);

    CellG *newCell = testSim.createCell(Point3D(1, 1, 1), 1, {
            Point3D(1, 1, 2),
            Point3D(1, 2, 1)
    });
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 1), 2, {
            Point3D(4, 1, 2)
    });

    ASSERT_DOUBLE_EQ(newCell->surface, 14.0);
    ASSERT_DOUBLE_EQ(oldCell->surface, 10.0);

    testSim.getCellField()->set(Point3D(4, 1, 1), newCell);

    EXPECT_DOUBLE_EQ(newCell->surface, 20.0);
    EXPECT_DOUBLE_EQ(oldCell->surface, 6.0);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(4, 1, 1)), newCell);
}

TEST(SurfaceTrackerPluginHigherOrderTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixelNeighborOrder2) {
    // Order 2 adds diagonal interfaces. The copied pixel is still isolated from Cell A, so
    // A gains 8 surface units while B loses 6.
    runHigherOrderSurfaceTrackerScenario(
            2,
            18.0,
            14.0,
            26.0,
            8.0
    );
}

TEST(SurfaceTrackerPluginHigherOrderTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixelNeighborOrder3) {
    // Order 3 expands the stencil beyond diagonals. For this geometry, the tracked surfaces
    // become 26 -> 35 for Cell A and 21 -> 12 for Cell B.
    runHigherOrderSurfaceTrackerScenario(
            3,
            26.0,
            21.0,
            35.0,
            12.0
    );
}

TEST(SurfaceTrackerPluginHigherOrderTest, UpdatesSurfacesWhenOneCellGainsAndOneCellLosesPixelNeighborOrder4) {
    // Order 4 expands the stencil further. For this geometry, the tracked surfaces
    // become 42 -> 57 for Cell A and 35 -> 20 for Cell B.
    runHigherOrderSurfaceTrackerScenario(
            4,
            42.0,
            35.0,
            57.0,
            20.0
    );
}

TEST(SurfaceTrackerPluginShapeTest, TracksSinglePixelSurfaceIn2D) {
    // With neighbor order 1 in 2D, an isolated single-pixel cell has four medium interfaces.
    runIsolatedSurfaceShapeScenario(
            Dim3D(20, 20, 1),
            1,
            {Point3D(10, 10, 0)},
            4.0
    );
}

TEST(SurfaceTrackerPluginShapeTest, TracksTwoPixelSurfaceIn2D) {
    // Two edge-adjacent pixels in 2D share one interface, so the total tracked surface is 6.
    runIsolatedSurfaceShapeScenario(
            Dim3D(20, 20, 1),
            1,
            {Point3D(10, 10, 0), Point3D(11, 10, 0)},
            6.0
    );
}

TEST(SurfaceTrackerPluginShapeTest, TracksSingleVoxelSurfaceIn3D) {
    // With neighbor order 1 in 3D, an isolated single voxel has six medium faces.
    runIsolatedSurfaceShapeScenario(
            Dim3D(20, 20, 20),
            1,
            {Point3D(10, 10, 10)},
            6.0
    );
}

TEST(SurfaceTrackerPluginShapeTest, TracksTwoVoxelSurfaceIn3D) {
    // Two face-adjacent voxels in 3D share one internal face pair, so the total tracked surface is 10.
    runIsolatedSurfaceShapeScenario(
            Dim3D(20, 20, 20),
            1,
            {Point3D(10, 10, 10), Point3D(11, 10, 10)},
            10.0
    );
}
