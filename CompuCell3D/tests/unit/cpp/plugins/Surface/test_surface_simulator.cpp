#include <gtest/gtest.h>

#include <CompuCell3D/plugins/Surface/SurfacePlugin.h>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

class SurfacePluginSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        testSim.addPottsData(Dim3D(20, 20, 1));
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        testSim.addPluginData("Surface");
        testSim.initializeSimulator();
        surfacePlugin = testSim.getPlugin<SurfacePlugin>("Surface");
        ASSERT_NE(surfacePlugin, nullptr);
    }

    CC3DTestSimulator testSim;
    SurfacePlugin *surfacePlugin = nullptr;
};

TEST_F(SurfacePluginSimulatorTest, ReturnsZeroWhenCopyDoesNotChangeCellIdentity) {
    // Copy source and target belong to the same cell, so the surface term must not change.
    CellG *cell = testSim.createCell(Point3D(1, 1, 0), 1, {Point3D(1, 2, 0)});
    cell->targetSurface = 6.0f;
    cell->lambdaSurface = 2.0f;

    EXPECT_DOUBLE_EQ(cell->surface, 6.0);
    EXPECT_DOUBLE_EQ(surfacePlugin->changeEnergy(Point3D(1, 1, 0), cell, cell), 0.0);
}

TEST_F(SurfacePluginSimulatorTest, ComputesEnergyWhenOneCellGainsAndOneCellLosesPixel) {
    // Cell A copies into a pixel occupied by Cell B. With default surface neighbor order 1
    // on the square lattice, A gains 4 surface units and B loses 2.
    CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
            Point3D(1, 2, 0),
            Point3D(2, 1, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2, {
            Point3D(4, 2, 0)
    });

    newCell->targetSurface = 9.0f;
    newCell->lambdaSurface = 2.0f;
    oldCell->targetSurface = 4.0f;
    oldCell->lambdaSurface = 1.5f;

    ASSERT_DOUBLE_EQ(newCell->surface, 8.0);
    ASSERT_DOUBLE_EQ(oldCell->surface, 6.0);

    const double expectedNew = 2.0 * (16 + 2 * 4 * (8.0 - 9.0));
    const double expectedOld = 1.5 * (4 + 2 * (-2) * (6.0 - 4.0));

    EXPECT_DOUBLE_EQ(surfacePlugin->changeEnergy(Point3D(4, 1, 0), newCell, oldCell), expectedNew + expectedOld);
}

TEST_F(SurfacePluginSimulatorTest, HandlesGainFromMedium) {
    // A three-pixel L-shaped cell expands into medium. Under neighbor order 1, the copied pixel
    // touches the cell on one side, so the cell surface increases by 2.
    CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
            Point3D(1, 2, 0),
            Point3D(2, 1, 0)
    });
    newCell->targetSurface = 10.0f;
    newCell->lambdaSurface = 1.25f;

    ASSERT_DOUBLE_EQ(newCell->surface, 8.0);

    const double expected = 1.25 * (4 + 2 * 2 * (8.0 - 10.0));

    EXPECT_DOUBLE_EQ(surfacePlugin->changeEnergy(Point3D(3, 1, 0), newCell, nullptr), expected);
}

TEST_F(SurfacePluginSimulatorTest, HandlesSinglePixelCellGainFromMedium) {
    // A one-pixel cell expands into medium. The copied pixel shares one face with the cell,
    // so the surface grows from 4 to 6.
    CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1);
    newCell->targetSurface = 6.0f;
    newCell->lambdaSurface = 1.25f;

    ASSERT_DOUBLE_EQ(newCell->surface, 4.0);

    const double expected = 1.25 * (4 + 2 * 2 * (4.0 - 6.0));

    EXPECT_DOUBLE_EQ(surfacePlugin->changeEnergy(Point3D(2, 1, 0), newCell, nullptr), expected);
}

TEST_F(SurfacePluginSimulatorTest, HandlesLossToMedium) {
    // Medium overwrites the corner pixel of a three-pixel L-shaped cell. With neighbor order 1,
    // the remaining two pixels become diagonally separated and the total tracked surface is unchanged.
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2, {
            Point3D(4, 2, 0),
            Point3D(3, 1, 0)
    });
    oldCell->targetSurface = 9.0f;
    oldCell->lambdaSurface = 0.5f;

    ASSERT_DOUBLE_EQ(oldCell->surface, 8.0);

    EXPECT_DOUBLE_EQ(surfacePlugin->changeEnergy(Point3D(4, 1, 0), nullptr, oldCell), 0.0);
}

TEST_F(SurfacePluginSimulatorTest, HandlesSinglePixelCellLossToMedium) {
    // A one-pixel cell is overwritten by medium, taking its surface from 4 down to 0.
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2);
    oldCell->targetSurface = 3.0f;
    oldCell->lambdaSurface = 0.5f;

    ASSERT_DOUBLE_EQ(oldCell->surface, 4.0);

    const double expected = 0.5 * (16 + 2 * (-4) * (4.0 - 3.0));

    EXPECT_DOUBLE_EQ(surfacePlugin->changeEnergy(Point3D(4, 1, 0), nullptr, oldCell), expected);
}
