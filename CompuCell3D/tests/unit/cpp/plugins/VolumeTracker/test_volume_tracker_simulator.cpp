#include <gtest/gtest.h>

#include <CompuCell3D/plugins/VolumeTracker/VolumeTrackerPlugin.h>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

class VolumeTrackerPluginSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        testSim.addPottsData(Dim3D(20, 20, 1));
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        testSim.addPluginData("VolumeTracker");
        testSim.initializeSimulator();
        volumeTrackerPlugin = testSim.getPlugin<VolumeTrackerPlugin>("VolumeTracker");
        ASSERT_NE(volumeTrackerPlugin, nullptr);
    }

    CC3DTestSimulator testSim;
    VolumeTrackerPlugin *volumeTrackerPlugin = nullptr;
};

TEST_F(VolumeTrackerPluginSimulatorTest, KeepsVolumeUnchangedWhenAssigningSameCellToSamePixel) {
    // Reassigning a pixel to the same cell should leave the tracked volume unchanged.
    CellG *cell = testSim.createCell(Point3D(1, 1, 0), 1, {Point3D(1, 2, 0)});
    ASSERT_EQ(cell->volume, 2);

    testSim.getCellField()->set(Point3D(1, 1, 0), cell);

    EXPECT_EQ(cell->volume, 2);
    EXPECT_EQ(testSim.getPotts()->getNumCells(), 1u);
}

TEST_F(VolumeTrackerPluginSimulatorTest, UpdatesVolumesWhenOneCellGainsAndOneCellLosesPixel) {
    // Cell A overwrites one pixel of Cell B. The tracker should increment A and decrement B immediately.
    CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
            Point3D(1, 2, 0),
            Point3D(2, 1, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2, {
            Point3D(4, 2, 0)
    });

    ASSERT_EQ(newCell->volume, 3);
    ASSERT_EQ(oldCell->volume, 2);

    testSim.getCellField()->set(Point3D(4, 1, 0), newCell);

    EXPECT_EQ(newCell->volume, 4);
    EXPECT_EQ(oldCell->volume, 1);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(4, 1, 0)), newCell);
}

TEST_F(VolumeTrackerPluginSimulatorTest, UpdatesVolumeWhenCellGainsPixelFromMedium) {
    // A cell expands into medium, so only the gaining cell volume increases.
    CellG *cell = testSim.createCell(Point3D(1, 1, 0), 1, {
            Point3D(1, 2, 0),
            Point3D(2, 1, 0)
    });

    ASSERT_EQ(cell->volume, 3);

    testSim.getCellField()->set(Point3D(3, 1, 0), cell);

    EXPECT_EQ(cell->volume, 4);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(3, 1, 0)), cell);
}

TEST_F(VolumeTrackerPluginSimulatorTest, UpdatesVolumeWhenCellLosesPixelToMedium) {
    // Medium overwrites one pixel of a multi-pixel cell, so the tracked volume decreases immediately.
    CellG *cell = testSim.createCell(Point3D(4, 1, 0), 2, {
            Point3D(4, 2, 0),
            Point3D(3, 1, 0)
    });

    ASSERT_EQ(cell->volume, 3);

    testSim.getCellField()->set(Point3D(4, 1, 0), nullptr);

    EXPECT_EQ(cell->volume, 2);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(4, 1, 0)), nullptr);
    EXPECT_EQ(testSim.getPotts()->getNumCells(), 1u);
}

TEST_F(VolumeTrackerPluginSimulatorTest, DefersSinglePixelCellDeletionUntilStep) {
    // When a one-pixel cell loses its last voxel, the tracker sets volume to zero immediately
    // but defers cell destruction until the stepper runs.
    CellG *cell = testSim.createCell(Point3D(4, 1, 0), 2);
    const long cellId = cell->id;

    ASSERT_EQ(cell->volume, 1);
    ASSERT_EQ(testSim.getPotts()->getNumCells(), 1u);

    testSim.getCellField()->set(Point3D(4, 1, 0), nullptr);

    EXPECT_EQ(cell->volume, 0);
    EXPECT_EQ(testSim.getCellField()->get(Point3D(4, 1, 0)), nullptr);
    EXPECT_EQ(testSim.getPotts()->getNumCells(), 1u);
    EXPECT_EQ(testSim.getPotts()->getCellInventory().attemptFetchingCellById(cellId), cell);

    volumeTrackerPlugin->step();

    EXPECT_EQ(testSim.getPotts()->getNumCells(), 0u);
    EXPECT_EQ(testSim.getPotts()->getCellInventory().attemptFetchingCellById(cellId), nullptr);
}
