#include <gtest/gtest.h>

#include <CompuCell3D/plugins/Volume/VolumePlugin.h>

#include "CC3DTestSimulator.h"

using namespace CompuCell3D;

class VolumePluginSimulatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        testSim.addPottsData(Dim3D(20, 20, 1));
        testSim.addCellTypePluginData({
                                              {"Medium", 0},
                                              {"A",      1},
                                              {"B",      2},
                                      });
        testSim.addPluginData("Volume");
        testSim.initializeSimulator();
        volumePlugin = testSim.getPlugin<VolumePlugin>("Volume");
        ASSERT_NE(volumePlugin, nullptr);
    }

    CC3DTestSimulator testSim;
    VolumePlugin *volumePlugin = nullptr;
};

TEST_F(VolumePluginSimulatorTest, ReturnsZeroWhenCopyDoesNotChangeCellIdentity) {
    CellG *cell = testSim.createCell(Point3D(1, 1, 0), 1, {Point3D(1, 2, 0)});
    cell->targetVolume = 4.0f;
    cell->lambdaVolume = 2.0f;

    EXPECT_EQ(cell->volume, 2);
    EXPECT_DOUBLE_EQ(volumePlugin->changeEnergy(Point3D(1, 1, 0), cell, cell), 0.0);
}

TEST_F(VolumePluginSimulatorTest, ComputesEnergyWhenOneCellGainsAndOneCellLosesPixel) {
    CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
            Point3D(1, 2, 0),
            Point3D(2, 1, 0)
    });
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2, {
            Point3D(4, 2, 0)
    });

    newCell->targetVolume = 4.0f;
    newCell->lambdaVolume = 2.0f;
    oldCell->targetVolume = 5.0f;
    oldCell->lambdaVolume = 1.5f;

    ASSERT_EQ(newCell->volume, 3);
    ASSERT_EQ(oldCell->volume, 2);

    const double expectedNew = 2.0 * (1 + 2 * (3 - 4.0));
    const double expectedOld = 1.5 * (1 - 2 * (2 - 5.0));

    EXPECT_DOUBLE_EQ(volumePlugin->changeEnergy(Point3D(4, 1, 0), newCell, oldCell), expectedNew + expectedOld);
}

TEST_F(VolumePluginSimulatorTest, HandlesGainFromMedium) {
    CellG *newCell = testSim.createCell(Point3D(1, 1, 0), 1, {
            Point3D(1, 2, 0),
            Point3D(2, 1, 0)
    });
    newCell->targetVolume = 5.0f;
    newCell->lambdaVolume = 1.25f;

    ASSERT_EQ(newCell->volume, 3);

    const double expected = 1.25 * (1 + 2 * (3 - 5.0));

    EXPECT_DOUBLE_EQ(volumePlugin->changeEnergy(Point3D(3, 1, 0), newCell, nullptr), expected);
}

TEST_F(VolumePluginSimulatorTest, HandlesLossToMedium) {
    CellG *oldCell = testSim.createCell(Point3D(4, 1, 0), 2, {
            Point3D(4, 2, 0),
            Point3D(3, 1, 0)
    });
    oldCell->targetVolume = 4.0f;
    oldCell->lambdaVolume = 0.5f;

    ASSERT_EQ(oldCell->volume, 3);

    const double expected = 0.5 * (1 - 2 * (3 - 4.0));

    EXPECT_DOUBLE_EQ(volumePlugin->changeEnergy(Point3D(4, 1, 0), nullptr, oldCell), expected);
}
