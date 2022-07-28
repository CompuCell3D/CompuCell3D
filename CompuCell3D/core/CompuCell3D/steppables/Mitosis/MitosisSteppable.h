
#ifndef MITOSISSTEPPABLE_H
#define MITOSISSTEPPABLE_H

#include <CompuCell3D/CC3D.h>
#include "MitosisSteppableDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class PixelTracker;

    class PixelTrackerPlugin;

    class Simulator;

    class CellG;

    class BoundaryStrategy;

    class PixelTrackerData;

    class RandomNumberGenerator;

    class MITOSISSTEPPABLE_EXPORT SteppableOrientationVectorsMitosis {
    public:
        SteppableOrientationVectorsMitosis() {}

        Vector3 semiminorVec;
        Vector3 semimajorVec;


    };

    class MITOSISSTEPPABLE_EXPORT CompartmentMitosisData {
    public:
        CompartmentMitosisData() : cell(0), type(0) {}

        Vector3 com;//center of mass
        CellG *cell;
        unsigned char type;
        Point3D pt;

    };


    class MITOSISSTEPPABLE_EXPORT MitosisSteppable : public Steppable {

        RandomNumberGenerator *randGen;
        int parentChildPositionFlag;

        bool tryAdjustingCompartmentCOM(Vector3 &_com, const std::set <PixelTrackerData> &_clusterPixels);

        double xFactor, yFactor, zFactor;
    public:

        typedef bool (MitosisSteppable::*doDirectionalMitosis2DPtr_t)();

        doDirectionalMitosis2DPtr_t doDirectionalMitosis2DPtr;


        CellG *childCell;
        CellG *parentCell;
        BoundaryStrategy *boundaryStrategy;

        int maxNeighborIndex;
        Simulator *sim;
        Potts3D *potts;

        bool divideAlongMinorAxisFlag;
        bool divideAlongMajorAxisFlag;
        bool flag3D;


        void setParentChildPositionFlag(int _flag);

        int getParentChildPositionFlag();

        ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr;
        PixelTrackerPlugin *pixelTrackerPlugin;
        //comaprtment mitosis members
        std::vector <CompartmentMitosisData> parentBeforeMitosis;
        std::vector <CompartmentMitosisData> comOffsetsMitosis;
        std::vector <CompartmentMitosisData> parentAfterMitosis;
        std::vector <CompartmentMitosisData> childAfterMitosis;

        Point3D boundaryConditionIndicator;
        Dim3D fieldDim;

        MitosisSteppable();

        virtual ~MitosisSteppable();

        void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        // Begin Steppable interface
        virtual void start() {}

        virtual void step(const unsigned int _currentStep) {}

        virtual void finish() {}
        // End Steppable interface



        // SimObject interface

        typedef SteppableOrientationVectorsMitosis (MitosisSteppable::*getOrientationVectorsMitosis2DPtr_t)(
                std::set <PixelTrackerData> &);

        getOrientationVectorsMitosis2DPtr_t getOrientationVectorsMitosis2DPtr;


        SteppableOrientationVectorsMitosis getOrientationVectorsMitosis(CellG *_cell);

        SteppableOrientationVectorsMitosis getOrientationVectorsMitosisCompartments(long _clusterId);

        SteppableOrientationVectorsMitosis getOrientationVectorsMitosis(std::set <PixelTrackerData> &clusterPixels);

        SteppableOrientationVectorsMitosis
        getOrientationVectorsMitosis2D_xy(std::set <PixelTrackerData> &clusterPixels);

        SteppableOrientationVectorsMitosis
        getOrientationVectorsMitosis2D_xz(std::set <PixelTrackerData> &clusterPixels);

        SteppableOrientationVectorsMitosis
        getOrientationVectorsMitosis2D_yz(std::set <PixelTrackerData> &clusterPixels);

        SteppableOrientationVectorsMitosis getOrientationVectorsMitosis3D(std::set <PixelTrackerData> &clusterPixels);


        bool doDirectionalMitosisOrientationVectorBased(CellG *_cell, double _nx, double _ny, double _nz);

        bool doDirectionalMitosisAlongMajorAxis(CellG *_cell);

        bool doDirectionalMitosisAlongMinorAxis(CellG *_cell);

        bool doDirectionalMitosisRandomOrientation(CellG *_cell);

        bool doDirectionalMitosisOrientationVectorBasedCompartments(CellG *_cell, double _nx, double _ny, double _nz);

        bool
        doDirectionalMitosisOrientationVectorBasedCompartments(long _clusterId, double _nx, double _ny, double _nz);

        bool doDirectionalMitosisRandomOrientationCompartments(long _clusterId);

        bool doDirectionalMitosisAlongMajorAxisCompartments(long _clusterId);

        bool doDirectionalMitosisAlongMinorAxisCompartments(long _clusterId);

        Vector3 getShiftVector(std::set <PixelTrackerData> &_sourcePixels);

        Vector3 calculateCOMPixels(std::set <PixelTrackerData> &_pixels);

        CellG *createChildCell(std::set <PixelTrackerData> &_pixels);

        void shiftCellPixels(std::set <PixelTrackerData> &_sourcePixels, std::set <PixelTrackerData> &_targetPixels,
                             Vector3 _shiftVec);

        bool divideClusterPixelsOrientationVectorBased(std::set <PixelTrackerData> &clusterPixels,
                                                       std::set <PixelTrackerData> &clusterParent,
                                                       std::set <PixelTrackerData> &clusterChild, double _nx,
                                                       double _ny, double _nz);

        Vector3 calculateClusterPixelsCOM(std::set <PixelTrackerData> &clusterPixels);


        void
        initializeClusters(std::vector<int> &originalCompartmentVolumeVec, std::set <PixelTrackerData> &clusterPixels,
                           std::vector <CompartmentMitosisData> &clusterKernels,
                           std::vector<double> &attractionRadiusVec,
                           std::vector <CompartmentMitosisData> &parentBeforeMitosisCMDVec, Vector3 shiftVec);
    };
};
#endif
