

#ifndef FIELDSECRETOR_H
#define FIELDSECRETOR_H

#include <CompuCell3D/CC3D.h>

#include "SecretionDLLSpecifier.h"


namespace CompuCell3D {

    class CellG;

    class Simulator;

    class BoundaryStrategy;

    class BoundaryPixelTrackerPlugin;

    class PixelTrackerPlugin;


    template<typename Y>
    class WatchableField3D;

    template<typename Y>
    class Field3DImpl;

    class SECRETION_EXPORT FieldSecretorResult {
    public:
        FieldSecretorResult(float tot_amount = 0.0, bool success_flag = true) :
                tot_amount(tot_amount),
                success_flag(success_flag) {}


        bool success_flag;
        float tot_amount;
    };

    class SECRETION_EXPORT FieldSecretorPixelData {
    public:
        FieldSecretorPixelData() {
            pixel = Point3D();
        }

        FieldSecretorPixelData(Point3D _pixel)
                : pixel(_pixel) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const FieldSecretorPixelData &_rhs) const {
            return pixel.x < _rhs.pixel.x || (!(_rhs.pixel.x < pixel.x) && pixel.y < _rhs.pixel.y)
                   || (!(_rhs.pixel.x < pixel.x) && !(_rhs.pixel.y < pixel.y) && pixel.z < _rhs.pixel.z);
        }

        bool operator==(const FieldSecretorPixelData &_rhs) const {
            return pixel == _rhs.pixel;
        }

        ///members
        Point3D pixel;


    };

    class SECRETION_EXPORT  FieldSecretor {
    private:
        double round(double d) {
            return floor(d + 0.5);
        }

    public:

        FieldSecretor();

        ~FieldSecretor();

        Field3D<float> *concentrationFieldPtr;
        BoundaryPixelTrackerPlugin *boundaryPixelTrackerPlugin;
        PixelTrackerPlugin *pixelTrackerPlugin;
        BoundaryStrategy *boundaryStrategy;
        unsigned int maxNeighborIndex;
        WatchableField3D<CellG *> *cellFieldG;

        //IMPORTANT to handle exceptions properly all _secrete functions have secrete counterpart defined in
        // the %extend CompuCell3D::Field secretor in the CompuCellExtraDeclarations.i
        bool _secreteInsideCellConstantConcentration(CellG *_cell, float _amount);

        FieldSecretorResult _secreteInsideCellConstantConcentrationTotalCount(CellG *_cell, float _amount);

        bool _secreteInsideCell(CellG *_cell, float _amount);

        FieldSecretorResult _secreteInsideCellTotalCount(CellG *_cell, float _amount);

        bool _secreteInsideCellAtBoundary(CellG *_cell, float _amount);

        FieldSecretorResult _secreteInsideCellAtBoundaryTotalCount(CellG *_cell, float _amount);

        bool _secreteInsideCellAtBoundaryOnContactWith(CellG *_cell, float _amount,
                                                       const std::vector<unsigned char> &_onContactVec);

        FieldSecretorResult _secreteInsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _amount,
                                                                                const std::vector<unsigned char> &_onContactVec);

        bool _secreteOutsideCellAtBoundary(CellG *_cell, float _amount);

        FieldSecretorResult _secreteOutsideCellAtBoundaryTotalCount(CellG *_cell, float _amount);

        bool _secreteOutsideCellAtBoundaryOnContactWith(CellG *_cell, float _amount,
                                                        const std::vector<unsigned char> &_onContactVec);

        FieldSecretorResult _secreteOutsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _amount,
                                                                                 const std::vector<unsigned char> &_onContactVec);

        bool secreteInsideCellAtCOM(CellG *_cell, float _amount);

        FieldSecretorResult secreteInsideCellAtCOMTotalCount(CellG *_cell, float _amount);

        bool _uptakeInsideCell(CellG *_cell, float _maxUptake, float _relativeUptake);

        FieldSecretorResult _uptakeInsideCellTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake);


        bool _uptakeInsideCellAtBoundary(CellG *_cell, float _maxUptake, float _relativeUptake);

        FieldSecretorResult
        _uptakeInsideCellAtBoundaryTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake);

        bool _uptakeInsideCellAtBoundaryOnContactWith(CellG *_cell, float _maxUptake, float _relativeUptake,
                                                      const std::vector<unsigned char> &_onContactVec);

        FieldSecretorResult
        _uptakeInsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake,
                                                           const std::vector<unsigned char> &_onContactVec);

        bool _uptakeOutsideCellAtBoundary(CellG *_cell, float _maxUptake, float _relativeUptake);

        FieldSecretorResult
        _uptakeOutsideCellAtBoundaryTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake);


        bool _uptakeOutsideCellAtBoundaryOnContactWith(CellG *_cell, float _maxUptake, float _relativeUptake,
                                                       const std::vector<unsigned char> &_onContactVec);

        FieldSecretorResult
        _uptakeOutsideCellAtBoundaryOnContactWithTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake,
                                                            const std::vector<unsigned char> &_onContactVec);


        bool uptakeInsideCellAtCOM(CellG *_cell, float _maxUptake, float _relativeUptake);

        FieldSecretorResult uptakeInsideCellAtCOMTotalCount(CellG *_cell, float _maxUptake, float _relativeUptake);

        float _amountSeenByCell(CellG *_cell);

        float totalFieldIntegral();

    };

};
#endif

