#ifndef STANDARDFLUCTUATIONAMPLITUDEFUNCTIONS_H
#define STANDARDFLUCTUATIONAMPLITUDEFUNCTIONS_H

#include "FluctuationAmplitudeFunction.h"

namespace CompuCell3D {


    class MinFluctuationAmplitudeFunction : public FluctuationAmplitudeFunction {

    public:
        MinFluctuationAmplitudeFunction(const Potts3D *_potts);

        /**
        * Calculates the fluctuationAmplitude based on source/destination cells fluctuation amplitudes
        * Fluctuation Amplitude is "more biological" code for temperature parameter used in classical POtts
        *
        *
        * @param _oldCell - destination cell.
        * @param _newCell - source cell.
        *
        * @return fluctuationAmplitude - in case fluctAmpl cell attribute is set to negative number we return global temperatue as a fluctuation Amplitude.
        */

        virtual double fluctuationAmplitude(const CellG *newCell, const CellG *oldCell);


    };

    class MaxFluctuationAmplitudeFunction : public FluctuationAmplitudeFunction {

    public:
        MaxFluctuationAmplitudeFunction(const Potts3D *_potts);

        /**
        * Calculates the fluctuationAmplitude based on source/destination cells fluctuation amplitudes
        * Fluctuation Amplitude is "more biological" code for temperature parameter used in classical POtts
        *
        *
        * @param _oldCell - destination cell.
        * @param _newCell - source cell.
        *
        * @return fluctuationAmplitude - in case fluctAmpl cell attribute is set to negative number we return global temperatue as a fluctuation Amplitude.
        */

        virtual double fluctuationAmplitude(const CellG *newCell, const CellG *oldCell);


    };

    class ArithmeticAverageFluctuationAmplitudeFunction : public FluctuationAmplitudeFunction {

    public:
        ArithmeticAverageFluctuationAmplitudeFunction(const Potts3D *_potts);

        /**
        * Calculates the fluctuationAmplitude based on source/destination cells fluctuation amplitudes
        * Fluctuation Amplitude is "more biological" code for temperature parameter used in classical POtts
        *
        *
        * @param _oldCell - destination cell.
        * @param _newCell - source cell.
        *
        * @return fluctuationAmplitude - in case fluctAmpl cell attribute is set to negative number we return global temperatue as a fluctuation Amplitude.
        */

        virtual double fluctuationAmplitude(const CellG *newCell, const CellG *oldCell);


    };


};
#endif
