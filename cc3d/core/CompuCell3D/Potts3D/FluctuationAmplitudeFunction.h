
#ifndef FLUCTUATIONAMPLITUDEFUNCTION_H
#define FLUCTUATIONAMPLITUDEFUNCTION_H
#include <vector>

namespace CompuCell3D {


	class Potts3D;
	class CellG;

	class FluctuationAmplitudeFunction {

	public:
		FluctuationAmplitudeFunction(const Potts3D * _potts):
		  potts(_potts)
		{}

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

		virtual double fluctuationAmplitude(const CellG * newCell, const CellG * oldCell) = 0;

	protected:
		const Potts3D *potts;
		

	};


};
#endif
