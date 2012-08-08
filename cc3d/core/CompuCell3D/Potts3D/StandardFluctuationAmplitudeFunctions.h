/*************************************************************************
*    CompuCell - A software framework for multimodel simulations of     *
* biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
*                             Indiana                                   *
*                                                                       *
* This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
*  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
*  CompuCell GNU General Public License RIDER you can redistribute it   *
* and/or modify it under the terms of the GNU General Public License as *
*  published by the Free Software Foundation; either version 2 of the   *
*         License, or (at your option) any later version.               *
*                                                                       *
* This program is distributed in the hope that it will be useful, but   *
*      WITHOUT ANY WARRANTY; without even the implied warranty of       *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
*             General Public License for more details.                  *
*                                                                       *
*  You should have received a copy of the GNU General Public License    *
*     along with this program; if not, write to the Free Software       *
*      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
*************************************************************************/

#ifndef STANDARDFLUCTUATIONAMPLITUDEFUNCTIONS_H
#define STANDARDFLUCTUATIONAMPLITUDEFUNCTIONS_H

#include "FluctuationAmplitudeFunction.h"

namespace CompuCell3D {


	class MinFluctuationAmplitudeFunction:public FluctuationAmplitudeFunction{

	public:
		MinFluctuationAmplitudeFunction(const Potts3D * _potts);

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

		virtual double fluctuationAmplitude(const CellG * newCell, const CellG * oldCell);


	};

	class MaxFluctuationAmplitudeFunction:public FluctuationAmplitudeFunction{

	public:
		MaxFluctuationAmplitudeFunction(const Potts3D * _potts);

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

		virtual double fluctuationAmplitude(const CellG * newCell, const CellG * oldCell);


	};

	class ArithmeticAverageFluctuationAmplitudeFunction:public FluctuationAmplitudeFunction{

	public:
		ArithmeticAverageFluctuationAmplitudeFunction(const Potts3D * _potts);

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

		virtual double fluctuationAmplitude(const CellG * newCell, const CellG * oldCell);


	};


};
#endif
