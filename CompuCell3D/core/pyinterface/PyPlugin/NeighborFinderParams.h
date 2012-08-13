#ifndef NEIGHBORFINDERPARAMS_H
#define NEIGHBORFINDERPARAMS_H

#include <CompuCell3D/Field3D/Point3D.h>

#include <Python.h>

namespace CompuCell3D{

class NeighborFinderParams{
	public:
		NeighborFinderParams(){
			pt=Point3D();
			token=0;
			distance=0.0;
			checkBounds=false;
		}
// 		NeighborFinderParams(Point3D _pt,unsigned int _token=0,double _distance=0.0,bool _checkBounds=false):
// 		pt(_pt),
// 		token(_token),
// 		distance(_distance),
// 		checkBounds(_checkBounds)
// 		{}
		~NeighborFinderParams(){}
		void reset()
		{
			pt=Point3D();
			token=0;
			distance=0.0;
			checkBounds=false;
		}

		
		Point3D pt;
		unsigned int token;
		double distance;
		bool checkBounds;
		
};

};



#endif
