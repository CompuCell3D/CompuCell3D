#include "NumericalUtils.h"
#include <cmath>
#include <CompuCell3D/Potts3D/Cell.h>
#include <CompuCell3D/Field3D/Point3D.h>
#include <iostream>

#include <CompuCell3D/Boundary/BoundaryStrategy.h>

using namespace std;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double round(double number)
{
    return number < 0.0 ? ceil(number - 0.5) : floor(number + 0.5);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float findMin( float _d , int _dim ){

   float minD = _d ;
   
   if( fabs(_d+_dim) < fabs(minD) ){
      minD=_d+_dim;
   }

   if( fabs(_d-_dim) < fabs(minD) ){
      minD=_d-_dim;
   }

   return minD;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double dist(double x0,double y0,double z0){
   return sqrt((x0)*(x0)+(y0)*(y0)+(z0)*(z0));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

double dist(double x0,double y0,double z0,double x1,double y1,double z1){
   return sqrt((x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1));
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::complex<double> > solveCubicEquationRealCoeeficients(std::vector<double> & aRef){
	//we are using Cardano's method as described in Stephen Fisher "Complex Variables" Second Edition 

	
	std::vector<double>  aVec=aRef;
	vector<complex<double> > roots(3);
	
	if(aVec[0]!=0.0){
		//setting coefficients so that aVec[0]=1.0
		for (int i =aVec.size()-1; i >= 0 ; --i){
			aVec[i]/=aVec[0];
			//cerr<<"coeff["<<i<<"]="<<aVec[i]<<endl;
		}
	}else{
		return roots;
	}


	//reduced form of the above equation after x=w-aVec[1]/3 substitution is:
	// w^3+a*w+b=0
	//where 
	double a,b;
	a=aVec[2]-1.0/3.0*aVec[1]*aVec[1];
	b=2/27.0*aVec[1]*aVec[1]*aVec[1]-1.0/3.0*aVec[1]*aVec[2]+aVec[3];
	complex<double> d(-aVec[1]/3,0.0);
	complex<double> aComplex(a,0.0);

	double PI=acos(-1.0);
	//cerr<<"a="<<a<<endl;
	//cerr<<"b="<<b<<endl;
	if(a==0.0){
		
		complex<double> bComplex(-b,0.0);
		double bAbs=abs(bComplex);
		double bArg=arg(bComplex);
		
		roots[0]=polar(pow(bAbs,1/3.0),bArg/3.0)+d;
		roots[1]=polar(pow(bAbs,1/3.0),bArg/3.0+2*PI/3)+d;
		roots[2]=polar(pow(bAbs,1/3.0),bArg/3.0+4*PI/3)+d;
		return roots;
	}else{
		complex<double> lambda=sqrt(-aComplex/3.0);
		complex<double> beta=pow(lambda,-3.0)*b;

		//cerr<<"lambda="<<lambda<<endl;
		//cerr<<"beta="<<beta<<endl;

		//now solve for roots of p^6+beta*p^3+1=0
		complex<double> p1=(-beta-sqrt(beta*beta-complex<double>(4.0,0)))/(2.0);
		complex<double> p2=(-beta+sqrt(beta*beta-complex<double>(4.0,0)))/(2.0);



		double p1Abs=abs(p1);
		double p1Arg=arg(p1);
		//cerr<<"p1="<<p1<<endl;
	 //  cerr<<"p2="<<p2<<endl;
		//cerr<<"gamma1="<<polar(pow(p1Abs,1/3.0),p1Arg/3.0)<<endl;
		
		complex<double> q0=polar(pow(p1Abs,1/3.0),p1Arg/3.0)+1.0/polar(pow(p1Abs,1/3.0),p1Arg/3.0);
		complex<double> q1=polar(pow(p1Abs,1/3.0),p1Arg/3.0+2*PI/3)+1.0/polar(pow(p1Abs,1/3.0),p1Arg/3.0+2*PI/3);
		complex<double> q2=polar(pow(p1Abs,1/3.0),p1Arg/3.0+4*PI/3)+1.0/polar(pow(p1Abs,1/3.0),p1Arg/3.0+4*PI/3);
		//cerr<<"q1="<<q1<<endl;
		//complex<double> w1=lambda*q1;
		//cerr<<"w1="<<w1<<endl;
		//complex<double> root1=w1+d;
		//cerr<<"root1="<<root1<<endl;
		roots[0]=lambda*q0+d;
		roots[1]=lambda*q1+d;
		roots[2]=lambda*q2+d;

		return roots;
	}

	
	return roots;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace CompuCell3D{


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Coordinates3D<double> precalculateCentroid(const Point3D & pt, const CellG *_cell, int _volumeIncrement,const Point3D & fieldDim, BoundaryStrategy *boundaryStrategy)
{

   //cerr<<"pt="<<pt<<endl;
   //cerr<<"boundaryStrategy="<<boundaryStrategy<<endl;
   Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);
   //cerr<<"ptTrans="<<ptTrans<<endl;
   Coordinates3D<double> fieldDimTrans= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,fieldDim.y-1,fieldDim.z-1));

   Coordinates3D<double> shiftVec;
   Coordinates3D<double> shiftedPt;

   Coordinates3D<double> distanceVecMin;
   //determines minimum coordinates for the perpendicular lines paccinig through pt
   Coordinates3D<double> distanceVecMax;
   Coordinates3D<double> distanceVecMax_1;
   //determines minimum coordinates for the perpendicular lines paccinig through pt
   Coordinates3D<double> distanceVec; //measures lattice diatances along x,y,z - they can be different for different lattices. The lines have to pass through pt

   distanceVecMin.x=boundaryStrategy->calculatePointCoordinates(Point3D(0,pt.y,pt.z)).x;
   distanceVecMin.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,0,pt.z)).y;
   distanceVecMin.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,0)).z;

   distanceVecMax.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x,pt.y,pt.z)).x;
   distanceVecMax.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y,pt.z)).y;
   distanceVecMax.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z)).z;


   distanceVecMax_1.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,pt.y,pt.z)).x;
   distanceVecMax_1.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y-1,pt.z)).y;
   distanceVecMax_1.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z-1)).z;



   distanceVec=distanceVecMax-distanceVecMin;

   double xCM,yCM,zCM;
//    if(pt.x==0 || pt.x==_fieldDim.x-1)
//       cerr<<"_cell->xCM="<<_cell->xCM<<" volumeIncr="<<_volumeIncrement<<" pt="<<pt<<endl;

   //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   

   shiftVec.x= (_cell->xCM/_cell->volume-((int)fieldDimTrans.x)/2);
   shiftVec.y= (_cell->yCM/_cell->volume-((int)fieldDimTrans.y)/2);
   shiftVec.z= (_cell->zCM/_cell->volume-((int)fieldDimTrans.z)/2);
	//cerr<<"fieldDimTrans="<<fieldDimTrans<<endl;
	//cerr<<"_cell->xCM/_cell->volume="<<_cell->xCM/_cell->volume<<" ((int)fieldDimTrans.x)/2="<<((int)fieldDimTrans.x)/2<<endl;

    //shift CM to approximately center of lattice, new centroids are:
    xCM = _cell->xCM - shiftVec.x*(_cell->volume);
    yCM = _cell->yCM - shiftVec.y*(_cell->volume);
    zCM = _cell->zCM - shiftVec.z*(_cell->volume);

    //Now shift pt
    shiftedPt=ptTrans;
	//cerr<<"ptTrans="<<ptTrans<<" shiftVec.x="<<shiftVec.x<<endl;
    shiftedPt-=shiftVec;
//     if(pt.x==0 || pt.x==_fieldDim.x-1)
      //cerr<<"EARLY shiftedPt.x="<<shiftedPt.x<<" xCM="<<xCM<<" xcm="<<xCM/(float)_cell->volume<<endl;


    //making sure that shifted point is in the lattice
    if(shiftedPt.x < distanceVecMin.x){
      shiftedPt.x += distanceVec.x;
    }else if (shiftedPt.x > distanceVecMax_1.x){
      shiftedPt.x -= distanceVec.x;
    }  

    if(shiftedPt.y < distanceVecMin.y){
      shiftedPt.y += distanceVec.y;
    }else if (shiftedPt.y > distanceVecMax_1.y){
      shiftedPt.y -= distanceVec.y;
    }  

    if(shiftedPt.z < distanceVecMin.z){
      shiftedPt.z += distanceVec.z;
    }else if (shiftedPt.z > distanceVecMax_1.z){
      shiftedPt.z -= distanceVec.z;
    }

    //update shifted centroids
    xCM += shiftedPt.x*(_volumeIncrement>0?1:-1);
    yCM += shiftedPt.y*(_volumeIncrement>0?1:-1);
    zCM += shiftedPt.z*(_volumeIncrement>0?1:-1);
    
    //shift back centroids
    xCM += shiftVec.x * (_cell->volume+_volumeIncrement);
    yCM += shiftVec.y * (_cell->volume+_volumeIncrement);
    zCM += shiftVec.z * (_cell->volume+_volumeIncrement);
   
//     if(pt.x==0 || pt.x==_fieldDim.x-1)
//       cerr<<"shiftedPt.x="<<shiftedPt.x<<" xCM="<<xCM<<endl;

   return Coordinates3D<double>(xCM,yCM,zCM);


//    Point3D shiftVec;
//    Point3D shiftedPt;
//    int xCM,yCM,zCM;
// //    if(pt.x==0 || pt.x==_fieldDim.x-1)
// //       cerr<<"_cell->xCM="<<_cell->xCM<<" volumeIncr="<<_volumeIncrement<<" pt="<<pt<<endl;
// 
//    //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
//    shiftVec.x= (short)((_cell->xCM/(float)(_cell->volume)-fieldDim.x/2));
//    shiftVec.y= (short)((_cell->yCM/(float)(_cell->volume)-fieldDim.y/2));
//    shiftVec.z= (short)((_cell->zCM/(float)(_cell->volume)-fieldDim.z/2));
// 
//     //shift CM to approximately center of lattice, new centroids are:
//     xCM = _cell->xCM - shiftVec.x*(_cell->volume);
//     yCM = _cell->yCM - shiftVec.y*(_cell->volume);
//     zCM = _cell->zCM - shiftVec.z*(_cell->volume);
// 
//     //Now shift pt
//     shiftedPt=pt;
//     shiftedPt-=shiftVec;
//     cerr<<"pt="<<pt<<endl;
//     if(pt.x==0 || pt.x==fieldDim.x-1){
//       cerr<<"shiftVec="<<shiftVec<<" _cell->volume="<<_cell->volume <<endl;
//       cerr<<"EARLY shiftedPt.x="<<shiftedPt.x<<" cell->xCM="<<_cell->xCM<<" xCM="<<xCM<<" xcm="<<xCM/(float)_cell->volume<<endl;
//     }
// 
//     if(pt.y==0 || pt.y==fieldDim.y-1){
//       cerr<<"shiftVec="<<shiftVec<<" _cell->volume="<<_cell->volume <<endl;
//       cerr<<"EARLY shiftedPt.y="<<shiftedPt.y<<" cell->yCM="<<_cell->yCM<<" yCM="<<yCM<<" ycm="<<yCM/(float)_cell->volume<<endl;
//     }
//     //making sure that shifted point is in the lattice
//     if(shiftedPt.x < 0){
//       shiftedPt.x += fieldDim.x;
// //       if(pt.x==0 || pt.x==_fieldDim.x-1)
// //          cerr<<"SHIFTTING shiftedPt.x="<<shiftedPt.x<<" _fieldDim.x="<<_fieldDim.x<<endl;
//     }else if (shiftedPt.x > fieldDim.x-1){
//       shiftedPt.x -= fieldDim.x;
//     }  
// 
//     if(shiftedPt.y < 0){
//       shiftedPt.y += fieldDim.y;
//     }else if (shiftedPt.y > fieldDim.y-1){
//       shiftedPt.y -= fieldDim.y;
//     }  
// 
//     if(shiftedPt.z < 0){
//       shiftedPt.z += fieldDim.z;
//     }else if (shiftedPt.z > fieldDim.z-1){
//       shiftedPt.z -= fieldDim.z;
//     }
//     //update shifted centroids
//     xCM += shiftedPt.x*(_volumeIncrement>0?1:-1);
//     yCM += shiftedPt.y*(_volumeIncrement>0?1:-1);
//     zCM += shiftedPt.z*(_volumeIncrement>0?1:-1);
//     
//     //shift back centroids
//     xCM += shiftVec.x * (_cell->volume+_volumeIncrement);
//     yCM += shiftVec.y * (_cell->volume+_volumeIncrement);
//     zCM += shiftVec.z * (_cell->volume+_volumeIncrement);
//    
//     if(pt.y==0 || pt.y==fieldDim.y-1){
//       cerr<<"\t\t\tshiftedPt.y="<<shiftedPt.y<<" yCM="<<yCM<<endl;
// //       exit(0);
//     }
//     if(pt.x==0 || pt.x==fieldDim.x-1){
//       cerr<<"\t\t\tshiftedPt.x="<<shiftedPt.x<<" xCM="<<xCM<<endl;
// //       exit(0);
//     }
// 
//    return Coordinates3D<int>(xCM,yCM,zCM);
}

// double distanceInvariantCM(double x0,double y0,double z0,double x1,double y1,double z1,const Point3D & fieldDim, BoundaryStrategy *boundaryStrategy,int _pbcX,int _pbcY,int _pbcZ){

    // // _pbcX !=0 ? _pbcX=1:_pbcX =0;
    // // _pbcY !=0 ? _pbcY=1:_pbcY =0;
    // // _pbcZ !=0 ? _pbcZ=1:_pbcZ =0;
    
   // double x0_t,y0_t,z0_t;
   // double x1_t,y1_t,z1_t;

   
   // Coordinates3D<double> fieldDimTrans= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x,fieldDim.y,fieldDim.z));
   
   // Coordinates3D<double> fieldDimTrans_1= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,fieldDim.y-1,fieldDim.z-1));

   // Coordinates3D<double> shiftVec;

   // //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
   // // shiftVec.x= (x0-fieldDimTrans.x/2)*_pbcX;
   // // shiftVec.y= (y0-fieldDimTrans.y/2)*_pbcY;
   // // shiftVec.z= (z0-fieldDimTrans.z/2)*_pbcZ;

   // shiftVec.x= (x0-fieldDimTrans.x/2)*_pbcX;
   // shiftVec.y= (y0-fieldDimTrans.y/2)*_pbcY;
   // shiftVec.z= (z0-fieldDimTrans.z/2)*_pbcZ;
   
   // //moving x0,y0,z0 to approximetely center of the lattice
   // x0_t=x0-shiftVec.x;
   // y0_t=y0-shiftVec.y;
   // z0_t=z0-shiftVec.z;

   // //shifting accordingly other coordinates
   // x1_t=x1-shiftVec.x;
   // y1_t=y1-shiftVec.y;
   // z1_t=z1-shiftVec.z;

    // //making sure that x1_t,y1_t,z1_t is in the lattice

    // if(x1_t < 0){
      // x1_t += fieldDimTrans.x;
    // }else if (x1_t > fieldDimTrans_1.x){
      // x1_t -= fieldDimTrans.x;
    // }  

    // if(y1_t < 0){
      // y1_t += fieldDimTrans.y;
    // }else if (y1_t > fieldDimTrans_1.y){
      // y1_t -= fieldDimTrans.y;
    // }  

    // if(z1_t < 0){
      // z1_t += fieldDimTrans.z;
    // }else if (z1_t > fieldDimTrans_1.z){
      // z1_t -= fieldDimTrans.z;
    // }  

   // return dist(x0_t,y0_t,z0_t,x1_t,y1_t,z1_t);
    
// }

double distanceInvariantCM(double x0,double y0,double z0,double x1,double y1,double z1,const Point3D & fieldDim, BoundaryStrategy *boundaryStrategy,int _pbcX,int _pbcY,int _pbcZ){
   double x0_t,y0_t,z0_t;
   double x1_t,y1_t,z1_t;



   //Coordinates3D<double> fieldDimTrans= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x,fieldDim.y,fieldDim.z));
   //
   //Coordinates3D<double> fieldDimTrans_1= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,fieldDim.y-1,fieldDim.z-1));

   Coordinates3D<double> fieldDimTrans= boundaryStrategy->getLatticeSizeVector();
   
   Coordinates3D<double> fieldDimTrans_1= boundaryStrategy->getLatticeSpanVector();


   Coordinates3D<double> shiftVec;


   //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
   shiftVec.x= (x0-((int)fieldDimTrans.x)/2)*_pbcX;
   shiftVec.y= (y0-((int)fieldDimTrans.y)/2)*_pbcY;
   shiftVec.z= (z0-((int)fieldDimTrans.z)/2)*_pbcZ;
   
   // cerr<<"THIS IS SHIFT VEC="<<shiftVec<<endl;

   //moving x0,y0,z0 to approximetely center of the lattice
   x0_t=x0-shiftVec.x;
   y0_t=y0-shiftVec.y;
   z0_t=z0-shiftVec.z;

   //shifting accordingly other coordinates
   x1_t=x1-shiftVec.x;
   y1_t=y1-shiftVec.y;
   z1_t=z1-shiftVec.z;

    //making sure that x1_t,y1_t,z1_t is in the lattice

    if(x1_t < 0){
      x1_t += fieldDimTrans.x;
    }else if (x1_t > fieldDimTrans_1.x){
      x1_t -= fieldDimTrans.x;
    }  

    if(y1_t < 0){
      y1_t += fieldDimTrans.y;
    }else if (y1_t > fieldDimTrans_1.y){
      y1_t -= fieldDimTrans.y;
    }  

    if(z1_t < 0){
      z1_t += fieldDimTrans.z;
    }else if (z1_t > fieldDimTrans_1.z){
      z1_t -= fieldDimTrans.z;
    }  

   // cerr<<" x0_t="<<x0_t<<" y0_t="<<y0_t<<" z0_t="<<z0_t<<endl;
   // cerr<<" x1_t="<<x1_t<<" y1_t="<<y1_t<<" z1_t="<<z1_t<<endl;
   
   return dist(x0_t,y0_t,z0_t,x1_t,y1_t,z1_t);


}

double distInvariantCM(double x0,double y0,double z0,double x1,double y1,double z1,const Point3D & fieldDim, BoundaryStrategy *boundaryStrategy){

   
   double x0_t,y0_t,z0_t;
   double x1_t,y1_t,z1_t;


////    Coordinates3D<double> ptTrans=boundaryStrategy->calculatePointCoordinates(pt);
//   Coordinates3D<double> fieldDimTrans= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x,fieldDim.y,fieldDim.z));
//   // cerr<<"fieldDim="<<fieldDim<<endl;
//   // cerr<<"fieldDimTrans"<<fieldDimTrans<<endl;
//   
//   Coordinates3D<double> fieldDimTrans_1= boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,fieldDim.y-1,fieldDim.z-1));


   Coordinates3D<double> fieldDimTrans= boundaryStrategy->getLatticeSizeVector();
   
   Coordinates3D<double> fieldDimTrans_1= boundaryStrategy->getLatticeSpanVector();

   Coordinates3D<double> shiftVec;
//    Coordinates3D<double> shiftedPt;

//    Coordinates3D<double> distanceVecMin;
//    //determines minimum coordinates for the perpendicular lines paccinig through pt
//    Coordinates3D<double> distanceVecMax;
//    Coordinates3D<double> distanceVecMax_1;
//    //determines minimum coordinates for the perpendicular lines paccinig through pt
// 
//    Coordinates3D<double> distanceVec; //measures lattice diatances along x,y,z - they can be different for different lattices. The lines have to pass through pt
// 
//    distanceVecMin.x=boundaryStrategy->calculatePointCoordinates(Point3D(0,pt.y,pt.z)).x;
//    distanceVecMin.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,0,pt.z)).y;
//    distanceVecMin.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,0)).z;
// 
//    distanceVecMax.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x,pt.y,pt.z)).x;
//    distanceVecMax.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y,pt.z)).y;
//    distanceVecMax.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z)).z;
// 
//    distanceVecMax_1.x=boundaryStrategy->calculatePointCoordinates(Point3D(fieldDim.x-1,pt.y,pt.z)).x;
//    distanceVecMax_1.y=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,fieldDim.y-1,pt.z)).y;
//    distanceVecMax_1.z=boundaryStrategy->calculatePointCoordinates(Point3D(pt.x,pt.y,fieldDim.z-1)).z;
// 
// 
//    distanceVec=distanceVecMax-distanceVecMin;


   //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
   shiftVec.x= (x0-((int)fieldDimTrans.x)/2);
   shiftVec.y= (y0-((int)fieldDimTrans.y)/2);
   shiftVec.z= (z0-((int)fieldDimTrans.z)/2);
   //moving x0,y0,z0 to approximetely center of the lattice
   x0_t=x0-shiftVec.x;
   y0_t=y0-shiftVec.y;
   z0_t=z0-shiftVec.z;

   //shifting accordingly other coordinates
   x1_t=x1-shiftVec.x;
   y1_t=y1-shiftVec.y;
   z1_t=z1-shiftVec.z;

    //making sure that x1_t,y1_t,z1_t is in the lattice

    if(x1_t < 0){
      x1_t += fieldDimTrans.x;
    }else if (x1_t > fieldDimTrans_1.x){
      x1_t -= fieldDimTrans.x;
    }  

    if(y1_t < 0){
      y1_t += fieldDimTrans.y;
    }else if (y1_t > fieldDimTrans_1.y){
      y1_t -= fieldDimTrans.y;
    }  

    if(z1_t < 0){
      z1_t += fieldDimTrans.z;
    }else if (z1_t > fieldDimTrans_1.z){
      z1_t -= fieldDimTrans.z;
    }  

   return dist(x0_t,y0_t,z0_t,x1_t,y1_t,z1_t);

//    Point3D shiftVec;
//    double x0_t,y0_t,z0_t;
//    double x1_t,y1_t,z1_t;
// 
//    //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
//    shiftVec.x= (short)(x0-fieldDim.x/2);
//    shiftVec.y= (short)(y0-fieldDim.y/2);
//    shiftVec.z= (short)(z0-fieldDim.z/2);
// 
//    //moving x0,y0,z0 to approximetely center of the lattice
//    x0_t=x0-shiftVec.x;
//    y0_t=y0-shiftVec.y;
//    z0_t=z0-shiftVec.z;
// 
//    //shifting accordingly other coordinates
//    x1_t=x1-shiftVec.x;
//    y1_t=y1-shiftVec.y;
//    z1_t=z1-shiftVec.z;
// 
//     //making sure that x1_t,y1_t,z1_t is in the lattice
// 
//     if(x1_t < 0){
//       x1_t += fieldDim.x;
//     }else if (x1_t > fieldDim.x-1){
//       x1_t -= fieldDim.x;
//     }  
// 
//     if(y1_t < 0){
//       y1_t += fieldDim.y;
//     }else if (y1_t > fieldDim.y-1){
//       y1_t -= fieldDim.y;
//     }  
// 
//     if(z1_t < 0){
//       z1_t += fieldDim.z;
//     }else if (z1_t > fieldDim.z-1){
//       z1_t -= fieldDim.z;
//     }  
// 
//    return dist(x0_t,y0_t,z0_t,x1_t,y1_t,z1_t);


}


//returns invariant distance vector - v=_pt1-_pt0 that it "immune" to periodic boundary conditions

Point3D distanceVectorInvariant(const Point3D & _pt1 ,const Point3D & _pt0,const Point3D & _fieldDim){

   Point3D shiftVec;
   double x0_t,y0_t,z0_t;
   double x1_t,y1_t,z1_t;

   //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
   shiftVec.x= (short)(_pt0.x-_fieldDim.x/2);
   shiftVec.y= (short)(_pt0.y-_fieldDim.y/2);
   shiftVec.z= (short)(_pt0.z-_fieldDim.z/2);

   //moving x0,y0,z0 to approximetely center of the lattice
   x0_t=_pt0.x-shiftVec.x;
   y0_t=_pt0.y-shiftVec.y;
   z0_t=_pt0.z-shiftVec.z;

   //shifting accordingly other coordinates
   x1_t=_pt1.x-shiftVec.x;
   y1_t=_pt1.y-shiftVec.y;
   z1_t=_pt1.z-shiftVec.z;

    //making sure that x1_t,y1_t,z1_t is in the lattice

    if(x1_t < 0){
      x1_t += _fieldDim.x;
    }else if (x1_t > _fieldDim.x-1){
      x1_t -= _fieldDim.x;
    }  

    if(y1_t < 0){
      y1_t += _fieldDim.y;
    }else if (y1_t > _fieldDim.y-1){
      y1_t -= _fieldDim.y;
    }  

    if(z1_t < 0){
      z1_t += _fieldDim.z;
    }else if (z1_t > _fieldDim.z-1){
      z1_t -= _fieldDim.z;
    }  


   return Point3D(static_cast<short>(x1_t-x0_t),static_cast<short>(y1_t-y0_t),static_cast<short>(z1_t-z0_t));
   

}

Coordinates3D<double> distanceVectorCoordinatesInvariant(const Coordinates3D<double> & _pt1 ,const Coordinates3D<double> & _pt0,const Point3D & _fieldDim){

   Coordinates3D<double> shiftVec;
   double x0_t,y0_t,z0_t;
   double x1_t,y1_t,z1_t;

   //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
   shiftVec.x= (_pt0.x-_fieldDim.x/2);
   shiftVec.y= (_pt0.y-_fieldDim.y/2);
   shiftVec.z= (_pt0.z-_fieldDim.z/2);

   //moving x0,y0,z0 to approximetely center of the lattice
   x0_t=_pt0.x-shiftVec.x;
   y0_t=_pt0.y-shiftVec.y;
   z0_t=_pt0.z-shiftVec.z;

   //shifting accordingly other coordinates
   x1_t=_pt1.x-shiftVec.x;
   y1_t=_pt1.y-shiftVec.y;
   z1_t=_pt1.z-shiftVec.z;

    //making sure that x1_t,y1_t,z1_t is in the lattice

    if(x1_t < 0){
      x1_t += _fieldDim.x;
    }else if (x1_t > _fieldDim.x-1){
      x1_t -= _fieldDim.x;
    }  

    if(y1_t < 0){
      y1_t += _fieldDim.y;
    }else if (y1_t > _fieldDim.y-1){
      y1_t -= _fieldDim.y;
    }  

    if(z1_t < 0){
      z1_t += _fieldDim.z;
    }else if (z1_t > _fieldDim.z-1){
      z1_t -= _fieldDim.z;
    }  


   return Coordinates3D<double>(x1_t-x0_t , y1_t-y0_t , z1_t-z0_t);



}

//works only on a square lattice for now
CenterOfMassPair_t precalculateAfterFlipCM(
                  const Point3D &pt, 
                  const CellG *newCell, 
                  const CellG *oldCell,
                  const Point3D & fieldDim, 
                  const Point3D & boundaryConditionIndicator
                  )

{

   CenterOfMassPair_t centerOfMassPair;
   centerOfMassPair.first = Coordinates3D<float>(0.,0.,0.);
   centerOfMassPair.second = Coordinates3D<float>(0.,0.,0.);

   Coordinates3D<float> &newCellCM=centerOfMassPair.first;
   Coordinates3D<float> &oldCellCM=centerOfMassPair.second;

   //if no boundary conditions are present
   if ( !boundaryConditionIndicator.x && !boundaryConditionIndicator.y && !boundaryConditionIndicator.z ){

      
      if (oldCell) {
         oldCellCM.XRef()=oldCell->xCM-pt.x;
         oldCellCM.YRef()=oldCell->xCM-pt.y;
         oldCellCM.ZRef()=oldCell->xCM-pt.z;

         if(oldCell->volume>1){
            oldCellCM.XRef()=(oldCell->xCM-pt.x)/((float)oldCell->volume-1);
            oldCellCM.YRef()=(oldCell->yCM-pt.y)/((float)oldCell->volume-1);
            oldCellCM.ZRef()=(oldCell->zCM-pt.z)/((float)oldCell->volume-1);
         }else{
         
            oldCellCM.XRef()=oldCell->xCM;
            oldCellCM.YRef()=oldCell->xCM;
            oldCellCM.ZRef()=oldCell->xCM;
         
            

         }

      }

      if (newCell) {
      
         newCellCM.XRef()=(newCell->xCM+pt.x)/((float)newCell->volume+1);
         newCellCM.YRef()=(newCell->xCM+pt.y)/((float)newCell->volume+1);
         newCellCM.ZRef()=(newCell->xCM+pt.z)/((float)newCell->volume+1);
      

      }

      return centerOfMassPair;
   }

   //if there are boundary conditions defined that we have to do some shifts to correctly calculate center of mass
   //This approach will work only for cells whose span is much smaller that lattice dimension in the "periodic "direction
   //e.g. cell that is very long and "wraps lattice" will have miscalculated CM using this algorithm. On the other hand, you do not real expect
   //cells to have dimensions comparable to lattice...
   

   
   Point3D shiftVec;
   Point3D shiftedPt;
   int xCM,yCM,zCM; //temp centroids

   int x,y,z;
   int xo,yo,zo;
//     cerr<<"CM PLUGIN"<<endl;
 
  if (oldCell) {

   xo=oldCell->xCM;
   yo=oldCell->yCM;
   zo=oldCell->zCM;

        

      x=oldCell->xCM-pt.x;
      y=oldCell->yCM-pt.y;
      z=oldCell->zCM-pt.z;
    //calculating shiftVec - to translate CM


    //shift is defined to be zero vector for non-periodic b.c. - everything reduces to naive calculations then   
    shiftVec.x= (short)((oldCell->xCM/(float)(oldCell->volume)-fieldDim.x/2)*boundaryConditionIndicator.x);
    shiftVec.y= (short)((oldCell->yCM/(float)(oldCell->volume)-fieldDim.y/2)*boundaryConditionIndicator.y);
    shiftVec.z= (short)((oldCell->zCM/(float)(oldCell->volume)-fieldDim.z/2)*boundaryConditionIndicator.z);

    //shift CM to approximately center of lattice, new centroids are:
    xCM = oldCell->xCM - shiftVec.x*(oldCell->volume);
    yCM = oldCell->yCM - shiftVec.y*(oldCell->volume);
    zCM = oldCell->zCM - shiftVec.z*(oldCell->volume);
    //Now shift pt
    shiftedPt=pt;
    shiftedPt-=shiftVec;
    
    //making sure that shifterd point is in the lattice
    if(shiftedPt.x < 0){
      shiftedPt.x += fieldDim.x;
    }else if (shiftedPt.x > fieldDim.x-1){
      shiftedPt.x -= fieldDim.x;
    }  

    if(shiftedPt.y < 0){
      shiftedPt.y += fieldDim.y;
    }else if (shiftedPt.y > fieldDim.y-1){
      shiftedPt.y -= fieldDim.y;
    }  

    if(shiftedPt.z < 0){
      shiftedPt.z += fieldDim.z;
    }else if (shiftedPt.z > fieldDim.z-1){
      shiftedPt.z -= fieldDim.z;
    }
    //update shifted centroids
    xCM -= shiftedPt.x;
    yCM -= shiftedPt.y;
    zCM -= shiftedPt.z;

    //shift back centroids
    xCM += shiftVec.x * (oldCell->volume-1);
    yCM += shiftVec.y * (oldCell->volume-1);
    zCM += shiftVec.z * (oldCell->volume-1);

    //Check if CM is in the lattice
    if( xCM/((float)oldCell->volume-1) < 0){
      xCM += fieldDim.x*(oldCell->volume-1);
    }else if ( xCM/((float)oldCell->volume -1)> fieldDim.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
                                                         //to avoid rollovers for unsigned int from oldCell->xCM
                                                         

       xCM -= fieldDim.x*(oldCell->volume-1);

     
    }

    if( yCM/((float)oldCell->volume-1) < 0){
      yCM += fieldDim.y*(oldCell->volume-1);
    }else if ( yCM/((float)oldCell->volume-1) > fieldDim.y){
      yCM -= fieldDim.y*(oldCell->volume-1);
    }

    if( zCM/((float)oldCell->volume-1) < 0){
      zCM += fieldDim.z*(oldCell->volume-1);
    }else if ( zCM/((float)oldCell->volume-1) > fieldDim.z){
      zCM -= fieldDim.z*(oldCell->volume-1);
    }

   
   if(oldCell->volume>1){
      oldCellCM.XRef()=xCM/((float)oldCell->volume-1);
      oldCellCM.YRef()=yCM/((float)oldCell->volume-1);
      oldCellCM.ZRef()=zCM/((float)oldCell->volume-1);
   }else{
   
      oldCellCM.XRef()=zCM;
      oldCellCM.YRef()=yCM;
      oldCellCM.ZRef()=zCM;

   }


  }

  if (newCell) {

    xo=newCell->xCM;
    yo=newCell->yCM;
    zo=newCell->zCM;


      x=newCell->xCM+pt.x;
      y=newCell->yCM+pt.y;
      z=newCell->zCM+pt.z;

  
    if(newCell->volume==1){
      shiftVec.x=0;
      shiftVec.y=0;
      shiftVec.z=0;
    }else{
      shiftVec.x= (short)((newCell->xCM/(float)(newCell->volume)-fieldDim.x/2)*boundaryConditionIndicator.x);
      shiftVec.y= (short)((newCell->yCM/(float)(newCell->volume)-fieldDim.y/2)*boundaryConditionIndicator.y);
      shiftVec.z= (short)((newCell->zCM/(float)(newCell->volume)-fieldDim.z/2)*boundaryConditionIndicator.z);
    
      
    }
    
    //if CM of the cell is too close to the "middle" of the lattice correct shift vector

    
    //shift CM to approximately center of lattice , new centroids are:
    xCM = newCell->xCM - shiftVec.x*(newCell->volume);
    yCM = newCell->yCM - shiftVec.y*(newCell->volume);
    zCM = newCell->zCM - shiftVec.z*(newCell->volume);
    //Now shift pt
    shiftedPt=pt;
    shiftedPt-=shiftVec;

    //making sure that shifted point is in the lattice
    if(shiftedPt.x < 0){
      shiftedPt.x += fieldDim.x;
    }else if (shiftedPt.x > fieldDim.x-1){
//       cerr<<"shifted pt="<<shiftedPt<<endl;
      shiftedPt.x -= fieldDim.x;
    }  

    if(shiftedPt.y < 0){
      shiftedPt.y += fieldDim.y;
    }else if (shiftedPt.y > fieldDim.y-1){
      shiftedPt.y -= fieldDim.y;
    }  

    if(shiftedPt.z < 0){
      shiftedPt.z += fieldDim.z;
    }else if (shiftedPt.z > fieldDim.z-1){
      shiftedPt.z -= fieldDim.z;
    }    

    //update shifted centroids
    xCM += shiftedPt.x;
    yCM += shiftedPt.y;
    zCM += shiftedPt.z;
    
    //shift back centroids
    xCM += shiftVec.x * (newCell->volume+1);
    yCM += shiftVec.y * (newCell->volume+1);
    zCM += shiftVec.z * (newCell->volume+1);
    
    //Check if CM is in the lattice
    if( xCM/((float)newCell->volume+1) < 0){
      xCM += fieldDim.x*(newCell->volume+1);
    }else if ( xCM/((float)newCell->volume+1) > fieldDim.x){ //will allow to have xCM/vol slightly bigger (by 1) value than max lattice point
                                                         //to avoid rollovers for unsigned int from oldCell->xCM
      xCM -= fieldDim.x*(newCell->volume+1);
    }

    if( yCM/((float)newCell->volume+1) < 0){
      yCM += fieldDim.y*(newCell->volume+1);
    }else if ( yCM/((float)newCell->volume+1) > fieldDim.y){
      yCM -= fieldDim.y*(newCell->volume+1);
    }

    if( zCM/((float)newCell->volume+1) < 0){
      zCM += fieldDim.z*(newCell->volume+1);
    }else if ( zCM/((float)newCell->volume+1) > fieldDim.z){
      zCM -= fieldDim.z*(newCell->volume+1);
    }

   
   newCellCM.XRef()=xCM/((float)newCell->volume+1);
   newCellCM.YRef()=yCM/((float)newCell->volume+1);
   newCellCM.ZRef()=zCM/((float)newCell->volume+1);

   return centerOfMassPair;
    

  }

}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::pair<InertiaTensorComponents,InertiaTensorComponents>precalculateInertiaTensorComponentsAfterFlip(const Coordinates3D<double> & ptTrans,const CellG *newCell ,const CellG *oldCell){
	// Assumption: COM and Volume has not been updated.
	InertiaTensorComponents oldCellInertiaTensor;
	InertiaTensorComponents newCellInertiaTensor;

	if (newCell != 0)
	{

		double xcmOld,ycmOld,zcmOld,xcm,ycm,zcm;

		if(newCell->volume>0){
			xcmOld = (newCell->xCM / (float) newCell->volume);
			ycmOld = (newCell->yCM / (float) newCell->volume);
			zcmOld = (newCell->zCM / (float) newCell->volume);

			xcm = (newCell->xCM + ptTrans.x)/((float)newCell->volume + 1);
			ycm = (newCell->yCM + ptTrans.y)/((float)newCell->volume + 1);
			zcm = (newCell->zCM + ptTrans.z)/((float)newCell->volume + 1);
		}else{
			xcmOld=0.0;
			ycmOld=0.0;
			zcmOld=0.0;
			xcm = ptTrans.x;
			ycm = ptTrans.y;
			zcm = ptTrans.z;
		}

		newCellInertiaTensor.iXX=newCell->iXX+(newCell->volume )*(ycmOld*ycmOld+zcmOld*zcmOld)-(newCell->volume+1)*(ycm*ycm+zcm*zcm)+ptTrans.y*ptTrans.y+ptTrans.z*ptTrans.z;
		newCellInertiaTensor.iYY=newCell->iYY+(newCell->volume )*(xcmOld*xcmOld+zcmOld*zcmOld)-(newCell->volume+1)*(xcm*xcm+zcm*zcm)+ptTrans.x*ptTrans.x+ptTrans.z*ptTrans.z;
		newCellInertiaTensor.iZZ=newCell->iZZ+(newCell->volume )*(xcmOld*xcmOld+ycmOld*ycmOld)-(newCell->volume+1)*(xcm*xcm+ycm*ycm)+ptTrans.x*ptTrans.x+ptTrans.y*ptTrans.y;

		newCellInertiaTensor.iXY=newCell->iXY-(newCell->volume )*xcmOld*ycmOld+(newCell->volume+1)*xcm*ycm-ptTrans.x*ptTrans.y;
		newCellInertiaTensor.iXZ=newCell->iXZ-(newCell->volume )*xcmOld*zcmOld+(newCell->volume+1)*xcm*zcm-ptTrans.x*ptTrans.z;	
		newCellInertiaTensor.iYZ=newCell->iYZ-(newCell->volume )*ycmOld*zcmOld+(newCell->volume+1)*ycm*zcm-ptTrans.y*ptTrans.z;	

	} 

	if (oldCell != 0)
	{

		double xcmOld,ycmOld,zcmOld,xcm,ycm,zcm;

		if(oldCell->volume>1){
			xcmOld = (oldCell->xCM / (float) oldCell->volume);
			ycmOld = (oldCell->yCM / (float) oldCell->volume);
			zcmOld = (oldCell->zCM / (float) oldCell->volume);

			xcm = (oldCell->xCM - ptTrans.x)/((float)oldCell->volume - 1);
			ycm = (oldCell->yCM - ptTrans.y)/((float)oldCell->volume - 1);
			zcm = (oldCell->zCM - ptTrans.z)/((float)oldCell->volume - 1);
		}else{
			xcmOld = (oldCell->xCM / (float) oldCell->volume);
			ycmOld = (oldCell->yCM / (float) oldCell->volume);
			zcmOld = (oldCell->zCM / (float) oldCell->volume);
			xcm = 0.0;
			ycm = 0.0;
			zcm = 0.0;

		}

		oldCellInertiaTensor.iXX= oldCell->iXX+(oldCell->volume )*(ycmOld*ycmOld+zcmOld*zcmOld)-(oldCell->volume-1)*(ycm*ycm+zcm*zcm)-ptTrans.y*ptTrans.y-ptTrans.z*ptTrans.z;
		oldCellInertiaTensor.iYY = oldCell->iYY+(oldCell->volume )*(xcmOld*xcmOld+zcmOld*zcmOld)-(oldCell->volume-1)*(xcm*xcm+zcm*zcm)-ptTrans.x*ptTrans.x-ptTrans.z*ptTrans.z;
		oldCellInertiaTensor.iZZ = oldCell->iZZ+(oldCell->volume )*(xcmOld*xcmOld+ycmOld*ycmOld)-(oldCell->volume-1)*(xcm*xcm+ycm*ycm)-ptTrans.x*ptTrans.x-ptTrans.y*ptTrans.y;

		oldCellInertiaTensor.iXY = oldCell->iXY-(oldCell->volume )*xcmOld*ycmOld+(oldCell->volume-1)*xcm*ycm+ptTrans.x*ptTrans.y;
		oldCellInertiaTensor.iXZ = oldCell->iXZ-(oldCell->volume )*xcmOld*zcmOld+(oldCell->volume-1)*xcm*zcm+ptTrans.x*ptTrans.z;	
		oldCellInertiaTensor.iYZ = oldCell->iYZ-(oldCell->volume )*ycmOld*zcmOld+(oldCell->volume-1)*ycm*zcm+ptTrans.y*ptTrans.z;		
	   
		
	} 
	return make_pair(newCellInertiaTensor,oldCellInertiaTensor);
}


};
