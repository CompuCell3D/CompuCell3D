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

#include <CompuCell3D/CC3D.h>
// // // #include <CompuCell3D/Field3D/Field3D.h>
// // // #include <CompuCell3D/Field3D/WatchableField3D.h>
// // // #include <CompuCell3D/Simulator.h>
// // // #include <CompuCell3D/Automaton/Automaton.h>
using namespace CompuCell3D;


// // // #include <BasicUtils/BasicString.h>
// // // #include <BasicUtils/BasicException.h>



#include "OrientedContactPlugin.h"




OrientedContactPlugin::OrientedContactPlugin() :xmlData(0),depth(1),alpha(1.0),weightDistance(false) {
}

OrientedContactPlugin::~OrientedContactPlugin() {
 
}


double OrientedContactPlugin::getOrientation(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
   double ecc1, ecc2, v1x, v1y, v2x,v2y, r1x,r1y,r2x,r2y,s01,s02, E1, E2, deltaE;
   double ecc1_before, ecc2_before, v1x_before, v1y_before, v2x_before,v2y_before, s01_before,s02_before;
    
   if (newCell != 0)
   {
       // Assumption: COM and Volume has not been updated.
        
      double xPtSum = newCell->xCM;
      double yPtSum = newCell->yCM;
      double zPtSum = newCell->zCM;
        
      double xcm = (newCell->xCM / (float) newCell->volume);
      double ycm = (newCell->yCM / (float) newCell->volume);
      double zcm = (newCell->zCM / (float) newCell->volume);
        
      double eq1 = newCell->iYY + newCell->volume*zcm*zcm + newCell->volume*xcm*xcm;
      double eq2 = newCell->iXX + newCell->volume*zcm*zcm + newCell->volume*ycm*ycm; 
      double eq3 = newCell->iZZ + newCell->volume*ycm*ycm + newCell->volume*xcm*xcm;
        
      double xPtSumSQ = (eq1+eq3-eq2)/2.0; 
      double zPtSumSQ = (eq1+eq2-eq3)/2.0;
      double yPtSumSQ = (eq2+eq3-eq1)/2.0;
        
      double yzSum = (newCell->iYZ - ycm*zPtSum - zcm*yPtSum + newCell->volume*ycm*zcm) / -1.0; 
      double xzSum = (newCell->iXZ - xcm*zPtSum - zcm*xPtSum + newCell->volume*xcm*zcm) / -1.0; 
      double xySum = (newCell->iXY - xcm*yPtSum - xcm*yPtSum + newCell->volume*xcm*ycm) / -1.0; 
        
        
        
      double newXCM = (newCell->xCM + pt.x)/((float)newCell->volume + 1);
      double newYCM = (newCell->yCM + pt.y)/((float)newCell->volume + 1);
      double newZCM = (newCell->zCM + pt.z)/((float)newCell->volume + 1);
        
      xPtSum += pt.x;
      yPtSum += pt.y;
      zPtSum += pt.z;
        
        
      xPtSumSQ += pt.x*pt.x; 
      yPtSumSQ += pt.y*pt.y;
      zPtSumSQ += pt.z*pt.z;
        
      yzSum += pt.y*pt.z;
      xzSum += pt.x*pt.z;
      xySum += pt.x*pt.y;
        
      double newIxx = zPtSumSQ + yPtSumSQ - (newZCM*zPtSum+newYCM*yPtSum);
      double newIyy = zPtSumSQ + xPtSumSQ - (newZCM*zPtSum+newXCM*xPtSum);
      double newIzz = yPtSumSQ + xPtSumSQ - (newYCM*yPtSum + newXCM*xPtSum);
        
      double newIyz = -yzSum + newYCM*zPtSum + newZCM*yPtSum - (newCell->volume+1)*newYCM*newZCM;
      double newIxz = -xzSum + newXCM*zPtSum + newZCM*xPtSum - (newCell->volume+1)*newXCM*newZCM;
      double newIxy = -xySum + newYCM*xPtSum + newXCM*yPtSum - (newCell->volume+1)*newXCM*newYCM;
        
        
      double l1_max_before = .5*((newCell->iXX+newCell->iYY)+sqrt((newCell->iXX-newCell->iYY)*(newCell->iXX-newCell->iYY)+4.0*newCell->iXY*newCell->iXY));
      double l1_min_before = .5*((newCell->iXX+newCell->iYY)-sqrt((newCell->iXX-newCell->iYY)*(newCell->iXX-newCell->iYY)+4.0*newCell->iXY*newCell->iXY));
      v1y_before = .5*((newCell->iXX-newCell->iYY)-sqrt((newCell->iXX-newCell->iYY)*(newCell->iXX-newCell->iYY)+4.0*newCell->iXY*newCell->iXY));
      v1x_before = newCell->iXY;
      ecc1_before = sqrt(1.0-l1_min_before/l1_max_before);
        /*cerr << "  l1_min_before: " << l1_min_before <<  "  l1_max_before: " << l1_max_before<< endl;
      cerr << "  iXX" << newCell->iXX << "  iXY" << newCell->iXY << endl;
        */ 
      double l1_max = .5*((newIxx+newIyy)+sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      double l1_min = .5*((newIxx+newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v1y = .5*((newIxx-newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v1x = newIxy;
      r1x = pt.x-xcm;
      r1y = pt.y-ycm;
      ecc1 = sqrt(1.0-l1_min/l1_max);
//         cerr << "  l1_min: " << l1_min <<  "  l1_max: " << l1_max<< endl;
	

   }
    
   else { 
      ecc1 = 0;
      ecc1_before = 0;
   }
   if (oldCell != 0)
   {
      double xPtSum = oldCell->xCM;
      double yPtSum = oldCell->yCM;
      double zPtSum = oldCell->zCM;
       
      double xcm = (oldCell->xCM / (float) oldCell->volume);
      double ycm = (oldCell->yCM / (float) oldCell->volume);
      double zcm = (oldCell->zCM / (float) oldCell->volume);
        
      double eq1 = oldCell->iYY + oldCell->volume*zcm*zcm + oldCell->volume*xcm*xcm;
      double eq2 = oldCell->iXX + oldCell->volume*zcm*zcm + oldCell->volume*ycm*ycm; 
      double eq3 = oldCell->iZZ + oldCell->volume*ycm*ycm + oldCell->volume*xcm*xcm;
        
      double xPtSumSQ = (eq1+eq3-eq2)/2.0; 
      double zPtSumSQ = (eq1+eq2-eq3)/2.0;
      double yPtSumSQ = (eq2+eq3-eq1)/2.0;
        
      double yzSum = (oldCell->iYZ - ycm*zPtSum - zcm*yPtSum + oldCell->volume*ycm*zcm) / -1.0; 
      double xzSum = (oldCell->iXZ - xcm*zPtSum - zcm*xPtSum + oldCell->volume*xcm*zcm) / -1.0; 
      double xySum = (oldCell->iXY - xcm*yPtSum - xcm*yPtSum + oldCell->volume*xcm*ycm) / -1.0; 
        
        
      double newXCM = (oldCell->xCM - pt.x)/((float)oldCell->volume - 1);
      double newYCM = (oldCell->yCM - pt.y)/((float)oldCell->volume - 1);
      double newZCM = (oldCell->zCM - pt.z)/((float)oldCell->volume - 1);
        
      xPtSum -= pt.x;
      yPtSum -= pt.y;
      zPtSum -= pt.z;
        
        
      xPtSumSQ += pt.x*pt.x; 
      yPtSumSQ += pt.y*pt.y;
      zPtSumSQ += pt.z*pt.z;
        
      yzSum += pt.y*pt.z;
      xzSum += pt.x*pt.z;
      xySum += pt.x*pt.y;
        
      double newIxx = zPtSumSQ + yPtSumSQ - (newZCM*zPtSum+newYCM*yPtSum);
      double newIyy = zPtSumSQ + xPtSumSQ - (newZCM*zPtSum+newXCM*xPtSum);
      double newIzz = yPtSumSQ + xPtSumSQ - (newYCM*yPtSum + newXCM*xPtSum);
        
      double newIyz = -yzSum + newYCM*zPtSum + newZCM*yPtSum - (oldCell->volume-1)*newYCM*newZCM;
      double newIxz = -xzSum + newXCM*zPtSum + newZCM*xPtSum - (oldCell->volume-1)*newXCM*newZCM;
      double newIxy = -xySum + newYCM*xPtSum + newXCM*yPtSum - (oldCell->volume-1)*newXCM*newYCM;
        
        
      double l2_max_before = .5*((oldCell->iXX+oldCell->iYY)+sqrt((oldCell->iXX-oldCell->iYY)*(oldCell->iXX-oldCell->iYY)+4.0*oldCell->iXY*oldCell->iXY));
      double l2_min_before = .5*((oldCell->iXX+oldCell->iYY)-sqrt((oldCell->iXX-oldCell->iYY)*(oldCell->iXX-oldCell->iYY)+4.0*oldCell->iXY*oldCell->iXY));
      v2y_before = .5*((oldCell->iXX-oldCell->iYY)-sqrt((oldCell->iXX-oldCell->iYY)*(oldCell->iXX-oldCell->iYY)+4.0*oldCell->iXY*oldCell->iXY));
      v2x_before = oldCell->iXY;
      ecc2_before = sqrt(1.0-l2_min_before/l2_max_before);
	 
      double l2_max = .5*((newIxx+newIyy)+sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      double l2_min = .5*((newIxx+newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v2y = .5*((newIxx-newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v2x = newIxy;
      r2x = pt.x-xcm;
      r2y = pt.y-ycm;
      ecc2 = sqrt(1.0-l2_min/l2_max);
//        cerr << "l2_min: " << l2_min<< "  l2_max: " << l2_max << endl;
   }
        
   else {
      ecc2_before = 0;
      ecc2 = 0;
   }

   s01 = (r1x*v1x+r1y*v1y)/sqrt((r1x*r1x+r1y*r1y)*(v1x*v1x+v1y*v1y));
   s01 = sqrt(1.0-s01*s01);
   s02 = (r2x*v2x+r2y*v2y)/sqrt((r2x*r2x+r2y*r2y)*(v2x*v2x+v2y*v2y));
   s02 = sqrt(1.0-s02*s02);
        
   s01_before = (r1x*v1x_before+r1y*v1y_before)/sqrt((r1x*r1x+r1y*r1y)*(v1x_before*v1x_before+v1y_before*v1y_before));
   s01_before = sqrt(1.0-s01_before*s01_before);
   s02_before = (r2x*v2x_before+r2y*v2y_before)/sqrt((r2x*r2x+r2y*r2y)*(v2x_before*v2x_before+v2y_before*v2y_before));
   s02_before = sqrt(1.0-s02_before*s02_before);
        
   E1 = ecc1_before*ecc2_before*s01_before*s01_before*sqrt((r1x*r1x+r1y*r1y)*(r2x*r2x+r2y*r2y));
   E2 = ecc1*ecc2*s01*s01*sqrt((r1x*r1x+r1y*r1y)*(r2x*r2x+r2y*r2y));
        
   deltaE = E2-E1;
        
//         cerr << "ecc1: " << ecc1 << "  ecc2: " << ecc2 << "  ecc1_before: " << ecc1_before <<  "  ecc2_before: " << ecc2_before << endl;
        
       //       " r1x: " << r1x << " r1y: " << r1y << "  vx: " << v1x << "  v1y: " << v1y << endl;
//         cerr << "Sin theta: " << s01 << "  Energy: " << deltaE << endl; 
        
   if(deltaE != deltaE) { //check for nan
      return 0;
   }
   else{
      return deltaE;
   }
}

double OrientedContactPlugin::getMediumOrientation(const Point3D &pt, const CellG *newCell, const CellG *oldCell) {
   double ecc1, ecc2, v1x, v1y, v2x,v2y, r1x,r1y,r2x,r2y,s01,s02, E1, E2, deltaE;
   double ecc1_before, ecc2_before, v1x_before, v1y_before, v2x_before,v2y_before, s01_before,s02_before;

   if (!newCell && !oldCell) {
//       cerr << "only medium" << endl;
      return 1.0;
   }
    
    
   if (newCell != 0)
   {
       // Assumption: COM and Volume has not been updated.
        
      double xPtSum = newCell->xCM;
      double yPtSum = newCell->yCM;
      double zPtSum = newCell->zCM;
        
      double xcm = (newCell->xCM / (float) newCell->volume);
      double ycm = (newCell->yCM / (float) newCell->volume);
      double zcm = (newCell->zCM / (float) newCell->volume);
        
      double eq1 = newCell->iYY + newCell->volume*zcm*zcm + newCell->volume*xcm*xcm;
      double eq2 = newCell->iXX + newCell->volume*zcm*zcm + newCell->volume*ycm*ycm; 
      double eq3 = newCell->iZZ + newCell->volume*ycm*ycm + newCell->volume*xcm*xcm;
        
      double xPtSumSQ = (eq1+eq3-eq2)/2.0; 
      double zPtSumSQ = (eq1+eq2-eq3)/2.0;
      double yPtSumSQ = (eq2+eq3-eq1)/2.0;
        
      double yzSum = (newCell->iYZ - ycm*zPtSum - zcm*yPtSum + newCell->volume*ycm*zcm) / -1.0; 
      double xzSum = (newCell->iXZ - xcm*zPtSum - zcm*xPtSum + newCell->volume*xcm*zcm) / -1.0; 
      double xySum = (newCell->iXY - xcm*yPtSum - xcm*yPtSum + newCell->volume*xcm*ycm) / -1.0; 
        
        
        
      double newXCM = (newCell->xCM + pt.x)/((float)newCell->volume + 1);
      double newYCM = (newCell->yCM + pt.y)/((float)newCell->volume + 1);
      double newZCM = (newCell->zCM + pt.z)/((float)newCell->volume + 1);
        
      xPtSum += pt.x;
      yPtSum += pt.y;
      zPtSum += pt.z;
        
        
      xPtSumSQ += pt.x*pt.x; 
      yPtSumSQ += pt.y*pt.y;
      zPtSumSQ += pt.z*pt.z;
        
      yzSum += pt.y*pt.z;
      xzSum += pt.x*pt.z;
      xySum += pt.x*pt.y;
        
      double newIxx = zPtSumSQ + yPtSumSQ - (newZCM*zPtSum+newYCM*yPtSum);
      double newIyy = zPtSumSQ + xPtSumSQ - (newZCM*zPtSum+newXCM*xPtSum);
      double newIzz = yPtSumSQ + xPtSumSQ - (newYCM*yPtSum + newXCM*xPtSum);
        
      double newIyz = -yzSum + newYCM*zPtSum + newZCM*yPtSum - (newCell->volume+1)*newYCM*newZCM;
      double newIxz = -xzSum + newXCM*zPtSum + newZCM*xPtSum - (newCell->volume+1)*newXCM*newZCM;
      double newIxy = -xySum + newYCM*xPtSum + newXCM*yPtSum - (newCell->volume+1)*newXCM*newYCM;
        
        
      double l1_max_before = .5*((newCell->iXX+newCell->iYY)+sqrt((newCell->iXX-newCell->iYY)*(newCell->iXX-newCell->iYY)+4.0*newCell->iXY*newCell->iXY));
      double l1_min_before = .5*((newCell->iXX+newCell->iYY)-sqrt((newCell->iXX-newCell->iYY)*(newCell->iXX-newCell->iYY)+4.0*newCell->iXY*newCell->iXY));
      v1y_before = .5*((newCell->iXX-newCell->iYY)-sqrt((newCell->iXX-newCell->iYY)*(newCell->iXX-newCell->iYY)+4.0*newCell->iXY*newCell->iXY));
      v1x_before = newCell->iXY;
      ecc1_before = sqrt(1.0-l1_min_before/l1_max_before);
        /*cerr << "  l1_min_before: " << l1_min_before <<  "  l1_max_before: " << l1_max_before<< endl;
      cerr << "  iXX" << newCell->iXX << "  iXY" << newCell->iXY << endl;
        */ 
      double l1_max = .5*((newIxx+newIyy)+sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      double l1_min = .5*((newIxx+newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v1y = .5*((newIxx-newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v1x = newIxy;
      r1x = pt.x-xcm;
      r1y = pt.y-ycm;
      ecc1 = sqrt(1.0-l1_min/l1_max);
      
      s01 = (r1x*v1x+r1y*v1y)/sqrt((r1x*r1x+r1y*r1y)*(v1x*v1x+v1y*v1y));
      s01 = sqrt(1.0-s01*s01);
        
      s01_before = (r1x*v1x_before+r1y*v1y_before)/sqrt((r1x*r1x+r1y*r1y)*(v1x_before*v1x_before+v1y_before*v1y_before));
      s01_before = sqrt(1.0-s01_before*s01_before);
      
      double theta = asin(s01);
      double theta_before = asin(s01_before);
      
      E2 = ecc1*alpha*cos(theta);
      E1 = ecc1_before*alpha*cos(theta_before);
//       cerr << "NEW CELL GET:  " << theta << "\t " << alpha << endl;
      return E2-E1;
      
   }   

   if (oldCell != 0)
   {
      double xPtSum = oldCell->xCM;
      double yPtSum = oldCell->yCM;
      double zPtSum = oldCell->zCM;
       
      double xcm = (oldCell->xCM / (float) oldCell->volume);
      double ycm = (oldCell->yCM / (float) oldCell->volume);
      double zcm = (oldCell->zCM / (float) oldCell->volume);
        
      double eq1 = oldCell->iYY + oldCell->volume*zcm*zcm + oldCell->volume*xcm*xcm;
      double eq2 = oldCell->iXX + oldCell->volume*zcm*zcm + oldCell->volume*ycm*ycm; 
      double eq3 = oldCell->iZZ + oldCell->volume*ycm*ycm + oldCell->volume*xcm*xcm;
        
      double xPtSumSQ = (eq1+eq3-eq2)/2.0; 
      double zPtSumSQ = (eq1+eq2-eq3)/2.0;
      double yPtSumSQ = (eq2+eq3-eq1)/2.0;
        
      double yzSum = (oldCell->iYZ - ycm*zPtSum - zcm*yPtSum + oldCell->volume*ycm*zcm) / -1.0; 
      double xzSum = (oldCell->iXZ - xcm*zPtSum - zcm*xPtSum + oldCell->volume*xcm*zcm) / -1.0; 
      double xySum = (oldCell->iXY - xcm*yPtSum - xcm*yPtSum + oldCell->volume*xcm*ycm) / -1.0; 
        
        
      double newXCM = (oldCell->xCM - pt.x)/((float)oldCell->volume - 1);
      double newYCM = (oldCell->yCM - pt.y)/((float)oldCell->volume - 1);
      double newZCM = (oldCell->zCM - pt.z)/((float)oldCell->volume - 1);
        
      xPtSum -= pt.x;
      yPtSum -= pt.y;
      zPtSum -= pt.z;
        
        
      xPtSumSQ += pt.x*pt.x; 
      yPtSumSQ += pt.y*pt.y;
      zPtSumSQ += pt.z*pt.z;
        
      yzSum += pt.y*pt.z;
      xzSum += pt.x*pt.z;
      xySum += pt.x*pt.y;
        
      double newIxx = zPtSumSQ + yPtSumSQ - (newZCM*zPtSum+newYCM*yPtSum);
      double newIyy = zPtSumSQ + xPtSumSQ - (newZCM*zPtSum+newXCM*xPtSum);
      double newIzz = yPtSumSQ + xPtSumSQ - (newYCM*yPtSum + newXCM*xPtSum);
        
      double newIyz = -yzSum + newYCM*zPtSum + newZCM*yPtSum - (oldCell->volume-1)*newYCM*newZCM;
      double newIxz = -xzSum + newXCM*zPtSum + newZCM*xPtSum - (oldCell->volume-1)*newXCM*newZCM;
      double newIxy = -xySum + newYCM*xPtSum + newXCM*yPtSum - (oldCell->volume-1)*newXCM*newYCM;
        
        
      double l2_max_before = .5*((oldCell->iXX+oldCell->iYY)+sqrt((oldCell->iXX-oldCell->iYY)*(oldCell->iXX-oldCell->iYY)+4.0*oldCell->iXY*oldCell->iXY));
      double l2_min_before = .5*((oldCell->iXX+oldCell->iYY)-sqrt((oldCell->iXX-oldCell->iYY)*(oldCell->iXX-oldCell->iYY)+4.0*oldCell->iXY*oldCell->iXY));
      v2y_before = .5*((oldCell->iXX-oldCell->iYY)-sqrt((oldCell->iXX-oldCell->iYY)*(oldCell->iXX-oldCell->iYY)+4.0*oldCell->iXY*oldCell->iXY));
      v2x_before = oldCell->iXY;
      ecc2_before = sqrt(1.0-l2_min_before/l2_max_before);
	 
      double l2_max = .5*((newIxx+newIyy)+sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      double l2_min = .5*((newIxx+newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v2y = .5*((newIxx-newIyy)-sqrt((newIxx-newIyy)*(newIxx-newIyy)+4.0*newIxy*newIxy));
      v2x = newIxy;
      r2x = pt.x-xcm;
      r2y = pt.y-ycm;
      ecc2 = sqrt(1.0-l2_min/l2_max);
//        cerr << "l2_min: " << l2_min<< "  l2_max: " << l2_max << endl;
      s02 = (r2x*v2x+r2y*v2y)/sqrt((r2x*r2x+r2y*r2y)*(v2x*v2x+v2y*v2y));
      s02 = sqrt(1.0-s02*s02);
      s02_before = (r2x*v2x_before+r2y*v2y_before)/sqrt((r2x*r2x+r2y*r2y)*(v2x_before*v2x_before+v2y_before*v2y_before));
      s02_before = sqrt(1.0-s02_before*s02_before);
      
      double theta = asin(s02);
      double theta_before = asin(s02_before);
      
      E2 = ecc2*alpha*cos(theta);
      E1 = ecc2_before*alpha*cos(theta_before);
//       cerr << "OLD CELL GET:  " << theta << "\t " << alpha << endl;
      return E2-E1;
      
   }

    
   
        
}

double OrientedContactPlugin::changeEnergy(const Point3D &pt,
                                  const CellG *newCell,
                                  const CellG *oldCell) {
//    cerr<<"ChangeEnergy"<<endl;
   
   
  double energy = 0;
  unsigned int token = 0;
  double distance = 0;
  Point3D n;
  
  CellG *nCell=0;
  WatchableField3D<CellG *> *fieldG = (WatchableField3D<CellG *> *)potts->getCellFieldG();
  Neighbor neighbor;
  
     
      for(unsigned int nIdx=0 ; nIdx <= maxNeighborIndex ; ++nIdx ){
         neighbor=boundaryStrategy->getNeighborDirect(const_cast<Point3D&>(pt),nIdx);
         if(!neighbor.distance){
         //if distance is 0 then the neighbor returned is invalid
         continue;
         }
         
         nCell = fieldG->get(neighbor.pt);
         if(nCell!=oldCell){
            if(!nCell) {
               energy -= orientedContactEnergy(oldCell, nCell)*getMediumOrientation(pt,oldCell,nCell);
                
            }
            else {
               energy -= orientedContactEnergy(oldCell, nCell)+getOrientation(pt,oldCell,nCell);
            }
//             cerr<<"!=oldCell neighbor.pt="<<neighbor.pt<<" energyTmp="<<energy<<endl;
         }
         if(nCell!=newCell){
            if(!nCell) {
					energy += orientedContactEnergy(newCell, nCell)*getMediumOrientation(pt,newCell,nCell);
            }
            else {
               energy += orientedContactEnergy(newCell, nCell)+getOrientation(pt,newCell,nCell);
            }
         }
      
   
      }
   
   
   

  return energy;
}

double OrientedContactPlugin::orientedContactEnergy(const CellG *cell1, const CellG *cell2) {
   
   return orientedContactEnergyArray[cell1 ? cell1->type : 0][cell2? cell2->type : 0];


}

void OrientedContactPlugin::setOrientedContactEnergy(const string typeName1,
				     const string typeName2,
				     const double energy) {
                    
  char type1 = automaton->getTypeId(typeName1);
  char type2 = automaton->getTypeId(typeName2);
    
  int index = getIndex(type1, type2);

  orientedContactEnergies_t::iterator it = orientedContactEnergies.find(index);
  ASSERT_OR_THROW(string("OrientedOrientedContact energy for ") + typeName1 + " " + typeName2 +
		  " already set!", it == orientedContactEnergies.end());

  orientedContactEnergies[index] = energy;
}

int OrientedContactPlugin::getIndex(const int type1, const int type2) const {
  if (type1 < type2) return ((type1 + 1) | ((type2 + 1) << 16));
  else return ((type2 + 1) | ((type1 + 1) << 16));
}


void OrientedContactPlugin::init(Simulator *simulator, CC3DXMLElement *_xmlData) {

  xmlData=_xmlData;
  potts=simulator->getPotts();
  potts->registerEnergyFunctionWithName(this,"OrientedContact");
  simulator->registerSteerableObject(this);

}
void OrientedContactPlugin::extraInit(Simulator *simulator){
	update(xmlData,true);
}

void OrientedContactPlugin::update(CC3DXMLElement *_xmlData, bool _fullInitFlag){

	automaton = potts->getAutomaton();
	ASSERT_OR_THROW("CELL TYPE PLUGIN WAS NOT PROPERLY INITIALIZED YET. MAKE SURE THIS IS THE FIRST PLUGIN THAT YOU SET", automaton)
   set<unsigned char> cellTypesSet;

   orientedContactEnergies.clear();
   orientedContactEnergyArray.clear();

	CC3DXMLElementList energyVec=_xmlData->getElements("Energy");

	for (int i = 0 ; i<energyVec.size(); ++i){

		setOrientedContactEnergy(energyVec[i]->getAttribute("Type1"), energyVec[i]->getAttribute("Type2"), energyVec[i]->getDouble());

		//inserting all the types to the set (duplicate are automatically eleminated) to figure out max value of type Id
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type1")));
		cellTypesSet.insert(automaton->getTypeId(energyVec[i]->getAttribute("Type2")));

	}

	if(_xmlData->findElement("Alpha")){
		alpha=_xmlData->getFirstElement("Alpha")->getDouble();
	}

  //Now that we know all the types used in the simulation we will find size of the contactEnergyArray
  vector<unsigned char> cellTypesVector(cellTypesSet.begin(),cellTypesSet.end());//coping set to the vector

  int size= * max_element(cellTypesVector.begin(),cellTypesVector.end());
  size+=1;//if max element is e.g. 5 then size has to be 6 for an array to be properly allocated
  
  int index ;
  orientedContactEnergyArray.clear();
  orientedContactEnergyArray.assign(size,vector<double>(size,0.0));

  for(int i = 0 ; i < size ; ++i)
   for(int j = 0 ; j < size ; ++j){
   
      index = getIndex(cellTypesVector[i],cellTypesVector[j]);
      
      orientedContactEnergyArray[i][j] = orientedContactEnergies[index];
      
   }
   cerr<<"size="<<size<<endl;
   
  for(int i = 0 ; i < size ; ++i)
   for(int j = 0 ; j < size ; ++j){
   
      cerr<<"contact["<<i<<"]["<<j<<"]="<<orientedContactEnergyArray[i][j]<<endl;
      
   }
   
   //Here I initialize max neighbor index for direct acces to the list of neighbors 
   boundaryStrategy=BoundaryStrategy::getInstance();
   maxNeighborIndex=0;


			if(_xmlData->getFirstElement("Depth")){
				maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromDepth(_xmlData->getFirstElement("Depth")->getDouble());
				//cerr<<"got here will do depth"<<endl;
			}else{
				//cerr<<"got here will do neighbor order"<<endl;
				if(_xmlData->getFirstElement("NeighborOrder")){

					maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(_xmlData->getFirstElement("NeighborOrder")->getUInt());	
				}else{
					maxNeighborIndex=boundaryStrategy->getMaxNeighborIndexFromNeighborOrder(1);

				}

			}

			cerr<<"Contact maxNeighborIndex="<<maxNeighborIndex<<endl;

   
}

std::string OrientedContactPlugin::toString(){
	return "OrientedContact";
}
std::string OrientedContactPlugin::steerableName(){

   return toString();

}


