#ifndef CONTOURLINES_H
#define CONTOURLINES_H

#include <stdio.h>
#include <math.h>
#include  <vector>
class QPainter;

int conrec(std::vector<std::vector<float> > & d,//float ** d,
      int ilb,
      int iub,
      int jlb,
      int jub,
      float *x,
      float *y,
      int nc,
      float *z,
      QPainter * painter,
      int zoomFactor,
      double scaleX=1.0,
      double scaleY=1.0
      
      );

// #define xsect(p1,p2) (h[p2]*xh[p1]-h[p1]*xh[p2])/(h[p2]-h[p1])
// #define ysect(p1,p2) (h[p2]*yh[p1]-h[p1]*yh[p2])/(h[p2]-h[p1])
// #define min(x,y) (x<y?x:y)
// #define max(x,y) (x>y?x:y)

#endif
