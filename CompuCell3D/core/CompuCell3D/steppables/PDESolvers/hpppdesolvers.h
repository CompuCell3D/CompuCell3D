#ifndef HPPDESOLVERS_H
#define HPPDESOLVERS_H

#ifdef __cplusplus
extern "C" {
#endif



#include "f2c.h"


int hw3crt_(doublereal *xs, doublereal *xf, integer *l, integer *lbdcnd, 
doublereal *bdxs, doublereal *bdxf, doublereal *ys, doublereal *yf, integer *m, integer *
mbdcnd, doublereal *bdys, doublereal *bdyf, doublereal *zs, doublereal *zf, integer *n, 
integer *nbdcnd, doublereal *bdzs, doublereal *bdzf, doublereal *elmbda, integer *ldimf,
integer *mdimf, doublereal *f, doublereal *pertrb, integer *ierror, doublereal *w);

//int hw3crt_(real *xs, real *xf, integer *l, integer *lbdcnd, 
//real *bdxs, real *bdxf, real *ys, real *yf, integer *m, integer *
//mbdcnd, real *bdys, real *bdyf, real *zs, real *zf, integer *n, 
//integer *nbdcnd, real *bdzs, real *bdzf, real *elmbda, integer *ldimf,
//integer *mdimf, real *f, real *pertrb, integer *ierror, real *w);

// int hwscrt_(real *a, real *b, integer *m, integer *mbdcnd, 
// real *bda, real *bdb, real *c__, real *d__, integer *n, integer *
// nbdcnd, real *bdc, real *bdd, real *elmbda, real *f, integer *idimf, 
// real *pertrb, integer *ierror, real *w);

int hwscrt_(doublereal *a, doublereal *b, integer *m, integer *mbdcnd, 
doublereal *bda, doublereal *bdb, doublereal *c__, doublereal *d__, integer *n, integer *
nbdcnd, doublereal *bdc, doublereal *bdd, doublereal *elmbda, doublereal *f, integer *idimf, 
doublereal *pertrb, integer *ierror, doublereal *w);

#ifdef __cplusplus
} 
#endif

     
#endif