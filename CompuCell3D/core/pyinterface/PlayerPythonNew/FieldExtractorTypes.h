#ifndef FIELDEXTRACTORTYPES_H
#define FIELDEXTRACTORTYPES_H


#if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
	typedef long long vtk_obj_addr_int_t ;
#else
	typedef long vtk_obj_addr_int_t;
#endif

#endif
