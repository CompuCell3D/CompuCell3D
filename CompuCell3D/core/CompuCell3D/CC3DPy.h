#ifndef CC3DPY_H
#define CC3DPY_H

#if defined(_MSC_VER)
#   if defined(_DEBUG)
#       define CC3D_DEBUG_MARKER
#       undef _DEBUG
#   endif
#endif

#include <Python.h>

#if defined(CC3D_DEBUG_MARKER)
#   define _DEBUG
#   undef CC3D_DEBUG_MARKER
#endif

#endif