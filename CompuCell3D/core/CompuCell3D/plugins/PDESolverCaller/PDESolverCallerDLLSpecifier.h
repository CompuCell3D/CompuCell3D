#ifndef PDESOLVERCALLER_EXPORT_H
#define PDESOLVERCALLER_EXPORT_H

    #if defined(_WIN32)
      #ifdef PDESolverCallerShared_EXPORTS
          #define PDESOLVERCALLER_EXPORT __declspec(dllexport)
          #define PDESOLVERCALLER_EXPIMP_TEMPLATE
      #else
          #define PDESOLVERCALLER_EXPORT __declspec(dllimport)
          #define PDESOLVERCALLER_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define PDESOLVERCALLER_EXPORT
         #define PDESOLVERCALLER_EXPIMP_TEMPLATE
    #endif

#endif
