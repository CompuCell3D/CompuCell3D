#ifndef EXPRESSIONEVALUATOR_EXPORT_H
#define EXPRESSIONEVALUATOR_EXPORT_H

    #if defined(_WIN32)
      #ifdef ExpressionEvaluatorShared_EXPORTS
          #define EXPRESSIONEVALUATOR_EXPORT __declspec(dllexport)
          #define EXPRESSIONEVALUATOR_EXPIMP_TEMPLATE
      #else
          #define EXPRESSIONEVALUATOR_EXPORT __declspec(dllimport)
          #define EXPRESSIONEVALUATOR_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define EXPRESSIONEVALUATOR_EXPORT
         #define EXPRESSIONEVALUATOR_EXPIMP_TEMPLATE
    #endif

#endif
