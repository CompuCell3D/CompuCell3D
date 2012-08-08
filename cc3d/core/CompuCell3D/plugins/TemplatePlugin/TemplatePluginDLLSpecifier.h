#ifndef TEMPLATE_EXPORT_H
#define TEMPLATE_EXPORT_H

    #if defined(_WIN32)
      #ifdef TemplateShared_EXPORTS
          #define TEMPLATE_EXPORT __declspec(dllexport)
          #define TEMPLATE_EXPIMP_TEMPLATE
      #else
          #define TEMPLATE_EXPORT __declspec(dllimport)
          #define TEMPLATE_EXPIMP_TEMPLATE extern
      #endif
    #else
         #define TEMPLATE_EXPORT
         #define TEMPLATE_EXPIMP_TEMPLATE
    #endif

#endif
