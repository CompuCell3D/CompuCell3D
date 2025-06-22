

#ifndef FIELDMANAGER_EXPORT_H
#define FIELDMANAGER_EXPORT_H

    #if defined(_WIN32)

      #ifdef FieldManagerShared_EXPORTS

          #define FIELDMANAGER_EXPORT __declspec(dllexport)

          #define FIELDMANAGER_EXPIMP_TEMPLATE

      #else

          #define FIELDMANAGER_EXPORT __declspec(dllimport)

          #define FIELDMANAGER_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define FIELDMANAGER_EXPORT

         #define FIELDMANAGER_EXPIMP_TEMPLATE

    #endif

#endif

