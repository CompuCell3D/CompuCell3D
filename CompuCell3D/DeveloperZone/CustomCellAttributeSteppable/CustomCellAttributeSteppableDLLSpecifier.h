

#ifndef CUSTOMCELLATTRIBUTESTEPPABLE_EXPORT_H

#define CUSTOMCELLATTRIBUTESTEPPABLE_EXPORT_H



    #if defined(_WIN32)

      #ifdef CustomCellAttributeSteppableShared_EXPORTS

          #define CUSTOMCELLATTRIBUTESTEPPABLE_EXPORT __declspec(dllexport)

          #define CUSTOMCELLATTRIBUTESTEPPABLE_EXPIMP_TEMPLATE

      #else

          #define CUSTOMCELLATTRIBUTESTEPPABLE_EXPORT __declspec(dllimport)

          #define CUSTOMCELLATTRIBUTESTEPPABLE_EXPIMP_TEMPLATE extern

      #endif

    #else

         #define CUSTOMCELLATTRIBUTESTEPPABLE_EXPORT

         #define CUSTOMCELLATTRIBUTESTEPPABLE_EXPIMP_TEMPLATE

    #endif



#endif

