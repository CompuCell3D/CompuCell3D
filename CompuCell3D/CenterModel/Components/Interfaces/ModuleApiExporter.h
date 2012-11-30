
#ifndef MODULEAPIEXPORTER_H
#define MODULEAPIEXPORTER_H

#define MODULE_EXTERNAL_API(dllSpecifierName,baseClassName, className) extern "C"{ \
    dllSpecifierName baseClassName * createModule() {\
        return new className();\
    }\
    dllSpecifierName std::string  getModuleName() {\
        return moduleName;\
    }\
    dllSpecifierName std::string  getModuleType() {\
        return moduleType;\
    }\
    dllSpecifierName std::string  getModuleAuthor() {\
        return author;\
    }\
    dllSpecifierName int  getVersionMajor() {\
        return versionMajor;\
    }\
    dllSpecifierName int  getVersionMinor() {\
        return versionMinor;\
    }\
    dllSpecifierName int  getVersionSubMinor() {\
        return versionSubMinor;\
    }\
};   




#endif
