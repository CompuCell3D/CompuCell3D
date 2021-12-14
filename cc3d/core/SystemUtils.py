# todo: add system utils support for additional software/extensions
"This module contains platform specific initializations"

from cc3d import cc3d_scripts_path


def setSwigPaths():
    from os import environ
    import string
    import sys
    platform=sys.platform
    if platform.startswith('win'):
        
        swig_lib_install_path=environ["SWIG_LIB_INSTALL_DIR"]
        appended=sys.path.count(swig_lib_install_path)
        if not appended:
            sys.path.append(swig_lib_install_path)    
        # sys.path.append(environ["SWIG_LIB_INSTALL_DIR"])
        
        soslib_path=environ["SOSLIB_PATH"]
        appended=sys.path.count(soslib_path)
        if not appended:
            sys.path.append(soslib_path)    
        # sys.path.append(environ["SOSLIB_PATH"])
    else:
    
        swig_path_list=string.split(environ["SWIG_LIB_INSTALL_DIR"])
        for swig_path in swig_path_list:            
            appended=sys.path.count(swig_path)
            if not appended:
                sys.path.append(swig_path)    
        
            # sys.path.append(swig_path)

        soslib_path=environ["SOSLIB_PATH"]
        appended=sys.path.count(soslib_path)
        if not appended:
            sys.path.append(soslib_path)    
        # sys.path.append(environ["SOSLIB_PATH"])
                
def getCC3DPlayerRunScriptPath():        
    '''returns full path name to player run script
    '''
    import sys
    import os

    if sys.platform.startswith('win'):
        cc3dPath = os.path.join(cc3d_scripts_path, 'compucell3d.bat')
    elif sys.platform.startswith('darwin'):
        cc3dPath = os.path.join(cc3d_scripts_path, 'compucell3d.command')
    else: # linux/unix
        cc3dPath = os.path.join(cc3d_scripts_path, 'compucell3d.sh')
        
    cc3dPath = os.path.abspath(cc3dPath)
    return cc3dPath

    
def getCC3DRunScriptPath():
    import sys
    import os

    if sys.platform.startswith('win'):
        cc3dPath = os.path.join(cc3d_scripts_path, 'runScript.bat')
    elif sys.platform.startswith('darwin'):
        cc3dPath = os.path.join(cc3d_scripts_path, 'runScript.command')
    else:  # linux/unix
        cc3dPath = os.path.join(cc3d_scripts_path, 'runScript.sh')

    cc3dPath = os.path.abspath(cc3dPath)
    return cc3dPath

def getCommandLineArgList():
    '''returns command line options for parameter scan WITHOUT actual run script. run script has to be fetched independently using getCC3DPlayerRunscriptPath or getCC3DPlayerRunscriptPath in SystemUtils
    '''
    import sys
    reminderArgs=sys.argv[1:] 
    return reminderArgs


def initializeSystemResources():
    platform=''
    RTLD_GLOBAL=0x0
    RTLD_NOW=0x0


    try:
        import sys
        platform=sys.platform
    except ImportError:
        print ("Could not find sys module needed for setting upe system dependent resources. "
               "Check your Python installation. This is a basic module")
    else:
        platform=sys.platform
    
    print("Platform:",platform)


    if platform.startswith('Linux') or platform.startswith('linux') or platform.startswith('linux2'):
        try:
            import dl
        except ImportError:
            print("Did not find dl module, will try manual dl initialization...")
            RTLD_GLOBAL=0x001000
            RTLD_NOW=0x00002
        else:
            RTLD_GLOBAL=dl.RTLD_GLOBAL
            RTLD_NOW=dl.RTLD_NOW
            
        sys.setdlopenflags(RTLD_GLOBAL | RTLD_NOW)
        
    elif platform.startswith('Darwin') or platform.startswith('darwin'):
        try:
            import dl
        except ImportError:
            print("Did not find dl module, will try manual dl initialization...")
            RTLD_GLOBAL=0x001000
            RTLD_NOW=0x00002
        else:
            RTLD_GLOBAL=dl.RTLD_GLOBAL
            RTLD_NOW=dl.RTLD_NOW
            
        sys.setdlopenflags(RTLD_GLOBAL | RTLD_NOW)
    elif platform.startswith('win'):
        print("MICROSOFT WINDOWS PLATFORM. Enjoy the bumpy ride ...")
    else:
        print("This platform is not supported for CompuCell Python Scripting")
        sys.exit()