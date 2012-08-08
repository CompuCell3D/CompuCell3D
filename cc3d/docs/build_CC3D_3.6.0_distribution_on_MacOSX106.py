#!/usr/bin/env python
# <--- the above line asks the 'env' system command to find the python executable, and then executes it.
# everything below is in python

###################################################################
#                                                                 #
# this script builds CompuCell3D 3.6.0 on Mac OS X 10.6.x         #
#                                                                 #
###################################################################
#                                                                 #
# assuming that you already have installed Qt on your system, and #
#   downloaded and built VTK, SIP, and PyQt                       #
#                                                                 #
# this script also assumes that you have downloaded and built     #
#   gcc 4.6.0 (or newer) including its OpenMP support libraries   #
#                                                                 #
###################################################################
#                                                                 #
# 2010-2011 compucell3d.org                                       #
#   edited by Mitja Hmeljak,                                      #
#   based on 2009 script by Benjamin Zaitlen                      #
#                                                                 #
###################################################################


###################################################################
#
#   *** Please note ***
#
# this file will start working *only* after you manually set:
#   the PATH_TO_WORK_DIR string to your OWN path where you want to build CC3D,
#            as for example "/Users/Shared/CC3D360build"
#   the VTK_BIN_AND_BUILD_DIR to your OWN path where you have installed VTK,
#            as for example "/Users/Shared/CC3Ddeps/vtk/VTK_5.4.2_bin_and_build"
#   the GCC_460 string to your OWN path where you have the GCC 4.6.0 distribution:
#            as for example "/Users/Shared/CC3Ddeps/gcc_4.6.0"

PATH_TO_WORK_DIR = ""
VTK_BIN_AND_BUILD_DIR = ""
GCC_460 = ""

#
# after you've set the PATH_TO_WORK_DIR string with your own setting, run this script in Terminal.app
#
###################################################################


import sys, shutil, os
from subprocess import call

print "###########################################"
print "building CompuCell3D 3.6.0 on Mac OS X 10.6"
print "###########################################"


if (PATH_TO_WORK_DIR is "") or (VTK_BIN_AND_BUILD_DIR is "") or (GCC_460 is ""):
    print "This file will start working *only* after you manually..."
    print "... set the PATH_TO_WORK_DIR string to your OWN path where you want to build CC3D,"
    print "... the VTK_BIN_AND_BUILD_DIR to your OWN path where you have installed VTK,"
    print "... and the GCC_460 string to your OWN path where you have the GCC 4.6.0 libraries."
    print ""
    print "Please open this script in a text editor and read code comments for more information."
    print ""
    sys.exit(1)
else:
    print "=====>=====> PATH_TO_WORK_DIR = " + PATH_TO_WORK_DIR
    print "=====>=====> VTK_BIN_AND_BUILD_DIR = " + VTK_BIN_AND_BUILD_DIR
    print "=====>=====> GCC_460 = " + GCC_460
    print "... CC3D build starting ..."



#######################################
#                                     #
# set, clear and create directories   #
#    for building CC3D using CMAKE,   #
#    3rd party supporting libraries,  #
#    distribution binaries            #
#                                     #
#######################################

print "=====>=====> the previous umask was =", os.umask(022)
print "=====>=====> now the umask is =", os.umask(022)

CC3D_ENDUSER_DIR_NAME = 'CC3D_3.6.0_MacOSX106'



###################################
#                                 #
#   BUILD_DIR_FOR_CMAKE           #
#                                 #
###################################

# BUILD_DIR_FOR_CMAKE is the directory into which CMAKE will conduct the entire build:
#
# (this directory is NOT the place where we run the scripts to prepare a CC3D distribution)
#
BUILD_DIR_FOR_CMAKE=os.path.join(PATH_TO_WORK_DIR, 'build_dir_aka_cmake_install_dir')
print "=====>=====> BUILD_DIR_FOR_CMAKE = " + BUILD_DIR_FOR_CMAKE

# first clean, then create the directory -
if  os.path.isdir(BUILD_DIR_FOR_CMAKE):
    print "=====>=====> Build directory (for CMAKE doing CC3D) exists... Removing ", BUILD_DIR_FOR_CMAKE, " and creating new directory."
    shutil.rmtree(BUILD_DIR_FOR_CMAKE)
# create the directory used during CMAKE's build and its own installation procedure:
os.mkdir(BUILD_DIR_FOR_CMAKE)



##########################
#                        #
#    DEP_DIR creation    #
#                        #
##########################
#
# DEP_DIR is the directory into which we place ALL 3rd party dependency libraries, for building
#
DEP_DIR=os.path.join(PATH_TO_WORK_DIR, 'third_party_support_libs_dependencies')
print "=====>=====> DEP_DIR = " + DEP_DIR


# For CC3D 3.6.0, it is necessary to manually include two libraries to support OpenMP: 
OPENMP_LIBS_DEP_DIR=os.path.join(DEP_DIR,'OpenMPlib')
print "=====>=====> OPENMP_LIBS_DEP_DIR = " + OPENMP_LIBS_DEP_DIR

# For CC3D 3.6.0, it is necessary to manually include QScintilla libraries for twedit++: 
QSCINTILLA_LIBS_DEP_DIR=os.path.join(DEP_DIR,'QScintillalib')
print "=====>=====> QSCINTILLA_LIBS_DEP_DIR = " + QSCINTILLA_LIBS_DEP_DIR


# For CC3D 3.6.0, only Qt libraries go in the DEP_DIR directory.
#    (in CC3D 3.4.1 and previous, versions of VTK, Qt, etc. were placed in other subdirectories)
# The Qt binaries copied from within the Qt installation frameworks go in here:
QT_DEPS_DEP_DIR=os.path.join(DEP_DIR,'Deps')
print "=====>=====> QT_DEPS_DEP_DIR = " + QT_DEPS_DEP_DIR

# For CC3D 3.6.0, almost all dependencies are placed temporarily in this directory:
FOR_PLAYER_DEP_DIR=os.path.join(DEP_DIR,'ForPlayerDir')
print "=====>=====> FOR_PLAYER_DEP_DIR = " + FOR_PLAYER_DEP_DIR
#
# Subdirectories have to be created (inside what will become "player/" in the final distribution) into which various libraries will be copied:
#    VTKLibs/      <--- all .dylib library files from the VTK distribution's "lib/vtk-5.4/" directory
#    vtk/     <--- all files from the VTK distribution's "Wrapping/Python/vtk/" for Python AND ALSO all files ending in .so from VTK's build/bin directory
#    PyQt4/   <--- the "PyQt4" directory from the system-wide "site-packages/" PyQt distribution
#    no subdirectory for sip* files from the system-wide "site-packages/" PyQt distribution
# so the above directory for 3rd party support libraries has to contain the following:
DEP_LIBVTK_DIR=os.path.join(FOR_PLAYER_DEP_DIR,'VTKLibs')
print "=====>=====> DEP_LIBVTK_DIR = " + DEP_LIBVTK_DIR
DEP_VTK_DIR=os.path.join(FOR_PLAYER_DEP_DIR,'vtk')
print "=====>=====> DEP_VTK_DIR = " + DEP_VTK_DIR
DEP_PYQT_DIR=os.path.join(FOR_PLAYER_DEP_DIR,'PyQt4')
print "=====>=====> DEP_PYQT_DIR = " + DEP_PYQT_DIR


if  os.path.isdir(DEP_DIR):
    print "=====>=====> 3rd party dependency libraries directory exists... Removing ", DEP_DIR, " and creating new 3rd party directory."
    shutil.rmtree(DEP_DIR)

print "=====>=====> Creating 3rd party dependency libraries directories, DEP_DIR = ", DEP_DIR
os.mkdir(DEP_DIR)
print "=====>=====> Creating 3rd party dependency libraries directories, OPENMP_LIBS_DEP_DIR = ", OPENMP_LIBS_DEP_DIR
os.mkdir(OPENMP_LIBS_DEP_DIR)
print "=====>=====> Creating 3rd party dependency libraries directories, OPENMP_LIBS_DEP_DIR = ", QSCINTILLA_LIBS_DEP_DIR
os.mkdir(QSCINTILLA_LIBS_DEP_DIR)
print "=====>=====> Creating 3rd party dependency libraries directories, QT_DEPS_DEP_DIR = ", QT_DEPS_DEP_DIR
os.mkdir(QT_DEPS_DEP_DIR)
print "=====>=====> Creating 3rd party dependency libraries directories, FOR_PLAYER_DEP_DIR = ", FOR_PLAYER_DEP_DIR
os.mkdir(FOR_PLAYER_DEP_DIR)
print "=====>=====> Creating 3rd party dependency libraries directories, DEP_LIBVTK_DIR = ", DEP_LIBVTK_DIR
os.mkdir(DEP_LIBVTK_DIR)
print "=====>=====> Creating 3rd party dependency libraries directories, DEP_VTK_DIR = ", DEP_VTK_DIR
os.mkdir(DEP_VTK_DIR)
print "=====>=====> Creating 3rd party dependency libraries directories, DEP_PYQT_DIR = ", DEP_PYQT_DIR
os.mkdir(DEP_PYQT_DIR)



###########################
#                         #
#    DEP_DIR file copy    #
#                         #
###########################

print "=====>=====> now getting all 3rd party (previously compiled) dependency libraries into one place <=====<====="

os.chdir(OPENMP_LIBS_DEP_DIR)
call('pwd')
# in here we copy: three library files grabbed from the GCC 4.6.0 distribution we have on our system:
#   (if you download or build GCC 4.6.0 separately, modify the GCC_460 path at the top of this script)
call('rsync -rlPpa '+GCC_460+'/usr/local/lib/libgomp.1.dylib . ',shell=True)
call('rsync -rlPpa '+GCC_460+'/usr/local/lib/libgcc_s.1.dylib . ',shell=True)
call('rsync -rlPpa '+GCC_460+'/usr/local/lib/libstdc++.6.dylib . ',shell=True)

os.chdir(QSCINTILLA_LIBS_DEP_DIR)
call('pwd')
# in here we copy: a library file grabbed from the QScintilla distribution, and symlinks:
call('rsync -rlPpa /Library/Frameworks/libqscintilla2.5.5.0.dylib . ',shell=True)
call('ln -f -s ./libqscintilla2.5.5.0.dylib ./libqscintilla2.dylib',shell=True)
call('ln -f -s ./libqscintilla2.5.5.0.dylib ./libqscintilla2.5.dylib',shell=True)
call('ln -f -s ./libqscintilla2.5.5.0.dylib ./libqscintilla2.5.5.dylib',shell=True)

os.chdir(QT_DEPS_DEP_DIR)
call('pwd')
# in here we copy: four binary files grabbed from WITHIN the frameworks that are part of the main Qt distribution:
#  (this could be done cleaner on Mac OS X - read developer.apple.com about it)
call('rsync -rlPpa /Library/Frameworks/QtGui.framework/Versions/Current/QtGui . ',shell=True)
call('rsync -rlPpa /Library/Frameworks/QtCore.framework/Versions/Current/QtCore . ',shell=True)
call('rsync -rlPpa /Library/Frameworks/QtNetwork.framework/Versions/Current/QtNetwork . ',shell=True)
call('rsync -rlPpa /Library/Frameworks/QtOpenGL.framework/Versions/Current/QtOpenGL . ',shell=True)
call('rsync -rlPpa /Library/Frameworks/QtXml.framework/Versions/Current/QtXml . ',shell=True)
# in here we copy: ALSO a .nib file (although it might be cleaner to provide the entire framework)
call('rsync -rlPpa /Library/Frameworks/QtGui.framework/Versions/Current/Resources/qt_menu.nib . ',shell=True)

# 2011: include QScintilla libraries in the install:
call('rsync -rlPpa /Library/Frameworks/libqscintilla2.5.5.0.dylib . ',shell=True)
call('ln -f -s ./libqscintilla2.5.5.0.dylib ./libqscintilla2.dylib',shell=True)
call('ln -f -s ./libqscintilla2.5.5.0.dylib ./libqscintilla2.5.dylib',shell=True)
call('ln -f -s ./libqscintilla2.5.5.0.dylib ./libqscintilla2.5.5.dylib',shell=True)

os.chdir(DEP_LIBVTK_DIR)
call('pwd')
# in here we copy: all .dylib files (including all the symbolic links to them in the same dir)
call('rsync -rlPpa '+VTK_BIN_AND_BUILD_DIR+'/lib/vtk-5.4/*dylib . ',shell=True)

#
# a useful note from rsync's man page:
#
#  A trailing slash on the source changes this behavior to avoid creating an additional directory level at the destination.
#  You can think of a trailing / on a source as meaning "copy the contents of this directory" as opposed to "copy the directory by name",
#  but in both cases the attributes  of the containing directory are transferred to the containing directory on the destination.
#  In other words, each of the following commands copies the files in the same way,
#  including their setting of the attributes of /dest/foo:
# 
#  rsync -av /src/foo /dest
#  rsync -av /src/foo/ /dest/foo
#
os.chdir(DEP_VTK_DIR)
call('pwd')
# in here we copy: all files from VTK's Wrapping directory for Python:
call('rsync -rlPpa '+VTK_BIN_AND_BUILD_DIR+'/Wrapping/Python/vtk/ . ',shell=True)
# in here we ALSO copy: all files ending in .so from VTK's build/bin directory:
call('rsync -rlPpa '+VTK_BIN_AND_BUILD_DIR+'/bin/*.so . ',shell=True)

os.chdir(DEP_PYQT_DIR)
call('pwd')
# in here we copy: the PyQt4 directory as created by the original (system-wide) PyQt install procedure)
call('rsync -rlPpa /Library/Python/2.6/site-packages/PyQt4/ . ',shell=True)

os.chdir(FOR_PLAYER_DEP_DIR)
call('pwd')
# in here we copy: some files from Python's directory into FOR_PLAYER_DEP_DIR:
call('rsync -rlPpa /Library/Python/2.6/site-packages/sip.so . ',shell=True)
call('rsync -rlPpa /Library/Python/2.6/site-packages/sipconfig.py . ',shell=True)
call('rsync -rlPpa /Library/Python/2.6/site-packages/sipdistutils.py . ',shell=True)



#################
#               #
#    CUR_DIR    #
#               #
#################

# other references:
CUR_DIR=os.path.join(os.getcwd(), PATH_TO_WORK_DIR)
print "=====>=====> CUR_DIR = " + CUR_DIR



#################
#               #
#    BIN_DIR    #
#               #
#################

# here will go the resulting complete distribution archive:
BIN_DIR=os.path.join(CUR_DIR, CC3D_ENDUSER_DIR_NAME)
print "=====>=====> BIN_DIR = " + BIN_DIR

# first clean, then create -
if  os.path.isdir(BIN_DIR):
    print "=====>=====> Binary distribution directory exists... Removing", BIN_DIR, " and creating new binary directory."
    shutil.rmtree(BIN_DIR)
# create a directory where the complete cc3d binary+supporting libraries will be placed for distribution
os.mkdir(BIN_DIR)



#################
#               #
#    SRC_DIR    #
#               #
#################

# this MUST be a directory called "3.6.0" because we grab the "3.6.0" version from the svn repository, it's hardcoded in this script:
SRC_DIR=os.path.join(PATH_TO_WORK_DIR, '3.6.0')
print "=====>=====> SRC_DIR = " + SRC_DIR

# assure that the source directory is cleared, we're going to build cc3d from it.
# the following has to be removed but NOT recreated right away, because it'll be downloaded below using svn:
if  os.path.isdir(SRC_DIR):
    print "=====>=====> Source directory exists... Removing", SRC_DIR
    shutil.rmtree(SRC_DIR)



#######################################
#                                     #
#            build CC3D               #
#                                     #
#######################################

print "=====>=====> now finally BUILD CompuCell 3D <=====<====="

print "=====>=====> obtaining the latest CC3D source code from the online svn repository:"
# cd to the directory we're using to hold it all, and grab the latest source code from svn repository:
os.chdir(CUR_DIR)
call('pwd')
call('svn export http://code.compucell3d.org/svn/cc3d/branch/3.6.0',shell=True)


print "=====>=====> building CC3D:"
# cd to the directory holding all the cc3d source code, and compile it:
os.chdir(SRC_DIR)
call('pwd')

print "=====>=====> prepare all make files using cmake with a few command-line settings/options:"

# this setting would build "fat" 32bit+64bit code, but the OpenMP libraries we use are 64 bit only anyway:
# call('cmake -DCMAKE_INSTALL_PREFIX:PATH='+BUILD_DIR_FOR_CMAKE+ " -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.6 -DVTK_DIR:PATH="+VTK_BIN_AND_BUILD_DIR+"/lib/vtk-5.4 -DCMAKE_OSX_ARCHITECTURES:STRING=i386\;x86_64" ,shell=True)

# this setting would build CC3D 3.6.0 (64bit code only) using gcc 4.2.0 as included with the Xcode distribution :
#call('cmake -DCMAKE_INSTALL_PREFIX:PATH='+BUILD_DIR_FOR_CMAKE+ " -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=10.6 -DVTK_DIR:PATH="+VTK_BIN_AND_BUILD_DIR+"/lib/vtk-5.4 -DCMAKE_OSX_ARCHITECTURES:STRING=x86_64" ,shell=True)

# this setting builds CC3D 3.6.0 (64bit code only) using gcc 4.6.0 from its own separate compiler distribution directory:
call('cmake -DCMAKE_C_COMPILER='+GCC_460+'/usr/local/bin/gcc -DCMAKE_CXX_COMPILER='+GCC_460+'/usr/local/bin/g++ -DCMAKE_INSTALL_PREFIX:PATH='+BUILD_DIR_FOR_CMAKE+ " -DVTK_DIR:PATH="+VTK_BIN_AND_BUILD_DIR+"/lib/vtk-5.4" , shell=True)


print "=====>=====> make CC3D:"
call('make install -j8',shell=True)
os.chdir(CUR_DIR)
call('pwd')



###################################################################
#                                                                 #
#            copy files into CC3D distribution directory          #
#                                                                 #
###################################################################



print "=====>=====> copy all the 3rd party supporting libraries into one directory for distribution"
# now copy all the supporting libraries that are necessary for the cc3d player into the directory for distribution:

call('rsync -rlPp '+OPENMP_LIBS_DEP_DIR+'/  '+BIN_DIR+'/lib',shell=True)

call('rsync -rlPpa '+QSCINTILLA_LIBS_DEP_DIR+'/  '+BIN_DIR+'/lib',shell=True)

call('rsync -rlPp '+FOR_PLAYER_DEP_DIR+'/  '+BIN_DIR+'/player',shell=True)

call('rsync -rlPp '+QT_DEPS_DEP_DIR+' ' + BIN_DIR,shell=True)



print "=====>=====> copy the compiled CC3D from the install directory used by cmake, and various shell files"

# new 2010 style:
call('chmod ugo+rx compucell3d.command compucell3d.sh runScript.command twedit++.command twedit++.sh ', shell=True)
call('rsync -rPlp '+BUILD_DIR_FOR_CMAKE+'/ compucell3d.command runScript.command twedit++.command '+BIN_DIR,shell=True)

# remove twedit++.sh from the distribution directory and replace it with our newer version, this could be fixed in CMake files <---- TODO
if  os.path.isfile(BIN_DIR+'/twedit++.sh'):
    os.remove(BIN_DIR+'/twedit++.sh')
call('rsync -rPlp  twedit++.sh  '+BIN_DIR,shell=True)

# remove compucell3d.sh from the distribution directory and replace it with our newer version, this could be fixed in CMake files <---- TODO
if  os.path.isfile(BIN_DIR+'/compucell3d.sh'):
    os.remove(BIN_DIR+'/compucell3d.sh')
call('rsync -rPlp  compucell3d.sh  '+BIN_DIR,shell=True)

# rename old stuff in the distribution directory:
if  os.path.isfile(BIN_DIR+'/runScript.sh'):
    os.rename(BIN_DIR+'/runScript.sh', BIN_DIR+'/runScript_older.sh')


# remove any precompiled python code, and any Mac OS X Finder-specific files:
call("find . -type f -iname \'*.pyc\' -exec rm -f {} \;", shell=True)
call("find . -type f -iname \'.DS_Store\' -exec rm -f {} \;", shell=True)


# create the .zip archive for distribution:
print "=====>=====> zipping the CC3D directory using \'ditto\':"
CC3D_ARCHIVE = 'CC3D_3.6.0_MacOSX106.zip'
os.chdir(CUR_DIR)
call("rm -f " + CC3D_ARCHIVE, shell=True)
call('ditto -c -k --keepParent -rsrcFork ' + CC3D_ENDUSER_DIR_NAME + ' ' + CC3D_ARCHIVE, shell=True)
call("ls -lad " + CC3D_ARCHIVE, shell=True)


print "=====>=====>=================<=====<====="
print "=====>=====> is it over now? <=====<====="
print "=====>=====>=================<=====<====="
print "===> CC3D 3.6.0 distribution ready!! <==="
print "=====>=====>=================<=====<====="
