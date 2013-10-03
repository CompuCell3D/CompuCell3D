#building dependencies for CC3 on OSX 10.6.8
# dependencies are built using standard SDK with gcc 4.2

#Qt
cd qt-everywhere-opensource-src-4.8.5
./configure -prefix /Users/m/installs/qt-4.8.5
make
make install

# sip
cd sip-4.15.2
python2.6 configure.py --deployment-target=10.6 -n -d ${HOME}/installs/Python/2.6/site-packages -b ${HOME}/installs/bin -e ${HOME}/installs/include -v ${HOME}/installs/share/sip --arch=x86_64 --arch=i386
make
make install

#PyQt4
cd Downloads/PyQt-mac-gpl-4.10.3
python2.6 configure.py -d ${HOME}/installs/Python/2.6/site-packages -b ${HOME}/installs/bin -e ${HOME}/installs/include  --use-arch=x86_64  --qmake=/Users/m/installs/qt-4.8.5/bin/qmake




#QScintilla
# add these two lines to qscintilla.pro
#!win32:VERSION = 9.0.2
#CONFIG+=x86 x86_64

cd Downloads/QScintilla-gpl-2.7.2/Qt4Qt5/
qmake -spec macx-g++ PREFIX=~/installs qscintilla.pro
make 
make install


cd Downloads/QScintilla-gpl-2.7.2/Python/
python2.6 configure.py --destdir=${HOME}/installs/Python/2.6/site-packages/PyQt4 --pyqt-sipdir=${HOME}/installs/share/sip/PyQt4 --sip-incdir=/Users/m/installs/include/
make install

#PyQwt 5.2.0
cd Downloads/PyQwt-5.2.0/configure
python configure.py -Q ../qwt-5.2 --module-install-path=/Users/m/installs/Python/2.6/site-package/PyQt4/Qwt5
make
make install

#vtk
#regular compilation via cmake gui
# to make fat binaries otherwise x86_64 is sufficient
CMAKE_OSX_ARCHITECTURES=i386;x86_64
# make sure python headers, library and executable are for the same version of Python

# to install python bindings
mkdir -p /Users/m/installs/vtk-5.10/lib/python2.6/site-packages/
# add /Users/m/installs/vtk-5.10/lib/python2.6/site-packages/ tp PYTHONPATH evn var in .bash_profile
# and then 
make install

# gcc
# to compile cc3d on mac you need to install relatively new gcc. Apple ships archaic verision which is buggy
brew tap homebrew/versions
brew install gcc48

#notice gcc47 does not compile on osx 10.6.8 
# gcc 4.7 from hpc.sourceforge.net is buggy
#welcome to osx reality ...

