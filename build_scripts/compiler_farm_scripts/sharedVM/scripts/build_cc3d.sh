#!/bin/bash
export VERSION=3.7.0
echo "THIS IS VERSION ${VERSION} "


export CC3D_GIT_DIR=~/CC3D_GIT
export CC3D_BINARIES_DIR=/media/sf_sharedVM/binaries/${VERSION}/linux
eval CC3D_GIT_DIR=$CC3D_GIT_DIR
export number_of_cpus=8
export install_path=~/install_projects_${VERSION}/${VERSION}
eval install_path=$install_path

# get binaries repository
# mkdir $CC3D_BINARIES_DIR
# cd $CC3D_BINARIES_DIR
# svn checkout http://www.compucell3d.org/BinDoc/cc3d_binaries/binaries/3.7.0/linux .
# svn update


cd $CC3D_GIT_DIR

git pull
cd $CC3D_GIT_DIR/build_scripts/linux

time ./build-debian-cc3d.sh -s=$CC3D_GIT_DIR -p=$install_path -c=$number_of_cpus

cd $CC3D_GIT_DIR/build_scripts/linux/DebianPackageBuilder

# remove old deb packages
rm -rf ${install_path}_deb/*

python ./deb_pkg_builder.py -d $install_path -i ${install_path}_deb


mkdir -p ${CC3D_BINARIES_DIR}
#removing old debian packages
rm -rf ${CC3D_BINARIES_DIR}/* 

cd ${install_path}_deb

cp *.deb $CC3D_BINARIES_DIR

#this command can be executed password free if you do e.g. 
# echo  "%m ALL=(ALL) NOPASSWD: /sbin/poweroff, /sbin/reboot, /sbin/shutdown" >> poweroff.sudo
# sudo chmod 0440 poweroff.sudo
# sudo mv poweroff.sudo /etc/sudoers.d/

poweroff
# cd $CC3D_BINARIES_DIR
# svn add *
# svn commit -m "build deb package "
