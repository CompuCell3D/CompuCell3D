#!/bin/bash
export VERSION=RR_LLVM
# echo "THIS IS VERSION ${VERSION} "

export CC3D_GIT_DIR=~/CC3D_GIT
eval CC3D_GIT_DIR=$CC3D_GIT_DIR

export RR_LLVM_GIT_DIR=~/RR_LLVM_GIT
# export CC3D_BINARIES_DIR=/media/sf_sharedVM/binaries/${VERSION}/linux
eval RR_LLVM_GIT_DIR=$RR_LLVM_GIT_DIR
export number_of_cpus=8
export install_path=~/install_projects_${VERSION}
eval install_path=$install_path

cd $CC3D_GIT_DIR
git checkout master
git pull

cd $RR_LLVM_GIT_DIR
git pull
git checkout develop

cd $CC3D_GIT_DIR/build_scripts/linux

time ./build-rr-llvm.sh -s=$RR_LLVM_GIT_DIR -p=$install_path -c=$number_of_cpus

