#!/bin/bash
#instructions how to run it

# 1. mkdir ~/scripts
# 2. cd ~/scripts
# 3. scp user:host:~/sharedVM/scripts/* .
# 4. source ./install_basics.sh
# After running install_basics install guest additions
# (in guest system Vbox window click  Devices->Install Guest Additions ...). If gues system is linux you must have dkms module installed before attempting to install guest additions
# cd /media/V (hit Tab to get to VBox Guest CD)
# sudo ./VBoxLinuxAdditions.run

# Next in the VirtualBox configuration GUI enable shared folders, enable more CPU and allocate more Ram to graphics and enable 3D Graphics extensions

# prior to enabling shared folders one has to install guest additions on guest system (in guest system Vbox window click  Devices->Install Guest Additions ...). If gues system is linux you must have dkms module installed before attempting to install guest additions
sudo apt-get install mc build-essential dkms
sudo useradd -G vboxsf m
sudo apt-get install libvtk5-qt4-dev g++ swig libqwt5-qt4-dev python-qt4 python-qscintilla2 cmake-gui python-qt4-gl python-vtk python-qwt5-qt4 python-dev libxml2-dev build-essential git llvm-3.2-dev

export CC3D_GIT_DIR=~/CC3D_GIT

mkdir $CC3D_GIT_DIR
cd $CC3D_GIT_DIR
git clone https://github.com/CompuCell3D/CompuCell3D.git .


export RR_LLVM_GIT_DIR=~/RR_LLVM_GIT

mkdir $RR_LLVM_GIT_DIR
cd $RR_LLVM_GIT_DIR
git clone https://github.com/AndySomogyi/roadrunner.git .


# adding group for folder sharing on VirtualBox
sudo groupadd vboxsf
# have to use --append switch to make sure that user gets added to new group keeping ald group membership
sudo usermod --append -G vboxsf m

cd ~/

# echo  "m ALL=(ALL) NOPASSWD: /sbin/poweroff, /sbin/reboot, /sbin/shutdown" >> 99poweroff
# sudo chown root:root 99poweroff
# sudo chmod 0440 99poweroff
# sudo mv 99poweroff /etc/sudoers.d/

sudo chmod u+s /sbin/poweroff
sudo chmod u+s /sbin/reboot
sudo chmod u+s /sbin/shutdown

mkdir -p ~/.config/autostart

cp ~/scripts/konsole.desktop ~/.config/autostart
# cp /media/sf_sharedVM/scripts/konsole.desktop ~/.config/autostart

