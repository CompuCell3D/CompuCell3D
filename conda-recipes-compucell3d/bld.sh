echo "beginning file copy"
echo ${SRC_DIR}
echo ${PREFIX}

cd conda-recipes-compucell3d
# one of those commands will fail on OX and one on linux but this is OK we need to have only one work
# on the right OS
cp compucell3d.sh ${PREFIX}/compucell3d.sh
cp compucell3d.command ${PREFIX}/compucell3d.command

