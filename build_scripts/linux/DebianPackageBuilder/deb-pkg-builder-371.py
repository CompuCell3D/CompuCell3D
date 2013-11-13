import os,sys
import re
import subprocess
from subprocess import Popen,PIPE
import shutil
# example command:
# python ./deb-pkg-builder-371.py -d /usr/share/compucell3d -i ~/cc3d_deb_pkg -v 3.7.0




def replaceTemplate(tplFileName,replaceDict={}):
    outFileName,ext=os.path.splitext(tplFileName)    
  
    inFile=open(tplFileName,'r')
    outFile=open(outFileName,'w')
    for line in inFile.readlines():
        line=line.rstrip()
        newLineTmp=str(line)
        for tplName,replaceVal in replaceDict.iteritems():
            newLineTmp=re.sub(tplName,replaceVal,newLineTmp)
            newLineTmp=newLineTmp.rstrip()
            #print 'newLineTmp=',newLineTmp
        
        print>>outFile,newLineTmp    
        
    inFile.close()
    outFile.close()
    os.remove(tplFileName)
    

def modifyRunScripts(fileName,location_on_target_system):
    targetSystemLocation=os.path.abspath('/'+location_on_target_system)
    
    tmpFileName=os.path.abspath(fileName+'.tmp')
    
    tmpFile=open(tmpFileName,'w')
    inFile=open(fileName,'r')
    
    for line in inFile.readlines():
        line=line.rstrip()
        prefix_match=re.match('[\s]*export[\s]*PREFIX_CC3D',line)
        if prefix_match:
            #print 'FOUND PREFIX LINE=',line
            print >>tmpFile,'export PREFIX_CC3D='+targetSystemLocation
        else:
            print>>tmpFile,line    
        
    inFile.close()
    tmpFile.close()
    shutil.move(tmpFileName,fileName)
    subprocess.call(['chmod','0755',fileName])
    
    


# -------------- parsing command line
from optparse import OptionParser
parser = OptionParser()
parser.add_option("-d", "--directory", dest="cc3d_install_dir",action="store", type="string",help="CC3D installation directory")
parser.add_option("-i", "--installer-dir", dest="installer_dir",action="store", default='', type="string",help="full installer directory name")
parser.add_option("-v", "--version", dest="version",action="store", type="string",help="CC3D version", default='3.7.0')


(options, args) = parser.parse_args()

# -------------- end of parsing command line

#extracting lunux architecture
p1=Popen(['uname','-m'],stdout=PIPE)
linux_architecture = p1.communicate()[0].rstrip()

p1=Popen(['dpkg','--print-architecture'],stdout=PIPE)
dpkg_architecture = p1.communicate()[0].rstrip()


#extracting ubuntu release
p1=Popen(['lsb_release','--release'],stdout=PIPE)
p2=Popen(['cut','-f2'],stdin=p1.stdout, stdout=PIPE)
release = p2.communicate()[0].rstrip()


location_on_target_system ='usr/lib/compucell3d' # this is where cc3d will be installed on a target system - notice first forward slash is missing

version=options.version

cc3d_install_dir=os.path.abspath(options.cc3d_install_dir)

cwd=os.getcwd()
package_path=os.path.join(cwd,'DebianPackageTemplate')


if options.installer_dir=='':
    print 'Please specify FULL installer directory name'    
    sys.exit()

#revision number 
from datetime import date
today=date.today()
revisionNumber=str(today.year)+str(today.month).zfill(2)+str(today.day).zfill(2)
version=options.version
    
installer_core_name='compucell3d-'+version+'-'+release+'-'+linux_architecture+'-'+revisionNumber    
installer_ext='.deb'

print 'installer_core_name=',installer_core_name

installer_dir=os.path.abspath(options.installer_dir)
installer_full_name=os.path.join(installer_dir,installer_core_name)+installer_ext

installer_working_dir = installer_dir+'_tmp'


installer_name_tmp = os.path.join(installer_working_dir, installer_core_name )+installer_ext

installer_working_dir_pkg_placeholder=os.path.join(installer_working_dir,installer_core_name)

print 'installer_working_dir =',installer_working_dir 
print 'installer_working_dir_pkg_placeholder=',installer_working_dir_pkg_placeholder


if os.path.isdir(installer_working_dir):
    shutil.rmtree(installer_working_dir)
os.makedirs(installer_working_dir)

subprocess.call(['cp','-r', package_path,installer_working_dir_pkg_placeholder])

#subprocess.call('cp -r '+package_path+' '+installer_working_dir,shell=True)
#copy deb package template files into workign dir
#shutil.copytree(package_path,installer_working_dir_pkg_placeholder)



#copy cc3d installation directory to Debian package workign directory
cc3d_debian_pkg_dest_dir = os.path.join(installer_working_dir_pkg_placeholder,location_on_target_system)
print 'cc3d_debian_pkg_dest_dir=',cc3d_debian_pkg_dest_dir
print 'cc3d_install_dir=',cc3d_install_dir
#shutil.copytree(cc3d_install_dir,cc3d_debian_pkg_dest_dir)

os.makedirs(cc3d_debian_pkg_dest_dir)
#subprocess.call(['cp','-r', cc3d_install_dir+'/*',cc3d_debian_pkg_dest_dir])
#subprocess.call('cp -r  ~/install_projects/3.7.0/*  /home/bioc/cc3d_deb_pkg_tmp',shell=True)
cc3d_copy_command='cp -r '+cc3d_install_dir+'/* '+cc3d_debian_pkg_dest_dir
#print 'command=',command
subprocess.call(cc3d_copy_command,shell=True)
#subprocess.call('cp -r'+cc3d_install_dir+'/* '+cc3d_debian_pkg_dest_dir ,shell=True)

#changing permissions
#subprocess.call('chmod -R 0622 '+cc3d_debian_pkg_dest_dir+'/include',shell=True)

#strip libraries
#strip_command='strip --strip-unneeded '+cc3d_debian_pkg_dest_dir+'/lib/*.so'
subprocess.call('strip --strip-unneeded '+cc3d_debian_pkg_dest_dir+'/lib/*.so',shell=True)
subprocess.call('strip --strip-unneeded '+cc3d_debian_pkg_dest_dir+'/lib/CompuCell3DPlugins/*.so',shell=True)
subprocess.call('strip --strip-unneeded '+cc3d_debian_pkg_dest_dir+'/lib/CompuCell3DSteppables/*.so',shell=True)
subprocess.call('strip --strip-unneeded '+cc3d_debian_pkg_dest_dir+'/lib/python/*.so',shell=True)
subprocess.call('strip --strip-unneeded '+cc3d_debian_pkg_dest_dir+'/bin/*',shell=True)

# check the size of the debian package
# we can use shell=True in subprocess Popen instead
p1=Popen(['du','-s',installer_working_dir],stdout=PIPE)
p2=Popen(['awk','{ print $1; }' ],stdin=p1.stdout, stdout=PIPE)
out, err = p2.communicate()
package_size=out
print 'size of debian package=',out

#replacing tpl files with target files
replaceTemplate(os.path.join(installer_working_dir_pkg_placeholder,'usr/share/applications/compucell3d.desktop.tpl'),{'<VERSION>':version})
replaceTemplate(os.path.join(installer_working_dir_pkg_placeholder,'usr/share/applications/twedit++.desktop.tpl'),{'<VERSION>':version})
replaceTemplate(os.path.join(installer_working_dir_pkg_placeholder,'usr/share/applications/celldraw.desktop.tpl'),{'<VERSION>':version})
replaceTemplate(os.path.join(installer_working_dir_pkg_placeholder,'DEBIAN/control.tpl'),{'<VERSION>':version,'<INSTALLED_SIZE>':str(package_size),'<ARCHITECTURE>':dpkg_architecture,'<EXTRA_DEPS>':',llvm-3.2'})    


#modifying run scripts
modifyRunScripts(os.path.join(installer_working_dir_pkg_placeholder,location_on_target_system+'/compucell3d.sh'),location_on_target_system)
modifyRunScripts(os.path.join(installer_working_dir_pkg_placeholder,location_on_target_system+'/celldraw.sh'),location_on_target_system)
modifyRunScripts(os.path.join(installer_working_dir_pkg_placeholder,location_on_target_system+'/runScript.sh'),location_on_target_system)
modifyRunScripts(os.path.join(installer_working_dir_pkg_placeholder,location_on_target_system+'/twedit++.sh'),location_on_target_system)

print 'WILL BUILD DEB PACKAGE NOW'
subprocess.call(['fakeroot','dpkg-deb','--build',installer_working_dir_pkg_placeholder])


print 'installer_name_tmp=',installer_name_tmp


#in case it does not exist, create destination directory for debian files 
if not os.path.isdir(installer_dir):
    os.makedirs(installer_dir)
    
# copy installer from tmp dir to destination dir
shutil.copyfile(installer_name_tmp,os.path.join(installer_dir , installer_core_name)+installer_ext)

#removing temporary directory
#shutil.rmtree(installer_working_dir)
    
print 'DONE'




