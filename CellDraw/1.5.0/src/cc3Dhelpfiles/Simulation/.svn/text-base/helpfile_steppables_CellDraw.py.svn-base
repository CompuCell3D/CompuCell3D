from PySteppables import *

import CompuCell

import Configuration # to obtain CC3D preferences/settings, specifically "OutputLocation"

import UI.UserInterface

import sys
import os      # for path and split functions





class HelpfileCellDrawSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=100):
        SteppableBasePy.__init__(self,_simulator,_frequency)

        self.theOutputDir = str(Configuration.getSetting("OutputLocation"))
        self.theHelperOutputDirectoryForCellDraw = os.path.join(self.theOutputDir, "cellDrawHelpFiles")
        print " ||||||||||||| in CompuCell3D steppable: 'self.theOutputDir' is [" + self.theOutputDir + "]"
        

    def start(self):
        print " ||||||||||||| in CompuCell3D steppable: 'HelpfileCellDrawSteppable' starting"
        self.Potts = self.simulator.getPotts()
        # print "self.Potts =", self.Potts
        self.cellFieldG = self.Potts.getCellFieldG()
        # print "self.cellFieldG =", self.cellFieldG
        self.dim = self.cellFieldG.getDim()
        # print "self.dim =", self.dim
        self.fieldXsize = self.dim.x
        self.fieldYsize = self.dim.y
        self.fieldZsize = self.dim.z
        print "self.fieldXsize =",self.fieldXsize,"self.fieldYsize =",self.fieldYsize,"self.fieldZsize =",self.fieldZsize
        self.savedPIF = False
        print " ||||||||||||| in CompuCell3D steppable: 'HelpfileCellDrawSteppable' done"

    def step(self,mcs):

        # print " ||||||||||||| in CompuCell3D steppable:  'HelpfileCellDrawSteppable' step(mcs = %s)"%(mcs)

        if (self.savedPIF is False) and (mcs > 100):

            # open output file, and make sure that it's writable:
            # lCurrentDirectory = os.getcwd()
            # print lCurrentDirectory
            # print os.umask(022)
            # print os.umask(022)

            lFileName = os.path.join(self.theHelperOutputDirectoryForCellDraw, "helpfileoutputfrompotts.piff")
            try:
                lFile = open(lFileName, 'w')
            except:
                print " ||||||||||||| in CompuCell3D steppable:  'HelpfileCellDrawSteppable' step(mcs = %s) can not write file %s ||||||||||||| "%(mcs, lFileName)
                return False

            # erase any 'flag' file:
            lFlagFileName = "flagfile.text"
            if os.path.isfile(lFlagFileName) :
                os.remove(lFlagFileName)

            # this would be Qt's way to open files, but we use plain Python calls here:
#             lOutputFileName=os.path.join(os.getcwd(),"helpfileoutputfrompotts.piff")
#             lFile = QtCore.QFile(lOutputFileName)
#             lOnlyThePathName,lOnlyTheFileName = os.path.split(str(lOutputFileName))
#             if not lFile.open( QtCore.QFile.WriteOnly | QtCore.QFile.Text):
#                 return False
#             # open a QTextStream, i.e. an "interface for reading and writing text":
#             lOutputStream = QtCore.QTextStream(lFile)

            # saving PIFF file cell IDs starting from 0 onwards is NOT used,
            #    because we use CC3D's own cell IDs:
            lPixelID = 0
       
            for i in xrange(self.fieldXsize):
                for j in xrange(self.fieldYsize):
                    for k in xrange(self.fieldZsize):
                        lCell = self.cellFieldG.get( CompuCell.Point3D(i,j,k) )
                        # avoid printing out Wall cells:
                        if lCell is not None:
                            if lCell.type is not 1:
                            # output one line per pixel to a second helpfile PIFF to be re-read by PIFF Generator:
                                # print "i=",i,"","j=",j,"","k=",k,"","lCell=",lCell
                                # NOTE NOTE NOTE: this is NOT the cell type name (where is that stored?)
                                #    but the cell type number, and PIFF Generator will have to convert it back to the cell name:
                                lTheCellTypeName = lCell.type
                                lTheCellID = lCell.id
                                xmin = i
                                xmax = i
                                ymin = j
                                ymax = j
                                zmin = k
                                zmax = k
                                lOutputLine = str(lTheCellID) + " " + \
                                    str(lTheCellTypeName) + " " + \
                                    str(xmin) + " " + \
                                    str(xmax) + " " + \
                                    str(ymin) + " " + \
                                    str(ymax) + " " + \
                                    str(zmin) + " " + \
                                    str(zmax) + "\n"
                                # print lOutputLine
                                lFile.write(lOutputLine)
                                lPixelID +=1

            lFile.close()

            # create 'flag' file, and make sure that it's writable:
            # erase any 'flag' file:
            lFlagFileName = os.path.join(self.theHelperOutputDirectoryForCellDraw, "flagfile.text")
            try:
                lFile = open(lFlagFileName, 'w')
            except:
                print " ||||||||||||| in CompuCell3D steppable:  'HelpfileCellDrawSteppable' step(mcs = %s) can not write file %s ||||||||||||| "%(mcs, lFlagFileName)
                return False
            lFile.write("PIFF output from CC3D done.\n")
            lFile.close()

            self.savedPIF = True
       
#         print "   self.cellList   =", self.cellList
#         print "   self.cellListByType()   =", self.cellListByType()
#         for cell in self.cellList:
#             print "CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume, " cell =", cell

#
#         for cell in self.cellList:
#             if cell.type is not 1:
#                 print "CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
# #         if not ( mcs % 200 ):
#         counter=0
#         for cell in self.cellListByType(1):
#             if cell.type is not 1:
#                 print "BY TYPE CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
#                 counter+=1
#         for cell in self.cellListByType(2):
#             if cell.type is not 1:
#                 print "BY TYPE CELL ID=",cell.id, " CELL TYPE=",cell.type," volume=",cell.volume
#                 counter+=1
#        
#         print "number of cells in typeInventory=",len(self.cellListByType)
#         print "number of cells in the entire cell inventory=",len(self.cellList)               

    def finish(self):
        print "Maybe this function could be called once after simulation?"
        #  TODO TODO: find how to call the "quit" event for closing CompuCell cleanly from here...
        os._exit(-1337)
        # UI.UserInterface.closeEvent()
        # print "UI.UserInterface = " , UI.UserInterface
        # print "dir(UI.UserInterface) = ", dir(UI.UserInterface) # .closeEvent()
