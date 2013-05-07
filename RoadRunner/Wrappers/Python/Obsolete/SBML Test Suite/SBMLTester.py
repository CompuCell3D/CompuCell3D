import os
import fnmatch
import csv
import re
import rrPython
path = 'C:/SBML_test_cases/'
os.chdir(path)

it = []#for name of xml file
ip = []#for name of settings file
ic = []#for name of results file
for path, dirs, files in os.walk(os.getcwd()):
    for xml in [os.path.abspath(os.path.join(path, filename)) for filename in files if fnmatch.fnmatch(filename, '*l2v4.xml')]:
        it.append(xml)
        ip.append(xml)
        ic.append(xml)
        f = open(xml,'r').read()
        rrPython.loadSBML(f)
        b = ""
        j=0
        while j<len(it[-1])-13:
            b+=it[-1][j]
            j+=1
        ip[-1]=b+"settings.txt"
        ic[-1]=b+"results.csv"
        concheck = 0 #used later to check if values should be concentrations or amounts
        f=open(ip[-1]) #loading model parameters
        for line in f: # reads all parameters from settings.txt
                text = line.split()
                if len(text) > 0: #checks for something in line
                        if text[0] == 'start:':
                                startTime = float(text[1])
                        if text[0] == 'duration:':
                                endTime = startTime + float(text[1])
                        if text[0] == 'steps:':
                                numberOfPoints = int(text[1])+1#add one to match online sbml validation
                        if text[0] == 'variables:':
                                varlist = text[1:]
                        if text[0] == 'amount:':
                                if len(text) >1:
                                    concheck = 1
                        if text[0] == 'concentration:':
                                if len(text) >1:
                                    amountcheck = 1
        rrPython.setTimeStart(startTime)
        rrPython.setTimeEnd(endTime)
        rrPython.setNumPoints(numberOfPoints)
        #if concheck == 1:
        #    compvalues = rrPython.getCompartmentByIndex()
        curtime = ['time']

        species = curtime + varlist
        #i=0
        #pattern = re.compile('[\W_]+')
        #for species[i] in species:
        #    species[i] = pattern.sub('',species[i])#removes non-alphanumeric characters from all variables names. Probably no longer needed.
        #    i+=1

        species = str(species).strip("[]")
        species = species.replace("'", "")
        species = species.replace(" ", "")
        rrPython.setSelectionList(species)
        #rrPython.ComputeAndAssignConservationLaws(0)
        k = rrPython.simulate()
        kk = []
        kk = k.split('\n')

############The block of code below was used to multiply values by the compartment
############volume to get results in moles instead of concentration where needed

        #iterate = 0
        #if concheck == 1:
        #    for iterate in range(len(kk)):
        #        iterate2 = 1
        #        kkelement = kk[iterate]
        #        for iterate2 in range(len(kkelement)):
        #            if iterate2 == 0:
        #                kkelement[iterate2] = kkelement[iterate2]
        #            else:
        #                kkelement[iterate2] = kkelement[iterate2]*compvalues
        #            iterate2+=1
#
#                kk[iterate] = kkelement
#                iterate +=1


#############
#############
        p = open('C:/Pyfiles/sbml.csv','w')
        writer = csv.writer(p, delimiter = '\n')
        writer.writerows([kk])
        p.close()

        q = 'C:/Pyfiles/sbml.csv'
        w = q[:-4] + ic[-1][25:]
        os.rename('C:/Pyfiles/sbml.csv',w)
        concheck = 0
        list.list
        kk = []
        k = ''
        list.list

