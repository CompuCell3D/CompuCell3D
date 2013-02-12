versionMajor=3
versionMinor=6
versionBuild=2
revisionNumber="20130211"

def getVersionAsString():
    return str(versionMajor)+"."+str(versionMinor)+"."+str(versionBuild)
    
def getVersionMajor():
    return versionMajor
    
def getVersionMinor():
    return versionMinor

def getVersionBuild():
    return versionBuild

def getSVNRevision():
    return revisionNumber

def getSVNRevisionAsString():    
    return str(getSVNRevision())
