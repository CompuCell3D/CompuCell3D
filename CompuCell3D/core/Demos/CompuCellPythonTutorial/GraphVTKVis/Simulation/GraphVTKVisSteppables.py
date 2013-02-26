from PySteppables import *
import CompuCell
import sys
import networkx as nx
from math import *
import colorsys
import vtk
from vtk.util.colors import banana, plum
import networkx
import CompuCellSetup

class ConeSteppable(SteppableBasePy):
    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        
        self.visData=CompuCellSetup.createCustomVisPy("CustomCone")                
        self.visData.registerVisCallbackFunction(self.visualize)
        self.visData.addActor("cone","vtkActor")
        
    def visualize(self,_actorsDict):
        # create cone
        cone = vtk.vtkConeSource()
        cone.SetResolution(60)
#         cone.SetCenter(-2,0,0)
         
        # mapper
        coneMapper = vtk.vtkPolyDataMapper()
        coneMapper.SetInput(cone.GetOutput())
         
        # actor
        coneActor = _actorsDict["cone"]
        coneActor.SetMapper(coneMapper)        



class GraphVTKVisSteppable(SteppableBasePy):    

    def __init__(self,_simulator,_frequency=10):
        SteppableBasePy.__init__(self,_simulator,_frequency)
    def start(self):
        try:
            import networkx
        except ImportError,e:
            return
        
        self.visData=CompuCellSetup.createCustomVisPy("CustomGraph")                
        self.visData.registerVisCallbackFunction(self.visualize)
        self.visData.addActor("glyph","vtkActor")
        self.visData.addActor("profile","vtkActor")
        self.visData.addActor("nodeColorBar","vtkScalarBarActor")
        self.visData.addActor("nodeSizeText","vtkTextActor")
        self.visData.addActor("edgeColorBar","vtkScalarBarActor")
        self.visData.addActor("edgeSizeText","vtkTextActor")
        
        self.cellIdDict = {}
        self.cellPosDict= {}
            
        self.cellContactNetwork=nx.Graph()
        
        self.updateCellIdDict()
        self.makeContactNetwork(self.cellContactNetwork, fillDegrees=True)

    def GenTable(self, lo, hi):
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(hi - lo + 1)
        lut.Build()
        golden_ratio = 0.618033988749895
        h = 0.84521183465356875
        for i in range(0, hi+1):
            h += golden_ratio
            h %= 1
            s = v = 0.95
            HSV_tuple = (h, 0.95, 0.95)       
            (r, g, b) = colorsys.hsv_to_rgb(*HSV_tuple)
            lut.SetTableValue(i, r, g, b, 1.0)
            ##void 	SetTableValue (vtkIdType indx, double rgba[4])
            ##void 	SetTableValue (vtkIdType indx, double r, double g, double b, double a=1.0)
        return lut                

    def step(self,mcs):
        try:
            import networkx
        except ImportError,e:
            return
        
        self.updateCellIdDict()
        self.makeContactNetwork(self.cellContactNetwork, fillDegrees=True)
        
        G = self.cellContactNetwork
        
        ## We can easily get information about connectivity
# # #         print "There are", nx.number_connected_components(G), "connected components."
        c = 0
        mainComponent = None
        maxNodes = 0    
        for component in nx.connected_components(G):
            for cellid in component:
                G.node[cellid]['type'] = c
                cell = self.getCellWithId(cellid)
#                 cell.type = c
            if len(component) > maxNodes:
                maxNodes = len(component)
                mainComponent = component
            c+=1
            
# # #         print "G.nodes()=",G.nodes()
        
    def visualize(self,_actorsDict):
        print "inside visualize"        
        nodeRadius=1
        nodeRadiusAttr='volume'
        nodeColorAttr='type'
        edgeRadius=.1
        edgeRadiusAttr=''
        edgeColorAttr='commonSurfaceArea'
        
#         print CompuCellSetup.graphStorage.graphDict

        G=self.cellContactNetwork
# # #         print "G=",G.nodes()
        nodePoints = vtk.vtkPoints()
        nodeAttr = vtk.vtkFloatArray()
        nodeAttr.SetNumberOfComponents(2)
        nodeAttr.SetName("NodeAttr")
        
        #     if unicode(nodeColorAttr).isdecimal():
        #         setNodeColor = float(nodeRadiusAttr)
        #         nodeRadiusAttr = ''

        numCells = 1
        nodeSizeHi = None
        nodeSizeLo = None
        nodeColorHi = None
        nodeColorLo = None
        
            
        if len(G.nodes()) < 1:
            return
            # raise networkx.NetworkXError, "graph is empty"
# # #         print " INIT ACTORS G.nodes()=",G.nodes()

        for n in G.nodes():
            try:
                (x, y, z) = G.node[n]['position']
            except KeyError:
                raise networkx.NetworkXError, "node %s doesn't have position"%n
            nodePoints.InsertNextPoint(x, y, z)
                
            if nodeRadiusAttr:
                size = G.node[n][nodeRadiusAttr]
    #             print "Cell", n, "has", nodeRadiusAttr, size
                if nodeSizeHi != None:
                    nodeSizeHi = max(size, nodeSizeHi)
                    nodeSizeLo = min(size, nodeSizeLo)
                else:
                    nodeSizeHi = nodeSizeLo = size
            else:
                size = 0
        
            if nodeColorAttr:
                color = G.node[n][nodeColorAttr]
    #             print "Cell", n, "has", nodeColorAttr, color
                if nodeColorHi != None:
                    nodeColorHi = max(color, nodeColorHi)
                    nodeColorLo = min(color, nodeColorLo)
                else:
                    nodeColorHi = nodeColorLo = color
            else:
                color = 0
            
            nodeAttr.InsertNextTuple2(size, color) 

            numCells+=1
            
        
        
        inputData = vtk.vtkPolyData()
        inputData.SetPoints(nodePoints)
        if nodeColorAttr or nodeRadiusAttr:
            inputData.GetPointData().SetScalars(nodeAttr)
        inputData.Modified()
        inputData.Update()

        # Use sphere as glyph source.
        balls = vtk.vtkSphereSource()
        balls.SetRadius(nodeRadius)
        balls.SetPhiResolution(20)
        balls.SetThetaResolution(20)

        ## Could clamp glyph sizes: 
        ##http://www.vtk.org/Wiki/VTK/Examples/Python/Visualization/ClampGlyphSizes
        
        glyphPoints = vtk.vtkGlyph3D()
        glyphPoints.SetInput(inputData)
        glyphPoints.SetSourceConnection(balls.GetOutputPort())
        glyphPoints.SetColorModeToColorByScalar()
        if nodeRadiusAttr:
            glyphPoints.SetScaleFactor(1)
            glyphPoints.SetScaleModeToScaleByScalar()
            glyphPoints.ClampingOn()
            glyphPoints.SetRange(-(nodeSizeHi/2), nodeSizeHi)
        else:
            glyphPoints.SetScaleModeToDataScalingOff()
            
            
        glyphMapper = vtk.vtkPolyDataMapper()
        glyphMapper.SetInputConnection(glyphPoints.GetOutputPort())
        if nodeColorAttr:
            glyphMapper.ScalarVisibilityOn()
            glyphMapper.SetColorModeToMapScalars()
            glyphMapper.ColorByArrayComponent('nodeAttr',1)
            glyphMapper.SetScalarRange(nodeColorLo, nodeColorHi)
            if nodeColorAttr=='type':
                ## Function GenTable defined below
                glyphMapper.SetLookupTable(self.GenTable(nodeColorLo, nodeColorHi))            
        else:
            glyphMapper.ScalarVisibilityOff()
        
        
        #(self.glyph,self.nodeColorBar,self.nodeSizeText,self.profile,self.edgeColorBar,self.edgeSizeText) actor order
        
        glyph = _actorsDict["glyph"]
        glyph.SetMapper(glyphMapper)
        if not nodeColorAttr:
            glyph.GetProperty().SetDiffuseColor(plum)
        glyph.GetProperty().SetSpecular(.3)
        glyph.GetProperty().SetSpecularPower(30)

        
        if nodeColorAttr:

            nodeColorBar = _actorsDict["nodeColorBar"]
            nodeColorBar.SetLookupTable(glyphMapper.GetLookupTable())
            nodeColorBar.SetTitle('Node ' + nodeColorAttr)
            nodeColorBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            nodeColorBar.GetPositionCoordinate().SetValue(0.1, 0.9)
            nodeColorBar.SetOrientationToHorizontal()
            nodeColorBar.SetWidth(.8)
            nodeColorBar.SetHeight(.1)
        
        if nodeRadiusAttr:

            nodeSizeText = _actorsDict["nodeSizeText"]
            nodeSizeText.SetDisplayPosition(10, 550)
            nodeSizeText.SetInput("Node Sizes: "+nodeRadiusAttr)
            node_tprop = nodeSizeText.GetTextProperty()
            node_tprop.SetFontSize(16)
            node_tprop.SetFontFamilyToArial()
            node_tprop.BoldOn()
            node_tprop.ItalicOn()
        
        # Generate the polyline for the spline.
        points = vtk.vtkPoints()
        edgeData = vtk.vtkPolyData()

        # Edges

        lines = vtk.vtkCellArray()
        edgeAttr = vtk.vtkFloatArray()
        edgeAttr.SetNumberOfComponents(2)
        edgeAttr.SetName("EdgeAttr")
        
        edgeSizeHi = 0
        edgeColorHi = 0
        i=0
        for e in G.edges_iter():
            # The edge e can be a 2-tuple (Graph) or a 3-tuple (Xgraph)
            u=e[0]
            v=e[1]
            if edgeRadiusAttr:
                size = G[u][v][edgeRadiusAttr]
                edgeSizeHi = max(size, edgeSizeHi)
            else:
                size = 0
                
            if edgeColorAttr:
                color = G[u][v][edgeColorAttr]
                edgeColorHi = max(color, edgeColorHi) 
            else:
                color = 0
           
            edgeAttr.InsertNextTuple2(size, color) 
            
            lines.InsertNextCell(2)
            for n in (u,v):
                (x,y,z)=G.node[n]['position']
                points.InsertPoint(i, x, y, z)
                lines.InsertCellPoint(i)
                i=i+1

        edgeData.SetPoints(points)
        edgeData.SetLines(lines)
        if edgeColorAttr or edgeRadiusAttr:
            edgeData.GetCellData().SetScalars(edgeAttr)
            edgeData.Modified()
            edgeData.Update()

        # Add thickness to the resulting line.
        Tubes = vtk.vtkTubeFilter()
        Tubes.SetNumberOfSides(16)
        Tubes.SetInput(edgeData)
        Tubes.SetRadius(edgeRadius)
        if edgeRadiusAttr:
            Tubes.SetVaryRadiusToVaryRadiusByScalar()
            Tubes.SetVaryRadius(6)
            Tubes.SetRadiusFactor(5)
        else:
            Tubes.SetVaryRadiusToVaryRadiusOff()

        #
        profileMapper = vtk.vtkPolyDataMapper()
        profileMapper.SetInputConnection(Tubes.GetOutputPort())
        if edgeColorAttr:
            profileMapper.ScalarVisibilityOn()
            profileMapper.SetColorModeToMapScalars()
            profileMapper.ColorByArrayComponent('edgeAttr',1)
            profileMapper.SetScalarRange(0, edgeColorHi)
        else:
            profileMapper.ScalarVisibilityOff()

        profile= _actorsDict["profile"]
        profile.SetMapper(profileMapper)
        if not edgeColorAttr:
            profile.GetProperty().SetDiffuseColor(banana)
        profile.GetProperty().SetSpecular(.3)
        profile.GetProperty().SetSpecularPower(30)
        
        if edgeColorAttr:

            edgeColorBar = _actorsDict["edgeColorBar"]

            edgeColorBar.SetLookupTable(profileMapper.GetLookupTable())
            edgeColorBar.SetTitle('Edge ' + edgeColorAttr)
            edgeColorBar.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
            edgeColorBar.GetPositionCoordinate().SetValue(0.1, 0.01)
            edgeColorBar.SetOrientationToHorizontal()
            edgeColorBar.SetWidth(.8)
            edgeColorBar.SetHeight(.1)
            
        if edgeRadiusAttr:
            edgeSizeText = _actorsDict["edgeSizeText"]
            
            edgeSizeText.SetDisplayPosition(10, 520)
            edgeSizeText.SetInput("Edge Sizes: "+edgeRadiusAttr)
            edge_tprop = edgeSizeText.GetTextProperty()
            edge_tprop.SetFontSize(16)
            edge_tprop.SetFontFamilyToArial()
            edge_tprop.BoldOn()
            edge_tprop.ItalicOn()
        
#         print "ACTORS INITIALIZED"    
           
        
    def breakPeriodicEdges(self, G, div=4):
    # Untested
        xmin = xmax = ymin = ymax = zmin = zmax = -1
        periodicEdges = []      
        for n in G:
            pos = G.node[n]['position']
#             print "Node", n, "is at", pos
#             print "and pos is", type(pos)
            if xmin == -1:
                (xmin, ymin, zmin) = (xmax, ymax, zmax) = pos
            else:
                [xmin, trash, xmax] = sorted([xmin, pos[0], xmax])
                [ymin, trash, ymax] = sorted([ymin, pos[1], ymax])
                [zmin, trash, zmax] = sorted([zmin, pos[2], zmax])
        maxFeasibleDistance = self.euclideanDistance((xmin, ymin, zmin), \
        (xmax, ymax, zmax))/div
        for n1, n2 in G.edges_iter():
            if G[n1][n2]['distance'] >= maxFeasibleDistance:
                periodicEdges.append((n1,n2))
        for n1, n2 in periodicEdges:
            G.remove_edge(n1,n2)
        
        
    def euclideanDistanceBetweenCells(self, cell1, cell2):
#         pt1=CompuCell.Point3D()
#         pt1.x=int(round(cell1.xCOM))
#         pt1.y=int(round(cell1.yCOM))
#         pt1.z=int(round(cell1.zCOM))
#         pt2=CompuCell.Point3D()
#         pt2.x=int(round(cell2.xCOM))
#         pt2.y=int(round(cell2.yCOM))
#         pt2.z=int(round(cell2.zCOM))
        pt1 = (cell1.xCOM, cell1.yCOM, cell1.zCOM)
        pt2 = (cell2.xCOM, cell2.yCOM, cell2.zCOM)
        return self.euclideanDistance(pt1, pt2)
        
    def euclideanDistance(self, pt1, pt2):
        return sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2 + (pt1[2]-pt2[2])**2)
        

    def getCellPositions(self, cList = []):
        if not cList:
            return self.cellPosDict
        else:pass #do something
            
    def getCellWithId(self, cellId):
        return self.cellIdDict[cellId]
        
    def getConcentrationFieldAtCell(self, fieldName, cell):
        chemField=CompuCell.getConcentrationField(self.simulator,fieldName)
        pt=CompuCell.Point3D()
        pt.x=int(round(cell.xCOM))
        pt.y=int(round(cell.yCOM))
        pt.z=int(round(cell.zCOM))
#         pt.x=int(round(cell.xCM/max(float(cell.volume),0.001)))
#         pt.y=int(round(cell.yCM/max(float(cell.volume),0.001)))
#         pt.z=int(round(cell.zCM/max(float(cell.volume),0.001)))
        return chemField.get(pt)
        
    def getConcentrationFieldAtCellId(self, fieldName, cellId):
        return getConcentrationFieldAtCell(self.getCellWithId(cellId))
    
    def getEdgeAttrDict(self, G, attr):
        D = {}
        for u,v in G.edges():
            D[u,v]=G[u][v][attr]
        return D
        
    def getMinConcentrationFieldAtCellId(self, fieldName, cellId):
        cell = self.getCellWithId(cellId)
        chemField=CompuCell.getConcentrationField(self.simulator,fieldName)
        pixelList=CellPixelList(self.pixelTrackerPlugin,cell)
        minConc = 1000000
        for pixelData in pixelList:
           pt=pixelData.pixel
           conc = chemField.get(pt)
           minConc = min(minConc, conc)
        return minConc

        
    def getNodeAttrDict(self, G, attr):   
        D = {}
        for n in G:
            D[n]=G.node[n][attr]
        return D
    
    def getRandomCell(self):
        numCells = len(self.cellList)
        if numCells<=1:
            raise NameError('Empty Simulation')
        validCell = False
        while validCell == False:
            toCheckNum = random.randint(1,numCells)
            print "There are", numCells, "cells in the simulation and we are checking number", toCheckNum, "."
            CL = self.cellList
            CLiter = CL.__iter__()
            for i in range(0,toCheckNum):
                cell=CLiter.next()
            if cell:
                return cell
            else:
                print "Randomly picked medium. Retrying."    
                
#     def getRandomCellOfType(self, typeId):
#         numCells = len(self.cellList)
        
#         if numCells<=1:
#             raise NameError('Empty Simulation')
#         l = []
#         CL = self.cellList       
#         CLiter = CL.__iter__()
#         for i in range(0,numCells):
#             cell=CLiter.next()
#             if cell and cell.type == typeId:
#                 l.append(cell)
#         numCellsOfType = len(l)
# #         return l[random.randint(1,numCellsOfType)-1]     
#         return random.choice(l)

    def getRandomCellOfType(self, G, typeId):
        cellId = random.sample(G.graph['cellsOfType'][typeId], 1)[0]
        return self.getCellWithId(cellId)
        
    def makeContactNetwork(self, G, fillDegrees=False):
        def add_cell_to_graph(cell, G):
            pt3d = (cell.xCOM, cell.yCOM, cell.zCOM)
            G.add_node(cell.id, type=cell.type, volume=cell.volume, \
            surface=cell.surface, position=pt3d)
            self.cellPosDict[cell.id] = pt3d
            G.graph['cellsOfType'][cell.type].add(cell.id)
        
        G.clear()
        G.graph['cellsOfType'] = [set()] * 255
        typeDict = {}
        CL = self.cellList
        for l1Cell in CL:
            add_cell_to_graph(l1Cell, G)
            G.node[l1Cell.id]['contactsMedium']=False
            typeDict[l1Cell.id] = l1Cell.type
            cellNeighborList=self.getCellNeighbors(l1Cell)    
            for neighborSurfaceData in cellNeighborList:
                if neighborSurfaceData.neighborAddress:
                    l2Cell = neighborSurfaceData.neighborAddress
                    add_cell_to_graph(l2Cell, G)
                    G.add_edge(l1Cell.id, l2Cell.id, \
                    distance=self.euclideanDistanceBetweenCells(l1Cell, l2Cell), \
                    commonSurfaceArea=neighborSurfaceData.commonSurfaceArea)
                else:
                    G.node[l1Cell.id]['contactsMedium']=True                
#                     print "Added edge from", cell.id, "to", l2Cell.id
        if fillDegrees:
            self.updateNetworkAttributes(G, 'degree') 
         
    def mergeGraphNodes(self, G, node1id, node2id):
    #Basically untested, but should be pretty cool
    #First get a new COM with a weighted average (not perfect, but it should work okay)
        degreeDict = nx.degree(G,[node1id,node2id])
        sizeDict = {node1id: G.node[node1id]['volume'], node2id: G.node[node2id]['volume']}
        if G.degree(node1id) != G.degree(node2id):
            [smallnode, bignode] = sorted(degreeDict.keys(), key=degreeDict.get)
        elif G.node[node1id]['volume'] != G.node[node2id]['volume']:
            [smallnode, bignode] = sorted(sizeDict.keys(), key=sizeDict.get)
        else:
            [smallnode, bignode] = [node1id, node2id]        
        newVol = sum(sizeDict.values())
        
        for nbr in G[smallnode]:
            if nbr != bignode:
                G.add_edge(bignode, nbr)
                for attr in G[smallnode][nbr]:
                    G[bignode][nbr][attr] = G[smallnode][nbr][attr]
        #Fix edge attributes?
        
        G.node[bignode]['degree'] += G.node[smallnode]['degree']
        G.remove_node(smallnode)
        
        #TODO: something sensible with this.
 
    def pruneBranches2(self, G, thresh=.25): #Based on link lengths
        def median(l):
            if l:
                idx = len(l)/2  #integer division
                if len(l)%2 == 1:
                    return l[idx]
                else:
                    return (l[idx] + l[idx-1])/2
        
        medianLen = None
        pruned = False
        while not pruned:
            pruned = True 
            lenDict = self.getEdgeAttrDict(G, 'distance')
            edges = sorted(lenDict.keys(), key=lenDict.get)
            if medianLen == None:
                 medianLen = median(lenDict.values())           
            for e in edges:
                if lenDict[e] <= medianLen*thresh:
                    print "Attempting to merge nodes", e[0], 'and', e[1]
                    self.mergeGraphNodes(G, *e)
                    pruned = False
                    break
                    
    def pruneBranches(self, G): #Based on high-degree neighbors - use on a pruned MST, before merging chains
        nodesToMerge = set()
        for n in G:
            if G.degree(n) >= 3:
                for nbr in G[n]:
                    if G.degree(nbr) >= 3:
                        nodesToMerge.add(n)
                        nodesToMerge.add(nbr)
                        
        SG = G.subgraph(nodesToMerge)
        print "pruneBranches2 is going to merge the following node clusters:", nx.connected_components(SG)
        #We're going to merge each connected component in this subgraph into a single node
        for C in nx.connected_components(SG):
            if len(C) <= 1: #should always be false
                raise InputError('Single node in pruneBranches2 subgraph')
            else:
                maxDegree = 0
                maxNode = None
                for n in C:
                    if SG.degree(n) > maxNode:
                        maxDegree = SG.degree(n)
                        maxNode = n
                C.remove(maxNode)
                for n in C:
                    self.mergeGraphNodes(G, maxNode, n)
                    
          
 
    def pruneLeaves(self, G):
    # Checks each node, if it has degree 1 and is connected to a node with degree > 2, 
    # it lops it off. This removes spurious branches from the minimum spanning tree.
    # It leaves 0-degree nodes alone, so this is best combined with a 1-core.
    
        # Decide how we're going to get the degree info.
#         if useStoredDegree:
#             deg = lambda n: G.node[n]['degree']
#         else:
#             deg = lambda n: G.degree(n)
        leavesToDelete = []    
        for n1 in G:
#             print "Getting degree of node", n1
            degree = G.degree(n1)
            if degree > 2:
                nbrDegrees = {}
                for n2 in G[n1].keys():
                    nbrDegrees[n2] = G.degree(n2)
                leaves = [n for n in nbrDegrees.keys() if nbrDegrees[n] == 1]
                # The following condition prevents chopping off all the leaves, making a new one.
                if (degree - len(leaves)) > 1:
                    leavesToDelete += leaves
#                     print "leavesToDelete:", leavesToDelete
        G.remove_nodes_from(leavesToDelete)

                
    def simplifyChains(self, G):
    # A chain link is defined as a node of degree 2. This function removes all
    # such nodes from the graph, attempting to preserve edge attributes such
    # as weight.
        simplified = False
        while not simplified:
            simplified = True
            for n in G:
                # if the node is a chain link, remove that link & update length
                if G.degree(n) == 2:
                    simplified = False
                    (n1, n2) = G[n]
                    old_edge_attrs = set(G[n][n1].keys() + G[n][n2].keys())
                    G.add_edge(n1, n2)
                    for attr in old_edge_attrs:
                        if G[n][n1][attr] == G[n][n2][attr]:
                            G[n1][n2][attr] = G[n][n1][attr]
                        else:
                            G[n1][n2][attr] = None
                    G[n1][n2]['distance'] = G[n][n1]['distance'] + G[n][n2]['distance']
                    G.remove_node(n)


    def updateCellIdDict(self):
        for cell in self.cellList:
            self.cellIdDict[cell.id]=cell  

    def updateNetworkAttributes(self, G, *args):
        
        if 'type' in args:
            G.graph['cellsOfType'] = [set()] * 255
            
        for n in G:
            cell = self.getCellWithId(n)
            if 'position' in args:
                G.node[n]['centerOfMass']=(cell.xCOM, cell.yCOM, cell.zCOM)
                self.cellPosDict[n] = (cell.xCOM, cell.yCOM, cell.zCOM)
                
            if 'type' in args:
                G.node[n]['type'] = cell.type
                G.graph['cellsOfType'][cell.type].add(n)
    
            if 'degree' in args:
                G.node[n]['degree']=G.degree(n)
                
            if 'surface' in args:
                G.node[n]['surface'] = cell.surface
                
            if 'volume' in args:
                G.node[n]['volume'] = cell.volume
                
            def get_distance(cell1, cell2):
                if isinstance(cell1, (int, long )):
                    cell1 = self.getCellWithId(cell1)
                if isinstance(cell2, (int, long )):
                    cell2 = self.getCellWithId(cell2)
                if 'position' in args: #use current position data
                    distance = self.euclideanDistanceBetweenCells\
                    (cell1, cell2)
                else: #use stored position data
                    distance = self.euclideanDistance\
                    (G.node[cell1.id]['position'],G.node[cell2.id]['position'])
                return distance
                              
            if 'neighbors' in args: #also updates distance and commonSurfaceArea
                G.node[n]['contactsMedium']=False  
                cellNeighborList=self.getCellNeighbors(cell)    
                for neighborSurfaceData in cellNeighborList:
                    if neighborSurfaceData.neighborAddress:
                        l2Cell = neighborSurfaceData.neighborAddress
                        if l2Cell.id in G:
                            G.add_edge(n, l2Cell.id, \
                            distance=get_distance(cell,l2Cell), \
                            commonSurfaceArea=neighborSurfaceData.commonSurfaceArea)
                    else:
                        G.node[n]['contactsMedium']=True
            else:
                if 'distance' in args:
                    for neighbor in G[n]:
                        G[n][neighbor]['distance'] = get_distance(n,neighbor)
                        
                if 'commonSurfaceArea' in args:pass
                    #Do this later       
    #### END  def updateNetworkAttributes





