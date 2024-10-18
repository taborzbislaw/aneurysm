import logging
import os
from typing import Annotated, Optional
import heapq
import math
import numpy as np
import json
import SimpleITK as sitk

import qt
import vtk

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode
from slicer import vtkMRMLSegmentationNode
from slicer import vtkMRMLMarkupsNode
from slicer import vtkMRMLMarkupsFiducialNode
from slicer import qMRMLNodeComboBox


try:
    from scipy.ndimage import distance_transform_edt,label
except ModuleNotFoundError:
    slicer.util.pip_install("scipy")
    from scipy.ndimage import distance_transform_edt,label

try:
    from skimage import measure, io
except ModuleNotFoundError:
    slicer.util.pip_install("scikit-image")
    from skimage import measure,io

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    slicer.util.pip_install("matplotlib")
    import matplotlib.pyplot as plt


# Finds a path between two 3D points start and end in a 3D image img
# The procedure uses Dijkstra's algorithm
   
def find_path_dijkstra(img, start, end):

    imMax = np.max(img)
   
    way = np.zeros(img.shape + (3,),dtype=np.uint16)

    summed = np.ones(img.shape,dtype=np.float32)*np.prod(img.shape)*np.max(img)
    summed[start[0],start[1],start[2]] = 0

    priority_queue = []
    heapq.heappush(priority_queue,(summed[start[0],start[1],start[2]],start))

    while priority_queue:
        priority, pix = heapq.heappop(priority_queue)
        #print(priority)
        if pix == end:
            break
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if pix[0]+i>=0 and pix[0]+i<img.shape[0] and pix[1]+j>=0 and pix[1]+j<img.shape[1] and pix[2]+k>=0 and pix[2]+k<img.shape[2]:
                        if img[pix[0]+i,pix[1]+j,pix[2]+k] == imMax:
                            continue
                        if i*i+j*j+k*k >= 1:
                            #print(pix[0]+i,pix[1]+j,pix[2]+k)
                            if img[pix[0]+i,pix[1]+j,pix[2]+k] + summed[pix[0],pix[1],pix[2]] < summed[pix[0]+i,pix[1]+j,pix[2]+k]:
                                way[pix[0]+i,pix[1]+j,pix[2]+k,0] = pix[0]
                                way[pix[0]+i,pix[1]+j,pix[2]+k,1] = pix[1]
                                way[pix[0]+i,pix[1]+j,pix[2]+k,2] = pix[2]
                                summed[pix[0]+i,pix[1]+j,pix[2]+k] = img[pix[0]+i,pix[1]+j,pix[2]+k] + summed[pix[0],pix[1],pix[2]]
                                heapq.heappush(priority_queue, (summed[pix[0]+i,pix[1]+j,pix[2]+k],(pix[0]+i,pix[1]+j,pix[2]+k)))
                   
    path = [end]
    while tuple(path[-1]) != tuple(start):
        path.append((way[path[-1][0],path[-1][1],path[-1][2],0],way[path[-1][0],path[-1][1],path[-1][2],1],way[path[-1][0],path[-1][1],path[-1][2],2]))
       
    return path

def find_path_bfs(img, start, end):
    
    priority_queue = []
    visited = np.zeros_like(img)
    way = np.zeros(img.shape + (3,),dtype=np.uint16)
    
    heapq.heappush(priority_queue,(img[start[0],start[1],start[2]],start))
    visited[start[0],start[1],start[2]] = 1
    
    step = 0
    while priority_queue:
        
        if step%100000 == 0:
            #workDir = os.path.dirname(__file__)
            #f = open(workDir + '/log.txt','a')
            #print(step,len(priority_queue),np.prod(img.shape),file=f)
            #f.close()
            print(step,len(priority_queue),np.prod(img.shape))
            
        priority, pix = heapq.heappop(priority_queue)
        #print(len(priority_queue))
        if pix == end:
            break
        for i in range(-1,2):
            for j in range(-1,2):
                for k in range(-1,2):
                    if pix[0]+i>=0 and pix[0]+i<img.shape[0] and pix[1]+j>=0 and pix[1]+j<img.shape[1] and pix[2]+k>=0 and pix[2]+k<img.shape[2]:
                        if i*i+j*j+k*k >= 1:
                            #print(pix[0]+i,pix[1]+j,pix[2]+k)
                            if visited[pix[0]+i,pix[1]+j,pix[2]+k] == 0:
                                visited[pix[0]+i,pix[1]+j,pix[2]+k] = 1
                                way[pix[0]+i,pix[1]+j,pix[2]+k,0] = pix[0]
                                way[pix[0]+i,pix[1]+j,pix[2]+k,1] = pix[1]
                                way[pix[0]+i,pix[1]+j,pix[2]+k,2] = pix[2]
                                heapq.heappush(priority_queue, (img[pix[0]+i,pix[1]+j,pix[2]+k],(pix[0]+i,pix[1]+j,pix[2]+k)))
        step += 1
                    
    path = [end]
    while tuple(path[-1]) != tuple(start):
        path.append((way[path[-1][0],path[-1][1],path[-1][2],0],way[path[-1][0],path[-1][1],path[-1][2],1],way[path[-1][0],path[-1][1],path[-1][2],2]))
        
    return path

# A simple procedure for finding multiplanar reconstruction projection from a 3D image
# Given a 3D image M, a point P and a vector V, the procedure finds a plane cross section through M
# such that the plane crosses P and is perpendicular to V
# Returns a 2D image - cross section and a 3D coordinates of points within the plane

# An alternative MPR reconstruction using VTK
#https://github.com/srinivasrvaidya/Multi-Planar-Reconstruction-using-VTK/blob/master/MPR/src/slicer.py
    
def MPR(M, P, V, BOX = 10,method='linear'):
    # M is a 3D image from which we want to extract MPR
    # P is a 3D point - a center of MPR projection
    # V is a 3D vector normal to the plane of MPR projection

    # Generate an orthogonal basis with the first vector being V
    basis0 = V / np.linalg.norm(V)
    basis1 = np.cross(P/ np.linalg.norm(P), basis0)
    basis1 = basis1/np.linalg.norm(basis1)
    basis2 = np.cross(basis0,basis1)
    basis2 = basis2/np.linalg.norm(basis2)

    dum = np.copy(M[int(P[0])-BOX:int(P[0])+BOX+1,int(P[1])-BOX:int(P[1])+BOX+1,int(P[2])-BOX:int(P[2])+BOX+1])
    dumP = np.array([BOX,BOX,BOX])
    coords = []
    inter = np.zeros((2*BOX+1,2*BOX+1),dtype=np.float32)
    
    for t1 in range(-BOX,BOX+1,1):
        for t2 in range(-BOX,BOX+1,1):
            plane_coordinates = dumP + t1*basis1 + t2*basis2
            coords.append(plane_coordinates)
            if int(plane_coordinates[0])>=0 and int(plane_coordinates[0])<dum.shape[0] and int(plane_coordinates[1])>=0 and \
                int(plane_coordinates[0])<dum.shape[1] and int(plane_coordinates[2])>=0 and int(plane_coordinates[2])<dum.shape[2]:
                try:
                    inter[int(t1+BOX),int(t2+BOX)] = dum[int(plane_coordinates[0]),int(plane_coordinates[1]),int(plane_coordinates[2])]
                except:
                    pass
    
    return inter,coords

# Fits line to a set of points in 3D
# returns a tuple: a point of a line and a vector pointing along the line

def fit_3d_line(data):
    
    # data is an array of 3D points with shape (N,3)
    
    datamean = data.mean(axis=0)
    # Do an SVD on the mean-centered data.
    uu, dd, vv = np.linalg.svd(data - datamean)
    
    #The line equation is x = datamean + vv[0]*t where t is a real parameter
    
    return datamean,vv[0]

def find_diff_index(list1, list2):
    """Finds the index of the first element that differs in two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        The index of the first differing element, or None if the lists are identical.
    """

    for i in range(min(len(list1), len(list2))):
        if list1[i] != list2[i]:
            return i

    return len(list2) - 1  # Lists are identical

def path_length(points):
    """
    Calculates the length of a path crossing all points in a sequence.

    Args:
      points: A list of 3D points represented as NumPy arrays.

    Returns:
      The total length of the path as a float.
    """

    if len(points) < 2:
        return 0  # No path for less than 2 points

    total_length = 0
    for i in range(1, len(points)):
        # Calculate Euclidean distance between consecutive points
        diff = np.array(points[i]) - np.array(points[i - 1])
        distance = np.linalg.norm(diff)
        total_length += distance

    return total_length

#
# aneurysm
#


class aneurysm(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Aneurysm assesment")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Aneurysm assesment")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Zbisław Tabor (AGH University of Cracow)"]  
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#aneurysm">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""")

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#


def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # aneurysm1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="aneurysm",
        sampleName="aneurysm1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "aneurysm1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="aneurysm1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="aneurysm1",
    )

    # aneurysm2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="aneurysm",
        sampleName="aneurysm2",
        thumbnailFileName=os.path.join(iconsPath, "aneurysm2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="aneurysm2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="aneurysm2",
    )


#
# aneurysmParameterNode
#


@parameterNodeWrapper
class aneurysmParameterNode:
    """
    The parameters needed by module.

    inputVolume - The volume to threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    """

    inputVolume: vtkMRMLScalarVolumeNode
    thresholdedVolume: vtkMRMLSegmentationNode
    markupEntry: vtkMRMLMarkupsFiducialNode 
    markupExit: vtkMRMLMarkupsFiducialNode
    markupCentral: vtkMRMLMarkupsFiducialNode

 
#
# aneurysmWidget
#


class aneurysmWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None
        
        self.distancesAlongPath = None
        self.minDiameters = None
        self.maxDiameters = None
        self.angles = None
        self.minimalAneurysmDiameters = []
        self.maximalAneurysmDiameters = []
        

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.

        uiWidget = slicer.util.loadUI(self.resourcePath("UI/aneurysm.ui"))

        self.layout.addWidget(uiWidget)

        self.ui = slicer.util.childWidgetVariables(uiWidget)

        self.ui.exitPointSelector.setVisible(False)
        self.ui.centralPointSelector.setVisible(False)
        self.ui.label2.setVisible(False)
        self.ui.label3.setVisible(False)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = aneurysmLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)
        self.ui.inputSelector.connect("currentNodeChanged(bool)",self.volumeChanged)
        self.ui.addButton.connect('clicked(bool)',self.addComboBox)
        
        self.ui.radioButtonDiameters.connect('clicked(bool)',self.displayDistances)
        self.ui.radioButtonAngles.connect('clicked(bool)',self.displayAngles)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def displayDistances(self) -> None:
        if self.distancesAlongPath == None:
            return

        results = (np.array(self.minDiameters),np.array(self.distancesAlongPath))
        tableNodeMin=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(tableNodeMin, results)
        tableNodeMin.GetTable().GetColumn(0).SetName("Minimal diameter [mm]")
        tableNodeMin.GetTable().GetColumn(1).SetName("Distance [mm]")
        
        # Create plot
        plotSeriesNodeMin = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "minimal vessel axis [mm]")
        plotSeriesNodeMin.SetAndObserveTableNodeID(tableNodeMin.GetID())
        plotSeriesNodeMin.SetXColumnName("Distance [mm]")
        plotSeriesNodeMin.SetYColumnName("Minimal diameter [mm]")
        plotSeriesNodeMin.SetPlotType(plotSeriesNodeMin.PlotTypeScatter )
        plotSeriesNodeMin.SetColor(0, 0, 1.0)

        results = (np.array(self.maxDiameters),np.array(self.distancesAlongPath))
        tableNodeMax=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(tableNodeMax, results)
        tableNodeMax.GetTable().GetColumn(0).SetName("Maximal diameter [mm]")
        tableNodeMax.GetTable().GetColumn(1).SetName("Distance [mm]")
        
        # Create plot
        plotSeriesNodeMax = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "maximal vessel axis [mm]")
        plotSeriesNodeMax.SetAndObserveTableNodeID(tableNodeMax.GetID())
        plotSeriesNodeMax.SetXColumnName("Distance [mm]")
        plotSeriesNodeMax.SetYColumnName("Maximal diameter [mm]")
        plotSeriesNodeMax.SetPlotType(plotSeriesNodeMax.PlotTypeScatter )
        plotSeriesNodeMax.SetColor(0, 1.0, 0)

        # Create chart and add plot
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
        plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMin.GetID())
        plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMax.GetID())
        
        plotChartNode.SetYAxisTitle('Diameter [mm]')
        plotChartNode.SetXAxisTitle('Distance from aneurysm junction [mm]')
        
        # Show plot in layout
        slicer.modules.plots.logic().ShowChartInLayout(plotChartNode)
            

    def displayAngles(self) -> None:
        if self.distancesAlongPath == None:
            return

        results = (np.array(self.angles),np.array(self.distancesAlongPath))
        tableNodeMin=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        slicer.util.updateTableFromArray(tableNodeMin, results)
        tableNodeMin.GetTable().GetColumn(0).SetName("Minimal diameter [mm]")
        tableNodeMin.GetTable().GetColumn(1).SetName("Distance [mm]")
        
        # Create plot
        plotSeriesNodeMin = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode" , "curvature [degrees]")
        plotSeriesNodeMin.SetAndObserveTableNodeID(tableNodeMin.GetID())
        plotSeriesNodeMin.SetXColumnName("Distance [mm]")
        plotSeriesNodeMin.SetYColumnName("Minimal diameter [mm]")
        plotSeriesNodeMin.SetPlotType(plotSeriesNodeMin.PlotTypeScatter )
        plotSeriesNodeMin.SetColor(0, 0, 1.0)

        # Create chart and add plot
        plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
        plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMin.GetID())
        
        plotChartNode.SetYAxisTitle('Vessel curvature [degrees]')
        plotChartNode.SetXAxisTitle('Distance from aneurysm junction [mm]')
        
        # Show plot in layout
        slicer.modules.plots.logic().ShowChartInLayout(plotChartNode)
            

    def addComboBox(self) -> None:
        if self.ui.exitPointSelector.isVisible() == False:
            self.ui.exitPointSelector.setVisible(True)
            self.ui.label2.setVisible(True)
            if not self._parameterNode.markupExit:
                firstMarkupNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if firstMarkupNode:
                    self._parameterNode.markupExit = firstMarkupNode
            return

        if self.ui.centralPointSelector.isVisible() == False:
            self.ui.centralPointSelector.setVisible(True)
            self.ui.label3.setVisible(True)
            if not self._parameterNode.markupCentral:
                firstMarkupNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
                if firstMarkupNode:
                    self._parameterNode.markupCentral = firstMarkupNode
            return


    def volumeChanged(self,flag) -> None:
        slicer.util.setSliceViewerLayers(background = self.ui.inputSelector.currentNode())
        slicer.util.resetSliceViews()

        if len(self.minimalAneurysmDiameters) > 0:
            if self.ui.inputSelector.currentNode().GetName() == 'CrossSection1':
                self.ui.labelMinimalDiameter.setText(f'Minimal diameter = {self.minimalAneurysmDiameters[0]:.2f} mm')
                self.ui.labelMaximalDiameter.setText(f'Maximal diameter = {self.maximalAneurysmDiameters[0]:.2f} mm')
                workDir = os.path.dirname(__file__)
                f = open(workDir + '/log.txt','w')
                print('1',file=f)
                f.close()
            elif self.ui.inputSelector.currentNode().GetName() == 'CrossSection2':
                self.ui.labelMinimalDiameter.setText(f'Minimal diameter = {self.minimalAneurysmDiameters[1]:.2f} mm')
                self.ui.labelMaximalDiameter.setText(f'Maximal diameter = {self.maximalAneurysmDiameters[1]:.2f} mm') 
                workDir = os.path.dirname(__file__)
                f = open(workDir + '/log.txt','w')
                print('2',file=f)
                f.close()
            elif self.ui.inputSelector.currentNode().GetName() == 'CrossSection3':
                self.ui.labelMinimalDiameter.setText(f'Minimal diameter = {self.minimalAneurysmDiameters[2]:.2f} mm')
                self.ui.labelMaximalDiameter.setText(f'Maximal diameter = {self.maximalAneurysmDiameters[2]:.2f} mm') 
                workDir = os.path.dirname(__file__)
                f = open(workDir + '/log.txt','w')
                print('3',file=f)
                f.close()
            else:
                self.ui.labelMaximalDiameter.setText('')
                self.ui.labelMinimalDiameter.setText('')
                workDir = os.path.dirname(__file__)
                f = open(workDir + '/log.txt','w')
                print('4',self.ui.inputSelector.currentNode().GetName(),file=f)
                f.close()
        else:
            workDir = os.path.dirname(__file__)
            f = open(workDir + '/log.txt','w')
            print('5',file=f)
            f.close()
            self.ui.labelMaximalDiameter.setText('')
            self.ui.labelMinimalDiameter.setText('')
#        workDir = os.path.dirname(__file__)
#        f = open(workDir + '/log.txt','w')
#        print(dir(self.ui.labelMinimalDiameter),file=f)
#        #print(self.ui.inputSelector.currentNode(),file=f)
#        #print(dir(self.ui.inputSelector.currentNode()),file=f)
#        print(self.ui.inputSelector.currentNode().GetName(),file=f)
#        f.close()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.inputVolume:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
            if firstVolumeNode:
                self._parameterNode.inputVolume = firstVolumeNode

        if not self._parameterNode.thresholdedVolume:
            firstSegmentationNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLSegmentationNode")
            if firstSegmentationNode:
                self._parameterNode.thresholdedVolume = firstSegmentationNode

        if not self._parameterNode.markupEntry:
            firstMarkupNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstMarkupNode:
                self._parameterNode.markupEntry = firstMarkupNode


        #workDir = os.path.dirname(__file__)
        #f = open(workDir + '/log.txt','w')
        #out = slicer.mrmlScene.GetNodesByClass("vtkMRMLMarkupsNode")
        #m0 = slicer.mrmlScene.GetNthNodeByClass(0,"vtkMRMLMarkupsFiducialNode")
        #m1 = slicer.mrmlScene.GetNthNodeByClass(1,"vtkMRMLMarkupsFiducialNode")
        #m2 = slicer.mrmlScene.GetNthNodeByClass(2,"vtkMRMLMarkupsFiducialNode")
        #print(m0,file=f)
        #print(m1,file=f)
        #print(m2,file=f)
        #print(self._parameterNode.markupEntry,file=f)
        #print(dir(slicer.mrmlScene),file=f)
        #f.close()


    def setParameterNode(self, inputParameterNode: Optional[aneurysmParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.inputVolume and self._parameterNode.thresholdedVolume:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

    def onApplyButton(self) -> None:
 
 
        segmentationNode = self._parameterNode.thresholdedVolume
        segName = segmentationNode.GetSegmentation().GetSegmentIDs()[0]
        segmentId = segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(segName)
        segArr = slicer.util.arrayFromSegmentBinaryLabelmap(segmentationNode, segmentId, self._parameterNode.inputVolume)

        imArr = slicer.util.arrayFromVolume(self._parameterNode.inputVolume)
                        
        imOrigin = self._parameterNode.inputVolume.GetOrigin()
        imSpacing = self._parameterNode.inputVolume.GetSpacing()

        workDir = os.path.dirname(__file__)
        f = open(workDir + '/log.txt','w')
        print(imSpacing,file=f)
        f.close()

        p1 = self._parameterNode.markupEntry.GetNthControlPointPosition(0)
        p2 = self._parameterNode.markupExit.GetNthControlPointPosition(0)
        p3 = self._parameterNode.markupCentral.GetNthControlPointPosition(0)

        startCandidate = (np.asarray((np.asarray(p1) - np.asarray(imOrigin))/np.asarray(imSpacing),dtype = np.int32))[::-1]
        endCandidate = (np.asarray((np.asarray(p2) - np.asarray(imOrigin))/np.asarray(imSpacing),dtype = np.int32))[::-1]
        aneurysmCenterCandidate = (np.asarray((np.asarray(p3) - np.asarray(imOrigin))/np.asarray(imSpacing),dtype = np.int32))[::-1]

####################################################################
# calculate medial axis points

        edt = distance_transform_edt(segArr)

        BOX_EDT = 0
        if self.ui.edtSpinBox.value > 0:
            BOX_EDT = self.ui.edtSpinBox.value

        maxEDT = np.max(edt)
        edt = maxEDT - edt
        
        if BOX_EDT > 0:
            matrix = np.copy(edt[startCandidate[0]-BOX_EDT:startCandidate[0]+BOX_EDT+1,startCandidate[1]-BOX_EDT:startCandidate[1]+BOX_EDT+1,startCandidate[2]-BOX_EDT:startCandidate[2]+BOX_EDT+1])
            min_indices = np.unravel_index(np.argmin(matrix), matrix.shape)
            start = (startCandidate[0]-BOX_EDT+min_indices[0],startCandidate[1]-BOX_EDT+min_indices[1],startCandidate[2]-BOX_EDT+min_indices[2])

            matrix = np.copy(edt[endCandidate[0]-BOX_EDT:endCandidate[0]+BOX_EDT+1,endCandidate[1]-BOX_EDT:endCandidate[1]+BOX_EDT+1,endCandidate[2]-BOX_EDT:endCandidate[2]+BOX_EDT+1])
            min_indices = np.unravel_index(np.argmin(matrix), matrix.shape)
            end = (endCandidate[0]-BOX_EDT+min_indices[0],endCandidate[1]-BOX_EDT+min_indices[1],endCandidate[2]-BOX_EDT+min_indices[2])

            matrix = np.copy(edt[aneurysmCenterCandidate[0]-BOX_EDT:aneurysmCenterCandidate[0]+BOX_EDT+1,aneurysmCenterCandidate[1]-BOX_EDT:aneurysmCenterCandidate[1]+BOX_EDT+1,aneurysmCenterCandidate[2]-BOX_EDT:aneurysmCenterCandidate[2]+BOX_EDT+1])
            min_indices = np.unravel_index(np.argmin(matrix), matrix.shape)
            aneurysmCenter = (aneurysmCenterCandidate[0]-BOX_EDT+min_indices[0],aneurysmCenterCandidate[1]-BOX_EDT+min_indices[1],aneurysmCenterCandidate[2]-BOX_EDT+min_indices[2])
        else:
            start = tuple(startCandidate)
            end = tuple(endCandidate)
            aneurysmCenter = tuple(aneurysmCenterCandidate)

####################################################################
####################################################################
# Now I find medial axis from BFS algorithm

#        dum = np.copy(edt)
#        m = np.max(edt)
#        dum[start[0]-2:start[0]+3,start[1]-2:start[1]+3,start[2]-2:start[2]+3] = m
#        dum[end[0]-2:end[0]+3,end[1]-2:end[1]+3,end[2]-2:end[2]+3] = m
#        dum[aneurysmCenter[0]-2:aneurysmCenter[0]+3,aneurysmCenter[1]-2:aneurysmCenter[1]+3,aneurysmCenter[2]-2:aneurysmCenter[2]+3] = m
#
        # This is medial axis from start point to end point
        if self.ui.dijkstraCheckBox.checked == True:
            pathStartEnd = find_path_dijkstra(edt,aneurysmCenter,start)
        else:
            pathStartEnd = find_path_bfs(edt,aneurysmCenter,start)

        # This is medial axis from start point to aneurysm center point
        if self.ui.dijkstraCheckBox.checked == True:
            pathAneurysmEnd = find_path_dijkstra(edt,end,aneurysmCenter)
        else:
            pathAneurysmEnd = find_path_bfs(edt,end,aneurysmCenter)

        dumList = pathStartEnd[::-1]
        indexDiff = find_diff_index(dumList,pathAneurysmEnd)
        pathStartEnd += pathAneurysmEnd[indexDiff:]

        indexDiff = find_diff_index(pathStartEnd[::-1],pathAneurysmEnd[::-1])
        indexDiff = len(pathStartEnd) - indexDiff

####################################################################

####################################################################
# Given the medial axis of the vessel we may find cross sections and their diameters
# For this purpose we will go iteratively along the medial axis, finding MPR at the points, and finding diameters of the cross sections

        if self.ui.numOfPointsSpinBox.value > 0:
            NPOINTS = self.ui.numOfPointsSpinBox.value
        else:
            NPOINTS = 10

        STEP = 1            

        if self.ui.paddingSpinBox.value > 0:
            PADDING = self.ui.paddingSpinBox.value # padding for cross section
        else:
            PADDING = int(maxEDT/2)
        
        PIX_SIZE = imSpacing[0]  # pixel size in mm

        min_axes = []
        max_axes = []
        bbox_w = []
        bbox_h = []
        points = []
        vectors = []
        for START in range(len(pathStartEnd)-NPOINTS):
            
            st = START
            en = min(START+NPOINTS,len(pathStartEnd))
            #if en - st < NPOINTS:
            #    st = st - (NPOINTS - (en - st))
            pointsToFit = np.array(pathStartEnd[st:en])
            P,V = fit_3d_line(pointsToFit)
            BoundingBox = maxEDT - edt[int(P[0]),int(P[1]),int(P[2])]
            #print(P,V,BOX)
            points.append(P)
            vectors.append(V)
            

            try:
                dum,coords = MPR(segArr,P,V,BOX=int(BoundingBox) + PADDING)
                dum[dum<0.5] = 0
                
                labels,_ = label(dum)
                idMax = labels[labels.shape[0]//2,labels.shape[1]//2]
                dum[labels!=idMax] = 0
                dum[dum!=0] = 1

                dum = np.asarray(dum,dtype=np.uint8)
                props = measure.regionprops(dum)
                
                min_axes.append(props[0].axis_minor_length*PIX_SIZE)
                max_axes.append(props[0].axis_major_length*PIX_SIZE)

                loc = np.where(dum!=0)
                w = np.max(loc[0]) - np.min(loc[0])
                h = np.max(loc[1]) - np.min(loc[1])
                bbox_w.append(w*PIX_SIZE)
                bbox_h.append(h*PIX_SIZE)
            except:
                min_axes.append(min_axes[-1])
                max_axes.append(max_axes[-1])
                bbox_w.append(bbox_w[-1])
                bbox_h.append(bbox_h[-1])


        ds = [0]
        for n in range(1,len(pathStartEnd)-NPOINTS):
            p1 = np.array(pathStartEnd[n-1])
            p2 = np.array(pathStartEnd[n])
            d = np.sqrt(np.sum(np.square(p1-p2)))
            ds.append(d)
            
        for n in range(1,len(ds)):
            ds[n] += ds[n-1]

        locOfAneurysm = ds[indexDiff]    
        distancesAlongPath = [(x - locOfAneurysm)*PIX_SIZE for x in ds]     


        angles = np.arccos(np.asarray([np.dot(vectors[i],vectors[i-1]) for i in range(1,len(vectors))]))*180./math.pi
        for i in range(len(angles)):
            if angles[i]>90:
                angles[i] = 180 - angles[i]

####################################################################
# przektore przez tętniaka

        def findCrossSection(P,V):
            crossSection,coords = MPR(imArr,P,V,BOX=int(BoundingBox) + PADDING)
            m = np.min(crossSection[crossSection!=0])
            crossSection[crossSection==0] = m
            
            dum,coords = MPR(segArr,P,V,BOX=int(BoundingBox) + PADDING)
            dum[dum<0.5] = 0
            
            labels,_ = label(dum)
            ids,counts = np.unique(labels,return_counts=True)
            idMax = labels[labels.shape[0]//2,labels.shape[1]//2]
            dum[labels!=idMax] = 0
            dum[dum!=0] = 1

            dum = np.asarray(dum,dtype=np.uint8)
            props = measure.regionprops(dum)
            
            return crossSection, dum, props[0].axis_minor_length*PIX_SIZE, props[0].axis_major_length*PIX_SIZE
        
        v0,p0 = vectors[indexDiff],points[indexDiff] # wektor i punkt wzdłuż osi naczynia w miejscu połaczenia z tętniakiem

        pointsToFit = np.array(pathAneurysmEnd[:NPOINTS]) 
        P,V = fit_3d_line(pointsToFit)
        v1,p1 = V, pathAneurysmEnd[0] # wektor wzdłuż osi łączącej środek tętniaka z naczyniem, wzdłuż szyjki i punkt centralny tętniaka
        section,segment,minAxis,maxAxis = findCrossSection(p1,v1)
        dum = section + segment*np.max(section)/2
        dum = 255/(np.max(dum)-np.min(dum))*(dum-np.min(dum))
        dum = np.asarray(dum,dtype=np.uint8)
        dum = np.reshape(dum,dum.shape + (1,))
        
        nodeName = 'CrossSection1'
        crossSectionNode1 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        slicer.util.updateVolumeFromArray(crossSectionNode1, dum)    

        self.minimalAneurysmDiameters.append(minAxis)
        self.maximalAneurysmDiameters.append(maxAxis)
        

        v2 = np.cross(v0/ np.linalg.norm(v0), v1/ np.linalg.norm(v1))
        section,segment,minAxis,maxAxis = findCrossSection(p1,v2)
        dum = section + segment*np.max(section)/2
        dum = 255/(np.max(dum)-np.min(dum))*(dum-np.min(dum))
        dum = np.asarray(dum,dtype=np.uint8)
        dum = np.reshape(dum,dum.shape + (1,))

        nodeName = 'CrossSection2'
        crossSectionNode2 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        slicer.util.updateVolumeFromArray(crossSectionNode2, dum)        

        self.minimalAneurysmDiameters.append(minAxis)
        self.maximalAneurysmDiameters.append(maxAxis)

        v3 = np.cross(v2/ np.linalg.norm(v2), v1/ np.linalg.norm(v1))
        section,segment,minAxis,maxAxis = findCrossSection(p1,v3)
        dum = section + segment*np.max(section)/2
        dum = 255/(np.max(dum)-np.min(dum))*(dum-np.min(dum))
        dum = np.asarray(dum,dtype=np.uint8)
        dum = np.reshape(dum,dum.shape + (1,))

        nodeName = 'CrossSection3'
        crossSectionNode3 = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        slicer.util.updateVolumeFromArray(crossSectionNode3, dum)        
        
        self.minimalAneurysmDiameters.append(minAxis)
        self.maximalAneurysmDiameters.append(maxAxis)



        self.distancesAlongPath = distancesAlongPath
        self.minDiameters = min_axes
        self.maxDiameters = max_axes
        self.angles = angles

        if self.ui.radioButtonDiameters.checked == True:
            results = (np.array(min_axes),np.array(distancesAlongPath))
            tableNodeMin=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
            slicer.util.updateTableFromArray(tableNodeMin, results)
            tableNodeMin.GetTable().GetColumn(0).SetName("Minimal diameter [mm]")
            tableNodeMin.GetTable().GetColumn(1).SetName("Distance [mm]")
            
            # Create plot
            plotSeriesNodeMin = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "minimal vessel axis [mm]")
            plotSeriesNodeMin.SetAndObserveTableNodeID(tableNodeMin.GetID())
            plotSeriesNodeMin.SetXColumnName("Distance [mm]")
            plotSeriesNodeMin.SetYColumnName("Minimal diameter [mm]")
            plotSeriesNodeMin.SetPlotType(plotSeriesNodeMin.PlotTypeScatter )
            plotSeriesNodeMin.SetColor(0, 0, 1.0)

            results = (np.array(max_axes),np.array(distancesAlongPath))
            tableNodeMax=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
            slicer.util.updateTableFromArray(tableNodeMax, results)
            tableNodeMax.GetTable().GetColumn(0).SetName("Maximal diameter [mm]")
            tableNodeMax.GetTable().GetColumn(1).SetName("Distance [mm]")
            
            # Create plot
            plotSeriesNodeMax = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "maximal vessel axis [mm]")
            plotSeriesNodeMax.SetAndObserveTableNodeID(tableNodeMax.GetID())
            plotSeriesNodeMax.SetXColumnName("Distance [mm]")
            plotSeriesNodeMax.SetYColumnName("Maximal diameter [mm]")
            plotSeriesNodeMax.SetPlotType(plotSeriesNodeMax.PlotTypeScatter )
            plotSeriesNodeMax.SetColor(0, 1.0, 0)

            # Create chart and add plot
            plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMin.GetID())
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMax.GetID())
            
            plotChartNode.SetYAxisTitle('Diameter [mm]')
            plotChartNode.SetXAxisTitle('Distance from aneurysm junction [mm]')
            
            # Show plot in layout
            slicer.modules.plots.logic().ShowChartInLayout(plotChartNode)
        else:
            results = (np.array(angles),np.array(distancesAlongPath))
            tableNodeMin=slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
            slicer.util.updateTableFromArray(tableNodeMin, results)
            tableNodeMin.GetTable().GetColumn(0).SetName("Minimal diameter [mm]")
            tableNodeMin.GetTable().GetColumn(1).SetName("Distance [mm]")
            
            # Create plot
            plotSeriesNodeMin = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", "curvature [degrees]")
            plotSeriesNodeMin.SetAndObserveTableNodeID(tableNodeMin.GetID())
            plotSeriesNodeMin.SetXColumnName("Distance [mm]")
            plotSeriesNodeMin.SetYColumnName("Minimal diameter [mm]")
            plotSeriesNodeMin.SetPlotType(plotSeriesNodeMin.PlotTypeScatter )
            plotSeriesNodeMin.SetColor(0, 0, 1.0)

            # Create chart and add plot
            plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
            plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNodeMin.GetID())
            
            plotChartNode.SetYAxisTitle('Vessel curvature [degrees]')
            plotChartNode.SetXAxisTitle('Distance from aneurysm junction [mm]')
            
            # Show plot in layout
            slicer.modules.plots.logic().ShowChartInLayout(plotChartNode)
            
            
#
# aneurysmLogic
#


class aneurysmLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return aneurysmParameterNode(super().getParameterNode())

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLSegmentationNode,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param showResult: show output volume in slice viewers
        """

        #if not inputVolume or not outputVolume:
        #    raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        #cliParams = {
        #    "InputVolume": inputVolume.GetID(),
        #    "OutputVolume": outputVolume.GetID(),
        #    "ThresholdValue": imageThreshold,
        #    "ThresholdType": "Above" if invert else "Below",
        #}
        #cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        #slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")


#
# aneurysmTest
#


class aneurysmTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_aneurysm1()

    def test_aneurysm1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("aneurysm1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = aneurysmLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
