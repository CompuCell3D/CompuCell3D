#!/usr/bin/env python

import vtk
print(vtk.vtkVersion.GetVTKSourceVersion())

def main():
    colors = vtk.vtkNamedColors()

    # Create an image
    source1 = vtk.vtkImageCanvasSource2D()
    source1.SetScalarTypeToUnsignedChar()
    source1.SetNumberOfScalarComponents(3)
    source1.SetExtent(0, 100, 0, 100, 0, 0)
    source1.SetDrawColor(0,0,0,1)
    source1.FillBox(0, 100, 0, 100)
    source1.SetDrawColor(255,0,0,1)
    source1.FillBox(10, 20, 10, 20)
    source1.FillBox(40, 50, 20, 30)
    source1.Update()

    # Convert the image to a polydata
    imageDataGeometryFilter = vtk.vtkImageDataGeometryFilter()
    # imageDataGeometryFilter = vtk.vtkStructuredPointsGeometryFilter()
    imageDataGeometryFilter.SetInputConnection(source1.GetOutputPort())
    imageDataGeometryFilter.Update()

    # p2c = vtk.vtkPointDataToCellData()
    # p2c.SetInputConnection(imageDataGeometryFilter.GetOutputPort())
    # stripper = vtk.vtkStripper()
    # stripper.SetInputConnection(p2c.GetOutputPort())
    # # stripper.PassCellDataAsFieldDataOn()

    p2c = vtk.vtkPointDataToCellData()
    p2c.PassPointDataOn()
    # p2c.SetInputConnection(source1.GetOutputPort())

    # geom_filter = vtk.vtkGeometryFilter()
    # geom_filter.SetInputConnection(p2c.GetOutputPort())

    # p2c.SetInputConnection(imageDataGeometryFilter.GetOutputPort())
    # stripper = vtk.vtkStripper()
    # stripper.SetInputConnection(p2c.GetOutputPort())
    # stripper.PassCellDataAsFieldDataOn()



    sphere = vtk.vtkSphereSource()
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    # mapper = vtk.vtkPolyDataMapper2D()
    # mapper.SetScalarModeToUseCellData()
    mapper.SetInputConnection(imageDataGeometryFilter.GetOutputPort())
    # mapper.SetInputConnection(geom_filter.GetOutputPort())
    # mapper.SetInputConnection(p2c.GetOutputPort())
    # mapper.SetInputConnection(sphere.GetOutputPort())

    actor = vtk.vtkActor()
    # actor.GetProperty().BackfaceCullingOn()
    # actor.GetProperty().LightingOff()
    # actor.GetProperty().ShadingOn()

    # actor.GetProperty().SetAmbient(1.0)
    # actor.GetProperty().SetDiffuse(0.0)
    # actor.GetProperty().SetSpecular(0.0)
    # actor.GetProperty().SetInterpolationToGouraud()
    # actor.GetProperty().SetInterpolationToFlat()

    # interp = actor.GetProperty().GetInterpolation()
    #
    #
    # print('interp ',interp )
    # actor.GetProperty().SetInterpolationToFlat()
    # actor.GetProperty().SetInterpolation(0)
    # interp = actor.GetProperty().GetInterpolation()
    # print('interp ', interp)
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToFlat()
    actor.GetProperty().BackfaceCullingOn()
    actor.GetProperty().LightingOff()
    actor.GetProperty().ShadingOn()

    actor.GetProperty().SetAmbient(1.0)
    actor.GetProperty().SetDiffuse(0.0)
    actor.GetProperty().SetSpecular(0.0)
    actor.GetProperty().SetInterpolationToGouraud()
    actor.GetProperty().SetInterpolationToFlat()




    # Visualization
    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(actor)
    renderer.SetBackground(colors.GetColor3d("white")) # Background color white
    renderWindow.Render()
    renderWindowInteractor.Start()

if __name__ == '__main__':
    main()