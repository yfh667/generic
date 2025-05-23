import time
import vtk
t0 = time.time()

filename = "test.tif"

reader = vtk.vtkTIFFReader()
reader.SetFileName(filename)
reader.Update()
image = reader.GetOutput()
print("Time to load from disk into VTK", time.time()-t0); t0=time.time()

# mapper = vtk.vtkSmartVolumeMapper()
mapper = vtk.vtkGPUVolumeRayCastMapper()
# mapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()
# mapper = vtk.vtkFixedPointVolumeRayCastMapper()

mapper.SetInputData(image)
opacityTransferFunction = vtk.vtkPiecewiseFunction()
opacityTransferFunction.AddPoint(20, 0.0)
opacityTransferFunction.AddPoint(255, 0.2)


colorTransferFunction = vtk.vtkColorTransferFunction()
colorTransferFunction.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
colorTransferFunction.AddRGBPoint(64.0, 1.0, 0.0, 0.0)
colorTransferFunction.AddRGBPoint(128.0, 0.0, 0.0, 1.0)
colorTransferFunction.AddRGBPoint(192.0, 0.0, 1.0, 0.0)
colorTransferFunction.AddRGBPoint(255.0, 0.0, 0.2, 0.0)

# The property describes how the data will look.
volumeProperty = vtk.vtkVolumeProperty()
volumeProperty.SetColor(colorTransferFunction)
volumeProperty.SetScalarOpacity(opacityTransferFunction)
volumeProperty.ShadeOn()
volumeProperty.SetInterpolationTypeToLinear()


# The volume holds the mapper and the property and
# can be used to position/orient the volume.
volume = vtk.vtkVolume()
volume.GetProperty().SetInterpolationType(1)
volume.GetProperty().SetShade(True)
volume.SetMapper(mapper)
volume.SetProperty(volumeProperty)

print("Time to do all VTK stuff except render", time.time()-t0); t0=time.time()

ren = vtk.vtkRenderer()
ren.AddVolume(volume)
ren.SetBackground(1,1,1)

renWin = vtk.vtkRenderWindow()
renWin.AddRenderer(ren)
renWin.SetSize(800, 800)

iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)
iren.Render()
iren.Start()
print("Time to render (press q as soon as volume is visible)", time.time()-t0); t0=time.time()