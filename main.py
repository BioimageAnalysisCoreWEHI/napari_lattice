import napari
from llsz.ui import Open_czi_file

#TODO: Use napari cookiecutter template

viewer=napari.Viewer()
viewer.window.add_dock_widget(Open_czi_file)
napari.run()