import napari
from llsz.ui import Open_czi_file

viewer=napari.Viewer()
viewer.window.add_dock_widget(Open_czi_file)
napari.run()