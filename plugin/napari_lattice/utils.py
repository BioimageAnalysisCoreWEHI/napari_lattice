from napari.viewer import current_viewer, Viewer

def get_viewer() -> Viewer:
    viewer = current_viewer()
    if viewer is None:
        raise Exception("No viewer present!")
    return viewer
