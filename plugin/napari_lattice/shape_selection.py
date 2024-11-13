from __future__ import annotations
from napari.utils.events import EventEmitter, Event
from napari.layers import Shapes

class ShapeLayerChangedEvent(Event):
    """
    Event triggered when the shape layer selection changes.
    """

class ShapeSelectionListener(EventEmitter):
    """
    Manages shape selection events for a given Shapes layer.

    Examples:
        This example code will open the viewer with an empty shape layer.
        Any selection changes to that layer will trigger a notification popup.
        >>> from napari import Viewer
        >>> from napari.layers import Shapes
        >>> viewer = Viewer()
        >>> shapes = viewer.add_shapes()
        >>> shape_selection = ShapeSelection(shapes)
        >>> shape_selection.connect(lambda event: print("Shape selection changed!"))
    """
    last_selection: set[int]
    layer: Shapes

    def __init__(self, layer) -> None:
        """
        Initializes the ShapeSelection with the given Shapes layer.

        Parameters:
            layer: The Shapes layer to listen to.
        """
        super().__init__(source=layer, event_class=ShapeLayerChangedEvent, type_name="shape_layer_selection_changed")
        self.layer = layer
        self.last_selection = set()
        layer.events.highlight.connect(self._on_highlight)

    def _on_highlight(self, event) -> None:
        new_selection = self.layer.selected_data
        if new_selection != self.last_selection:
            self()
        self.last_selection = set(new_selection)

def test_script():
    """
    Demo for testing the event behaviour.
    """
    from napari import run, Viewer
    from napari.utils.notifications import show_info
    viewer = Viewer()
    shapes = viewer.add_shapes()
    event = ShapeSelectionListener(shapes)
    event.connect(lambda x: show_info("Shape selection changed!"))
    run()

if __name__ == "__main__":
    test_script()
