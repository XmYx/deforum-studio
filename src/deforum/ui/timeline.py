from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem, QVBoxLayout, \
    QPushButton, QWidget, QGraphicsRectItem, QGraphicsTextItem, QLabel, QGraphicsPixmapItem, QSlider
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, pyqtSlot, QObject
from PyQt6.QtGui import QPen, QBrush, QColor, QPainterPath, QPixmap, QTransform


class TimelineHandle(QGraphicsItem):
    def __init__(self, height, offset_x=0):
        super().__init__()
        self.offset_x = offset_x
        self.height = height
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(1000)
        self.isHovered = False  # Track hover state
        self.isDragging = False  # Track drag state

        self.setAcceptHoverEvents(True)  # Accept hover events to change color on hover

    def boundingRect(self):
        # Increased bounding rect size for easier interaction
        return QRectF(self.offset_x - 15, 0, 30, self.height)

    def paint(self, painter, option, widget=None):
        if self.isHovered or self.isDragging:
            painter.setPen(QPen(QColor("orange"), 2))  # Orange when hovered or dragged
        else:
            painter.setPen(QPen(Qt.GlobalColor.red, 2))  # Red otherwise
        painter.drawLine(QPointF(self.offset_x + 1, 0), QPointF(self.offset_x + 1, self.height))

    def hoverEnterEvent(self, event):
        self.isHovered = True
        self.update()  # Trigger a repaint when the mouse hovers

    def hoverLeaveEvent(self, event):
        self.isHovered = False
        self.update()  # Trigger a repaint when the mouse leaves

    def mousePressEvent(self, event):
        self.isDragging = True  # Set dragging state to true
        self.update()  # Trigger a repaint on drag start
        QGraphicsItem.mousePressEvent(self, event)

    def mouseReleaseEvent(self, event):
        self.isDragging = False  # Reset dragging state
        self.update()  # Trigger a repaint on drag end
        QGraphicsItem.mouseReleaseEvent(self, event)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            adjusted_x = max(self.offset_x, min(value.x(), self.scene().width() - self.boundingRect().width() + 15))
            closest_frame_x = round((adjusted_x - self.offset_x) / 105) * 105 + self.offset_x
            new_pos = QPointF(closest_frame_x, self.y())
            self.updateFrameReadout((closest_frame_x - self.offset_x) / 105)
            return new_pos
        return super().itemChange(change, value)

    def updateFrameReadout(self, frame_number):
        self.scene().views()[0].parentWidget().frameReadout.setText(f"Frame: {int(frame_number)}")


class ResizingHandle(QGraphicsRectItem):
    def __init__(self, parent=None):
        super().__init__(0, 0, 10, 50, parent)
        self.setBrush(QBrush(Qt.GlobalColor.gray))
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)  # Enable hover events

    def paint(self, painter, option, widget=None):
        painter.setBrush(QBrush(QColor(100, 100, 255)) if self.isUnderMouse() or self.parentItem().isResizing else QBrush(Qt.GlobalColor.gray))
        painter.drawRect(self.rect())

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor(100, 100, 255)))
        self.update()

    def hoverLeaveEvent(self, event):
        if not self.parentItem().isResizing:
            self.setBrush(QBrush(Qt.GlobalColor.gray))
        self.update()


class Frame(QGraphicsRectItem):
    def __init__(self, pixmap, frame_number, parent=None):
        super().__init__(0, 0, 100, 50, parent)
        self.pixmap_item = QGraphicsPixmapItem(pixmap.scaled(100, 50), self)  # Add a pixmap item to this item
        self.setBrush(QBrush(Qt.GlobalColor.transparent))  # Remove the background color

        self.frame_number = frame_number
        self.parent = parent
        self.updatePosition()

    def updatePosition(self):
        track = self.parent  # Get the parent track
        if track:
            # Calculate the position based on the frame number and track width
            x = track.label_total_width + (self.frame_number * 105)
            self.setPos(x, track.y)



class VideoObject:
    def __init__(self, track):
        self.frames = []  # Store individual frames (QPixmap objects)
        self.totalFrames = 0  # Total number of frames
        self.effects = []  # List to store effects applied to the video
        self.track = track

    def add_frame(self, frame):
        self.frames.append(Frame(frame, self.totalFrames, self.track))
        self.totalFrames += 1

    def apply_effect(self, effect):
        self.effects.append(effect)


class VideoTrackSignals(QObject):
    trackClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent


class VideoTrack(QGraphicsItem):
    trackClicked = pyqtSignal()

    def __init__(self, y, parent=None):
        super().__init__(parent)
        self.signals = VideoTrackSignals(self)
        self.y = y
        self.original_width = 700  # Store the original width
        self.width = self.original_width  # Use original width as the initial width
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)  # Enable hover events for the track

        # Dimensions and positions
        self.label_total_width = 100  # Width reserved for label and button
        label_width = 80
        button_width = 20
        button_offset = label_width

        # Label
        self.label = QGraphicsTextItem(f"Track {y // 55 + 1}", self.scene())
        self.label.setPos(10, y + 15)  # Assume relative to scene for now

        # Button
        self.delete_button = QGraphicsRectItem(button_offset, y, button_width, 50, self.scene())
        self.delete_button.setBrush(QBrush(QColor(255, 0, 0)))

        # Resizing handle
        self.isResizing = False
        self.resizingHandle = ResizingHandle(self)
        self.resizingHandle.setPos(self.width - 5, self.y)

        # Initialize items in the scene (if not already added in the scene setup)
        if self.scene():
            self.scene().addItem(self.label)
            self.scene().addItem(self.delete_button)

    def setHorizontalScale(self, scale):
        """Scale only the video track area."""
        self.prepareGeometryChange()
        self.width = self.original_width * scale
        self.update()

        # Keep label and button at their positions
        self.label.setPos(10, self.y + 15)
        self.delete_button.setRect(self.label_total_width, self.y, self.delete_button.rect().width(), 50)

    # Other methods...

    def paint(self, painter, option, widget=None):
        # Draw the track
        painter.setBrush(QBrush(QColor(200, 200, 250) if self.isSelected() else Qt.GlobalColor.lightGray))
        painter.drawRect(QRectF(self.label_total_width, self.y, self.width, 50))

        # Ensure label and delete button are positioned correctly
        self.label.setPos(self.x() + 10, self.y + 15)
        self.delete_button.setPos(self.x() + self.label_total_width, self.y)

        # Update resizing handle position
        self.resizingHandle.setPos(self.label_total_width + self.width - 5, self.y)


    def add_image(self, pixmap):
        if not self.currentVideoObject:
            self.currentVideoObject = VideoObject(self)
            self.videoObjects.append(self.currentVideoObject)
            if len(self.videoObjects) > 1:  # If new video object and not the first one
                self.setPos(self.boundingRect().left(), 0)  # Reset track's position
        self.currentVideoObject.add_frame(pixmap)
        self.width = max(self.width,
                         self.currentVideoObject.totalFrames * 105)  # Extend the track if new frame would exceed its current length
        self.update()  # Update track's representation

    def boundingRect(self):
        return QRectF(self.label_total_width, self.y, self.width, 50)

    def paint(self, painter, option, widget=None):
        # Draw the track
        painter.setBrush(QBrush(QColor(200, 200, 250) if self.isSelected() else Qt.GlobalColor.lightGray))
        painter.drawRect(QRectF(self.label_total_width, self.y, self.width, 50))

        # Position label and delete button in scene coordinates

        self.label.setPos(self.x() + 10, self.y + 15)
        self.delete_button.setPos(self.x() + self.label_total_width, self.y)

        # Update resizing handle position
        self.resizingHandle.setPos(self.label_total_width + self.width - 5, self.y)


    def mousePressEvent(self, event):
        self.signals.trackClicked.emit()
        if self.resizingHandle.isUnderMouse():
            self.isResizing = True
            self.resizingHandle.setBrush(QBrush(QColor(100, 100, 255)))
        else:
            # Emit the signal when the track is clicked
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isResizing:
            new_width = event.pos().x() - self.label_total_width
            self.width = max(50, min(new_width, self.scene().width() - self.label_total_width))
            self.update()  # Update the entire track including the handle
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.isResizing:
            self.width = round(self.width / 105) * 105  # Snap to nearest frame on release
            self.isResizing = False
            self.resizingHandle.setBrush(QBrush(Qt.GlobalColor.gray))
            self.update()  # Final update to confirm all visuals are correct
        else:
            super().mouseReleaseEvent(event)


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.layout.addWidget(self.view)

        self.frameReadout = QLabel("Frame: 0", self)
        self.layout.addWidget(self.frameReadout)

        # Zoom Slider
        self.zoomSlider = QSlider(Qt.Orientation.Horizontal, self)
        self.zoomSlider.setMinimum(1)
        self.zoomSlider.setMaximum(100)
        self.zoomSlider.setValue(10)  # Initial zoom level
        self.zoomSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.zoomSlider.setTickInterval(10)
        self.zoomSlider.valueChanged.connect(self.zoomTimeline)
        self.layout.addWidget(self.zoomSlider)

        self.handle = TimelineHandle(self.view.height(), offset_x=100)
        self.scene.addItem(self.handle)

        self.addTrackButton = QPushButton("Add Track", self)
        self.addTrackButton.clicked.connect(self.addTrack)
        self.layout.addWidget(self.addTrackButton)

        self.tracks = []
        self.addTrack()  # Initial track
        self.selectedTrack = None

    @pyqtSlot()
    def selectTrack(self):
        sender_track = self.sender().parent  # Get the track that emitted the signal
        if sender_track in self.tracks:
            self.selectedTrack = sender_track

    def add_image_to_track(self, pixmap):
        if self.selectedTrack:
            self.selectedTrack.add_image(pixmap)
            self.handle.setPos(self.handle.x() + 105, 0)  # advance the handle
        else:
            print("No track selected!")

    def addTrack(self, name=""):
        y = len(self.tracks) * 55 + 30
        track = VideoTrack(y)
        track.signals.trackClicked.connect(self.selectTrack)
        self.scene.addItem(track)
        self.tracks.append(track)
        self.handle.setPos(100, 0)

    def updateHandleHeight(self):
        total_height = len(self.tracks) * 55
        self.handle.height = total_height + 30
        self.handle.update()

    def removeTrack(self, track):
        self.tracks.remove(track)
        track.deleteLater()
        self.updateHandleHeight()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        self.handle.height = self.view.height()
        self.handle.update()

    def zoomTimeline(self, value):
        scale_factor = value / 10.0  # Adjust the scale factor as needed
        for track in self.tracks:
            track.setHorizontalScale(scale_factor)  # Apply scale only to tracks horizontally




if __name__ == "__main__":
    app = QApplication([])
    timelineWidget = TimelineWidget()
    timelineWidget.show()
    app.exec()
