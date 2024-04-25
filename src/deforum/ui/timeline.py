from PyQt6.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsItem, QVBoxLayout, \
    QPushButton, QWidget, QGraphicsRectItem, QGraphicsTextItem
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPen, QBrush, QColor, QPainterPath

class TimelineHandle(QGraphicsItem):
    def __init__(self, height):
        super().__init__()
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.height = height

    def boundingRect(self):
        return QRectF(0, 0, 2, self.height)

    def paint(self, painter, option, widget=None):
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        painter.drawLine(QPointF(1, 0), QPointF(1, self.height))

class VideoFrame(QGraphicsItem):
    def __init__(self, width, height, parent=None):
        super().__init__(parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.width = width
        self.height = height
        self.original_y = None  # To keep track of the original y-coordinate

    def boundingRect(self):
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter, option, widget=None):
        painter.setBrush(QBrush(QColor(100, 100, 250, 120)))
        painter.drawRect(self.boundingRect())

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            if self.original_y is None:
                self.original_y = value.y()

            new_pos = value
            scene = self.scene()

            # Restrict vertical movement unless there are multiple tracks
            if len(scene.views()[0].parentWidget().tracks) <= 1:
                new_pos.setY(self.original_y)
            else:
                # Allow snapping to the nearest track
                nearest_track = None
                min_distance = float('inf')
                for track in scene.views()[0].parentWidget().tracks:
                    distance = abs(track.y + track.boundingRect().height()/2 - value.y())
                    if distance < min_distance:
                        min_distance = distance
                        nearest_track = track

                if nearest_track:
                    new_pos.setY(nearest_track.y)

            # Restrict horizontal movement to within the scene bounds
            new_pos.setX(max(0, min(new_pos.x(), scene.width() - self.width)))

            return new_pos
        return super().itemChange(change, value)


class VideoTrack(QGraphicsItem):
    def __init__(self, y, parent=None):
        super().__init__(parent)
        self.y = y
        self.frames = []
        self.label = QGraphicsTextItem(f"Track {y // 55 + 1}", self)
        self.label.setPos(0, y)

        # Add a delete button as a child item
        self.delete_button = QGraphicsRectItem(0, y, 20, 20, self)
        self.delete_button.setBrush(QBrush(QColor(255, 0, 0)))
        self.label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)

    def boundingRect(self):
        return QRectF(0, self.y, 1000, 50)

    def paint(self, painter, option, widget=None):
        path = QPainterPath()
        path.addRect(self.boundingRect())
        painter.setPen(QPen(Qt.GlobalColor.black, 1))
        painter.setBrush(QBrush(Qt.GlobalColor.lightGray))
        painter.drawPath(path)

    def addFrame(self):
        frame = VideoFrame(100, 50)
        frame.setPos(len(self.frames) * 105, self.y)
        self.scene().addItem(frame)
        self.frames.append(frame)

class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.view = QGraphicsView(self)
        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)
        self.layout.addWidget(self.view)

        self.handle = TimelineHandle(200)  # Dynamic height based on tracks
        self.scene.addItem(self.handle)

        self.addTrackButton = QPushButton("Add Track", self)
        self.addTrackButton.clicked.connect(self.addTrack)
        self.layout.addWidget(self.addTrackButton)

        self.tracks = []
        self.addTrack("VIDEO 1")  # Initial track

    def addTrack(self, name=""):
        y = len(self.tracks) * 55
        track = VideoTrack(y)
        self.scene.addItem(track)
        self.tracks.append(track)
        track.addFrame()  # Adding a default frame
        self.handle.setPos(50, 0)  # Set handle to start position
        self.updateHandleHeight()

    def updateHandleHeight(self):
        self.handle.height = len(self.tracks) * 55
        self.handle.update()

    def removeTrack(self, track):
        self.tracks.remove(track)
        track.deleteLater()
        self.updateHandleHeight()