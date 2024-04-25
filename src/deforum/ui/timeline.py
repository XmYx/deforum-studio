from PyQt6.QtGui import QPixmap, QImage, QDrag
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QScrollArea, QStyleOptionFrame, QFrame
from PyQt6.QtCore import Qt, QMimeData, pyqtSignal, QPoint


class TimeLineHandle(QLabel):
    frameChanged = pyqtSignal(QPixmap)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("|")
        self.setFixedWidth(10)
        self.setStyleSheet("color: red; font-size: 24px;")
        self.setMouseTracking(True)
        self.is_dragging = False

    def mousePressEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.is_dragging = True

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            new_x = self.parent().mapFromGlobal(event.globalPos()).x()
            closest_frame = None
            min_distance = float('inf')

            # Determine the nearest frame
            for track in self.parent().tracks:
                for frame in track.frames:
                    frame_center = frame.pos().x() + frame.width() // 2
                    distance = abs(frame_center - new_x)
                    if distance < min_distance:
                        min_distance = distance
                        closest_frame = frame_center

            # Snap to the nearest frame if found
            if closest_frame is not None:
                self.move(closest_frame - self.width() // 2, self.y())
            else:
                # Default behavior, clamp within bounds
                self.move(max(0, min(new_x, self.parent().width() - self.width())), self.y())

            self.emitCurrentFrame()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.emitCurrentFrame()

    def emitCurrentFrame(self):
        # Check which frame is under the handle
        for track in self.parent().tracks:
            for frame in track.frames:
                if frame.geometry().contains(self.pos() + QPoint(frame.width() // 2, 0)):
                    self.frameChanged.emit(frame.pixmap())
                    break
class DraggableLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

    def mousePressEvent(self, event):
        if self.pixmap() is not None:
            drag = QDrag(self)
            mimeData = QMimeData()
            drag.setMimeData(mimeData)
            drag.setPixmap(self.pixmap())
            drag.exec(Qt.DropAction.CopyAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage():
            event.acceptProposedAction()

    def dropEvent(self, event):
        if event.mimeData().hasImage():
            self.setPixmap(QPixmap.fromImage(QImage(event.mimeData().imageData())))
            event.acceptProposedAction()

class VideoTrack(QWidget):
    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.label = QLabel(name)
        self.layout.addWidget(self.label)
        self.frames = []
        self.addFrameButton = QPushButton('+ Frame')
        self.addFrameButton.clicked.connect(self.addFrame)
        self.layout.addWidget(self.addFrameButton)
        self.removeTrackButton = QPushButton('x')
        self.removeTrackButton.clicked.connect(self.removeSelf)
        self.layout.addWidget(self.removeTrackButton)

    def addFrame(self):
        frame = DraggableLabel('Empty Frame')
        self.frames.append(frame)
        self.layout.insertWidget(self.layout.count() - 2, frame)

    def removeSelf(self):
        self.setParent(None)
        self.deleteLater()

    def addImageToFrame(self, image):
        if not self.frames:
            self.addFrame()  # Ensure there's at least one frame
        self.frames[-1].setPixmap(image)

class TimelineWidget(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.widget.setLayout(self.layout)
        self.setWidget(self.widget)
        self.setWidgetResizable(True)
        self.tracks = []
        self.handle = TimeLineHandle(self.widget)
        self.handle.move(50, 10)  # Start position for the handle
        self.addTrack("VIDEO 1")

    def addTrack(self, name):
        track = VideoTrack(name)
        self.tracks.append(track)
        self.layout.addWidget(track)

    def removeTrack(self, track):
        self.tracks.remove(track)
        track.deleteLater()