from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QLabel, QHBoxLayout, QPushButton, QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox, \
    QVBoxLayout, QWidget, QMdiSubWindow


# class ResizableImageLabel(QLabel):
#     def __init__(self, parent=None):
#         # Replacing deprecated sip function with recommended alternative
#         # Assuming you identified where sipPyTypeDict() was used and replaced it
#         super(ResizableImageLabel, self).__init__(parent)
#         self.setScaledContents(True)  # Enable scaling to allow both enlarging and reducing the pixmap size
#         self.pixmap_original = None
#
#     def setPixmap(self, pixmap):
#         self.pixmap_original = pixmap  # Store the original pixmap
#         super().setPixmap(pixmap)  # Set the initial pixmap
#         self.setMinimumSize(0, 0)
#     def resizeEvent(self, event):
#         if self.pixmap_original:
#             # Scale pixmap to fit the current label size while maintaining the aspect ratio
#             scaled_pixmap = self.pixmap_original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
#             super().setPixmap(scaled_pixmap)
#             # Adjust minimum size based on a minimum scale factor, e.g., allow scaling down to 10% of original
#             min_width = max(1, self.pixmap_original.width() * 0.1)
#             min_height = max(1, self.pixmap_original.height() * 0.1)
#             self.setMinimumSize(int(min_width), int(min_height))
#         super().resizeEvent(event)

class AspectRatioMdiSubWindow(QMdiSubWindow):
    def __init__(self, image_label, *args, **kwargs):
        super(AspectRatioMdiSubWindow, self).__init__(*args, **kwargs)
        self.image_label = image_label
        self.aspect_ratio = 1  # Default aspect ratio

        # Connect the image label's signal to update the aspect ratio
        self.image_label.pixmapChanged.connect(self.updateAspectRatio)

    def updateAspectRatio(self, new_aspect_ratio):
        self.aspect_ratio = new_aspect_ratio
        self.updateSize()

    def updateSize(self):
        if self.aspect_ratio > 0:
            new_width = self.width()
            new_height = int(new_width / self.aspect_ratio)
            self.resize(new_width, new_height)

    def resizeEvent(self, event):
        self.updateSize()  # Ensure the window size respects the new aspect ratio
        super(AspectRatioMdiSubWindow, self).resizeEvent(event)
class ResizableImageLabel(QLabel):
    pixmapChanged = pyqtSignal(float)  # Emit new aspect ratio

    def __init__(self, parent=None):
        super(ResizableImageLabel, self).__init__(parent)
        self.setScaledContents(True)
        self.pixmap_original = None

    def setPixmap(self, pixmap):
        self.pixmap_original = pixmap
        super().setPixmap(pixmap)
        self.updateMinimumSize()
        if pixmap and pixmap.width() and pixmap.height():
            aspect_ratio = pixmap.width() / pixmap.height()
            self.pixmapChanged.emit(aspect_ratio)  # Emit signal with new aspect ratio

    def updateMinimumSize(self):
        if self.pixmap_original:
            min_width = max(1, self.pixmap_original.width() * 0.1)
            min_height = max(1, self.pixmap_original.height() * 0.1)
            self.setMinimumSize(int(min_width), int(min_height))

    def resizeEvent(self, event):
        if self.pixmap_original:
            scaled_pixmap = self.pixmap_original.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            super().setPixmap(scaled_pixmap)
        super().resizeEvent(event)
class JobQueueItem(QWidget):
    def __init__(self, job_name, job_data, parent=None):
        super().__init__(parent)
        self.job_name = job_name
        self.job_data = job_data

        layout = QHBoxLayout(self)
        self.label = QLabel(job_name)
        self.status_label = QLabel("Batched")
        layout.addWidget(self.label)
        layout.addWidget(self.status_label)
        self.setLayout(layout)
    def markStatus(self, status):
        self.status_label.setText(str(status))

    def markRunning(self):
        self.status_label.setText("Running")
        self.job_data['is_running'] = True

    def markComplete(self):
        self.status_label.setText("Completed")
        self.job_data['is_running'] = False



class JobDetailPopup(QWidget):
    def __init__(self, job_data, parent=None):
        super(JobDetailPopup, self).__init__(parent)
        self.job_data = job_data
        self.setupUI()

    def setupUI(self):
        layout = QVBoxLayout(self)
        self.setLayout(layout)

        for key, value in self.job_data.items():
            if isinstance(value, bool):
                widget = QCheckBox(key)
                widget.setChecked(value)
            elif isinstance(value, int):
                widget = QSpinBox()
                widget.setValue(value)
            elif isinstance(value, float):
                widget = QDoubleSpinBox()
                widget.setValue(value)
            else:
                widget = QLineEdit(str(value))

            layout.addWidget(widget)

        # Adding Save and Cancel buttons
        btnSave = QPushButton("Save")
        btnCancel = QPushButton("Cancel")
        btnSave.clicked.connect(self.saveChanges)
        btnCancel.clicked.connect(self.close)

        buttonLayout = QHBoxLayout()
        buttonLayout.addWidget(btnSave)
        buttonLayout.addWidget(btnCancel)
        layout.addLayout(buttonLayout)

    def saveChanges(self):
        # Implement logic to update job data
        self.close()
