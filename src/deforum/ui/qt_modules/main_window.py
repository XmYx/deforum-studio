import copy
import datetime
import os
import sys
import json

import numpy as np

from PyQt6.QtGui import QImage, QAction
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import QApplication, QTabWidget, QWidget, QVBoxLayout, QDockWidget, QSlider, QLabel, QMdiArea, \
    QPushButton, QComboBox, QFileDialog, QSpinBox, QLineEdit, QCheckBox, QTextEdit, QHBoxLayout, QListWidget, \
    QDoubleSpinBox, QListWidgetItem, QMessageBox
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QUrl

from deforum import logger
from deforum.ui.qt_helpers.qt_image import npArrayToQPixmap
from deforum.ui.qt_modules.backend_thread import BackendThread
from deforum.ui.qt_modules.core import DeforumCore
from deforum.ui.qt_modules.custom_ui import ResizableImageLabel, JobDetailPopup, JobQueueItem
from deforum.ui.qt_modules.timeline import TimelineWidget



class MainWindow(DeforumCore):
    def __init__(self):
        super().__init__()
        self.player = QMediaPlayer()  # Create a media player object
        self.setWindowTitle("Deforum Engine")
        self.setupDynamicUI()
        self.currentTrack = None  # Store the currently selected track
        self.presets_folder = os.path.join(os.path.expanduser('~'), 'deforum', 'presets')
        self.loadPresetsDropdown()
        self.presetsDropdown.activated.connect(self.loadPresetsDropdown)
        self.setupVideoPlayer()
        self.tileMdiSubWindows()
        self.timelineDock.hide()
        self.setupBatchControls()
        self.job_queue = []  # List to keep jobs to be processed
        self.current_job = None  # To keep track of the currently running job

    def onJobDoubleClicked(self, item):
        widget = self.jobQueueList.itemWidget(item)
        if widget:
            # Create a JobDetailPopup and add it to the MDI area
            popup = JobDetailPopup(widget.job_data)
            subWindow = self.mdiArea.addSubWindow(popup)  # Add popup to MDI area
            popup.setWindowTitle("Job Details - " + widget.job_name)
            subWindow.show()

    def setupBatchControls(self):
        self.batchControlTab = QWidget()
        layout = QVBoxLayout()
        self.batchControlTab.setLayout(layout)
        self.jobQueueList = QListWidget()
        self.jobQueueList.itemDoubleClicked.connect(self.onJobDoubleClicked)

        # Buttons for batch control
        loadFilesButton = QPushButton("Load Batch Files")
        startButton = QPushButton("Start Batch")
        cancelButton = QPushButton("Cancel Batch")

        # Connect buttons to their respective slots (functions)
        startButton.clicked.connect(self.startBatchProcess)
        cancelButton.clicked.connect(self.stopBatchProcess)
        loadFilesButton.clicked.connect(self.loadBatchFiles)
        layout.addWidget(loadFilesButton)
        layout.addWidget(startButton)
        layout.addWidget(cancelButton)
        layout.addWidget(self.jobQueueList)

        # Add the batch control tab to the main tab widget
        self.tabWidget.addTab(self.batchControlTab, "Batch Control")

    def loadBatchFiles(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        file_dialog.setNameFilter("Text Files (*.txt)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            for file_path in selected_files:
                self.createJobFromConfig(file_path)
    def createJobFromConfig(self, file_path):
        try:
            with open(file_path, 'r') as file:
                job_data = json.load(file)
                batch_name = os.path.basename(file_path)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                job_name = f"{batch_name} {timestamp}"
                self.addCurrentParamsAsJob(job_name, job_data)
        except json.JSONDecodeError as e:
            print(f"Error loading {file_path}: {e}")
            QMessageBox.warning(self, "Loading Error", f"Failed to load {file_path}: {e}")
    def addCurrentParamsAsJob(self, job_name=None, job_params=None):

        if job_name == None:
            job_name = f"{self.params.get('batch_name')}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            job_params = self.params
        # Adding current params as a new job
        # job_description = json.dumps(self.params, indent=4)
        item = QListWidgetItem(self.jobQueueList)
        widget = JobQueueItem(job_name, job_params)
        item.setSizeHint(widget.sizeHint())
        self.jobQueueList.addItem(item)
        self.jobQueueList.setItemWidget(item, widget)
        self.job_queue.append(widget)  # Append job widget to the queue
    def loadPresetsDropdown(self):
        current_text = self.presetsDropdown.currentText()  # Remember current text
        if not os.path.exists(self.presets_folder):
            os.makedirs(self.presets_folder)

        preset_files = [f for f in os.listdir(self.presets_folder) if f.endswith('.txt')]
        self.presetsDropdown.clear()
        self.presetsDropdown.addItems(preset_files)

        # Re-select the current text if it exists in the new list
        if current_text in preset_files:
            index = self.presetsDropdown.findText(current_text)
            self.presetsDropdown.setCurrentIndex(index)



    def loadPreset(self):
        selected_preset = self.presetsDropdown.currentText()
        preset_path = os.path.join(self.presets_folder, selected_preset)
        try:
            with open(preset_path, 'r') as file:
                config = json.load(file)
                for key, value in config.items():
                    if key in self.params:
                        self.params[key] = value
            self.updateUIFromParams()

        except:
            pass

    def savePreset(self):
        preset_name, _ = QFileDialog.getSaveFileName(self, 'Save Preset', self.presets_folder, 'Text Files (*.txt)')
        if preset_name:
            preset_name = os.path.splitext(os.path.basename(preset_name))[0] + '.txt'
            preset_path = os.path.join(self.presets_folder, preset_name)

            with open(preset_path, 'w') as file:
                json.dump(self.params, file, indent=4)

            self.loadPresetsDropdown()
    def setupDynamicUI(self):
        self.toolbar = self.addToolBar('Main Toolbar')
        self.renderButton = QAction('Render', self)
        self.stopRenderButton = QAction('Stop Render', self)

        self.toolbar.addAction(self.renderButton)
        self.toolbar.addAction(self.stopRenderButton)
        # Add presets dropdown to the toolbar
        self.presetsDropdown = QComboBox()
        self.presetsDropdown.currentIndexChanged.connect(self.loadPreset)
        self.toolbar.addWidget(self.presetsDropdown)

        # Add actions to load and save presets
        self.loadPresetAction = QAction('Load Preset', self)
        self.loadPresetAction.triggered.connect(self.loadPreset)
        self.toolbar.addAction(self.loadPresetAction)
        self.tileSubWindowsAction = QAction('Tile Subwindows', self)
        self.tileSubWindowsAction.triggered.connect(self.tileMdiSubWindows)
        self.toolbar.addAction(self.tileSubWindowsAction)
        self.savePresetAction = QAction('Save Preset', self)
        self.savePresetAction.triggered.connect(self.savePreset)
        self.toolbar.addAction(self.savePresetAction)

        self.addJobButton = QAction('Add to Batch', self)
        self.addJobButton.triggered.connect(self.addCurrentParamsAsJob)
        self.toolbar.addAction(self.addJobButton)

        self.renderButton.triggered.connect(self.startBackendProcess)
        self.stopRenderButton.triggered.connect(self.stopBackendProcess)
        # Initialize the tabbed control layout docked on the left
        self.controlsDock = self.addDock("Controls", Qt.DockWidgetArea.LeftDockWidgetArea)
        self.tabWidget = QTabWidget()
        self.controlsDock.setWidget(self.tabWidget)

        # Load UI configuration from JSON
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(curr_folder, 'ui.json')
        with open(json_path, 'r') as file:
            config = json.load(file)

        for category, settings in config.items():
            tab = QWidget()
            layout = QVBoxLayout()
            tab.setLayout(layout)

            for setting, params in settings.items():
                if params['widget_type'] == 'number':
                    self.createSpinBox(params['label'], layout, params['min'], params['max'], 1, params['default'], setting)
                elif params['widget_type'] == 'float':
                    self.createDoubleSpinBox(params['label'], layout, params['min'], params['max'], 0.01, params['default'], setting)

                elif params['widget_type'] == 'dropdown':
                    self.createComboBox(params['label'], layout, [str(param) for param in params['options']], setting)
                elif params['widget_type'] == 'text input':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'text box':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'slider':
                    self.createSlider(params['label'], layout, params['min'], params['max'], params['default'], setting)
                elif params['widget_type'] == 'checkbox':
                    self.createCheckBox(params['label'], layout, bool(params['default']), setting)
                elif params['widget_type'] == 'file_input':
                    # Add file input handler if needed
                    pass

            self.tabWidget.addTab(tab, category)
        # self.createPushButton('Render', layout, self.startBackendProcess)

        # Create preview area
        self.setupPreviewArea()

        # self.timeline = TimelineWidget()
        # self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timeline)
        # Create the timeline dock widget
        self.timelineDock = QDockWidget("Timeline", self)
        self.timelineDock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)

        # Create and set the timeline widget inside the dock
        self.timelineWidget = TimelineWidget()
        self.timelineDock.setWidget(self.timelineWidget)

        # Add the dock widget to the main window
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timelineDock)

    def setupPreviewArea(self):
        # Setting up an MDI area for previews
        self.mdiArea = QMdiArea()
        self.setCentralWidget(self.mdiArea)
        self.previewSubWindow = self.mdiArea.addSubWindow(QWidget())
        self.previewLabel = ResizableImageLabel()
        self.previewLabel.setScaledContents(True)
        self.previewLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previewSubWindow.setWidget(self.previewLabel)
        self.previewSubWindow.show()

    def setupVideoPlayer(self):
        self.videoSubWindow = self.mdiArea.addSubWindow(QWidget())
        self.videoSubWindow.setWindowTitle("Video Player")
        # Setting up the video player within its subwindow
        self.videoWidget = QVideoWidget()
        self.audioOutput = QAudioOutput()  # Create an audio output
        self.player.setAudioOutput(self.audioOutput)  # Set audio output for the player
        self.player.setVideoOutput(self.videoWidget)  # Set video output

        # Adding playback controls
        self.playButton = QPushButton("Play")
        self.pauseButton = QPushButton("Pause")
        self.stopButton = QPushButton("Stop")
        self.volumeSlider = QSlider(Qt.Orientation.Horizontal)
        self.volumeSlider.setMinimum(0)
        self.volumeSlider.setMaximum(100)
        self.volumeSlider.setValue(50)  # Default volume level at 50%
        self.volumeSlider.setTickInterval(10)  # Optional: makes it easier to slide in intervals
        self.audioOutput.setVolume(0.5)  # Set the initial volume to 50%

        # Connect control buttons and slider
        self.playButton.clicked.connect(self.player.play)
        self.pauseButton.clicked.connect(self.player.pause)
        self.stopButton.clicked.connect(self.player.stop)
        self.volumeSlider.valueChanged.connect(lambda value: self.audioOutput.setVolume(value / 100))
        self.videoSlider = QSlider(Qt.Orientation.Horizontal)
        self.videoSlider.sliderMoved.connect(self.setPosition)
        self.player.positionChanged.connect(self.updatePosition)
        self.player.durationChanged.connect(self.updateDuration)
        # Layout for the video player controls
        controlsLayout = QHBoxLayout()
        controlsLayout.addWidget(self.playButton)
        controlsLayout.addWidget(self.pauseButton)
        controlsLayout.addWidget(self.stopButton)
        controlsLayout.addWidget(QLabel("Volume:"))
        controlsLayout.addWidget(self.volumeSlider)

        # Main layout for the video subwindow
        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.videoWidget)
        mainLayout.addWidget(self.videoSlider)

        mainLayout.addLayout(controlsLayout)

        self.videoSubWindow.widget().setLayout(mainLayout)
    def tileMdiSubWindows(self):
        """Resize and evenly distribute all subwindows in the MDI area."""
        # Optionally, set a uniform size for all subwindows
        subwindows = self.mdiArea.subWindowList()
        if not subwindows:
            return

        # Calculate the optimal size for subwindows based on the number of windows
        areaSize = self.mdiArea.size()
        numWindows = len(subwindows)
        width = int(areaSize.width() / numWindows ** 0.5)
        height = int(areaSize.height() / numWindows ** 0.5)

        # Resize each subwindow (this step is optional)
        for window in subwindows:
            window.resize(width, height)

        # Tile the subwindows
        self.mdiArea.tileSubWindows()
    def setPosition(self, position):
        self.player.setPosition(position)

    def updatePosition(self, position):
        self.videoSlider.setValue(position)

    def updateDuration(self, duration):
        self.videoSlider.setMaximum(duration)

    @pyqtSlot(str)
    def playVideo(self, path):
        # Play video in the QMediaPlayer
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()
        self.videoSubWindow.show()  # Ensure the video subwindow is visible

    def addDock(self, title, area):
        dock = QDockWidget(title, self)
        self.addDockWidget(area, dock)
        return dock

    def createSlider(self, label, layout, minimum, maximum, value, key):
        # Creating a slider and adding it to the layout
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(minimum)
        slider.setMaximum(maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda val, k=key: self.updateParam(k, val))
        self.params[key] = value
        layout.addWidget(QLabel(label))
        layout.addWidget(slider)

    def startBackendProcess(self):
        #params = {key: widget.value() for key, widget in self.params.items() if hasattr(widget, 'value')}
        # if not self.thread:
        self.thread = BackendThread(self.params)
        self.thread.imageGenerated.connect(self.updateImage)
        self.thread.finished.connect(self.playVideo)
        self.thread.start()
    def stopBackendProcess(self):
        try:
            from deforum.shared_storage import models
            models["deforum_pipe"].gen.max_frames = len(models["deforum_pipe"].images)
        except:
            pass


    def startBatchProcess(self):
        if not self.current_job and self.job_queue:
            self.runNextJob()

    def runNextJob(self):
        if self.job_queue:
            self.current_job = self.job_queue.pop(0)
            self.current_job.markRunning()  # Mark the job as running
            self.thread = BackendThread(self.current_job.job_data)
            self.thread.finished.connect(self.onJobFinished)
            self.thread.start()

    @pyqtSlot(str)
    def onJobFinished(self, result):
        print(f"Job completed with result: {result}")
        if self.current_job:
            self.current_job.markComplete()  # Mark the job as complete
            self.current_job = None  # Reset current job
            self.runNextJob()  # Run next job in the queue
        else:
            self.jobQueueList.clear()

    def stopBatchProcess(self):
        # if self.current_job:
            # Logic to stop the current thread if it's running
        try:
            from deforum.shared_storage import models
            models["deforum_pipe"].gen.max_frames = len(models["deforum_pipe"].images)
        except Exception as e:
            logger.info(repr(e))
        self.current_job = None
        # self.jobQueueList.clear()


    @pyqtSlot(dict)
    def updateImage(self, data):
        # Update the image on the label
        if 'image' in data:
            img = copy.deepcopy(data['image'])
            qpixmap = npArrayToQPixmap(np.array(img).astype(np.uint8))  # Convert to QPixmap
            # Get the size of the container QMdiSubWindow
            container_size = self.previewSubWindow.size()

            # Scale the QPixmap to fit within the container while maintaining the aspect ratio
            scaled_pixmap = qpixmap.scaled(container_size, Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation)

            # Set the scaled pixmap to the label
            self.previewLabel.setPixmap(scaled_pixmap)
            self.previewLabel.setScaledContents(True)
            self.timelineWidget.add_image_to_track(qpixmap)
    def updateParam(self, key, value):
        super().updateParam(key, value)
        from deforum.shared_storage import models

        # Load the deforum pipeline if not already loaded
        if "deforum_pipe" in models:

            prom = self.params.get('prompts', 'cat sushi')
            key = self.params.get('keyframes', '0')
            if prom == "":
                prom = "Abstract art"
            if key == "":
                key = "0"

            if not isinstance(prom, dict):
                new_prom = list(prom.split("\n"))
                new_key = list(key.split("\n"))
                self.params["animation_prompts"] = dict(zip(new_key, new_prom))
            else:
                self.params["animation_prompts"] = prom

            models["deforum_pipe"].live_update_from_kwargs(**self.params)
            print("UPDATED DEFORUM PARAMS")
    def updateUIFromParams(self):
        for widget in self.findChildren(QWidget):
            if hasattr(widget, 'accessibleName'):
                key = widget.accessibleName()
                if key in self.params:
                    value = self.params[key]
                    if isinstance(widget, QSpinBox):
                        widget.setValue(value)
                    elif isinstance(widget, QComboBox):
                        index = widget.findText(str(value))
                        if index != -1:
                            widget.setCurrentIndex(index)
                    elif isinstance(widget, QLineEdit):
                        widget.setText(str(value))
                    elif isinstance(widget, QTextEdit):
                        widget.setText(str(value))
                    elif isinstance(widget, QSlider):
                        widget.setValue(value)
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(value)