from PyQt6.QtWidgets import QApplication

from deforum import logger

import copy
import datetime
import json
import math
import os
import shutil
import sys

import imageio.v2 as imageio
import numpy as np
from qtpy.QtGui import QIcon, QPen, QBrush, QColor, QPainterPath
from qtpy.QtWidgets import QGraphicsEllipseItem, QGraphicsScene, QGraphicsView, QGraphicsPathItem, QCheckBox
from qtpy.QtCore import QTimer
from qtpy.QtGui import QKeyEvent
from qtpy.QtWidgets import QDockWidget, QMenu
from qtpy import QtWidgets
from qtpy.QtCore import Qt, Slot, QUrl, QSize, Signal
from qtpy.QtGui import QAction
from qtpy.QtMultimedia import QMediaPlayer, QAudioOutput
from qtpy.QtMultimediaWidgets import QVideoWidget
from qtpy.QtWidgets import QWidget, QVBoxLayout, QSlider, QLabel, QMdiArea, \
    QPushButton, QComboBox, QFileDialog, QHBoxLayout, QListWidget, \
    QListWidgetItem, QMessageBox, QScrollArea, QProgressBar, QSizePolicy

from deforum.pipelines.deforum_animation.animation_helpers import FrameInterpolator
from deforum.ui.qt_helpers.qt_image import npArrayToQPixmap
from deforum.ui.qt_modules.backend_thread import BackendThread
from deforum.ui.qt_modules.console_widget import NodesConsole, StreamRedirect
from deforum.ui.qt_modules.core import DeforumCore
from deforum.ui.qt_modules.custom_ui import ResizableImageLabel, JobDetailPopup, JobQueueItem, AspectRatioMdiSubWindow, \
    DetachableTabWidget, AutoReattachDockWidget, CustomTextBox
from deforum.ui.qt_modules.help import HelpDialog, AboutDialog
from deforum.ui.qt_modules.ref import TimeLineQDockWidget
from deforum.utils.constants import config
from deforum.ui.qt_modules.viz_thread import VisualGeneratorThread

DEFAULT_MILK = ""

# def list_milk_presets(folder):
#     return [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
#
# os.makedirs(os.path.join(config.root_path, 'milks'), exist_ok=True)
#
# known_dropdowns = {"milk_path":list_milk_presets(os.path.join(config.root_path, 'milks'))}

def list_presets(folder, known_extensions):
    return [f for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f)) and f.split('.')[-1] in known_extensions]

# Create necessary directories
os.makedirs(os.path.join(config.root_path, 'milks'), exist_ok=True)


# Define known extensions
known_extensions = ['pth', 'bin', 'ckpt', 'safetensors']  # Replace with actual extensions

# Populate known_dropdowns
known_dropdowns = {
    "milk_path": list_presets(os.path.join(config.root_path, 'milks'), ['txt']),
    "ad_sd_model": list_presets(os.path.join(config.comfy_path, 'models', 'checkpoints'), known_extensions),
    "ad_lora": list_presets(os.path.join(config.comfy_path, 'models', 'loras'), known_extensions),
    "ad_model": list_presets(os.path.join(config.comfy_path, 'models', 'animatediff_models'), known_extensions)
}

try:
    import pygame


    class JoystickHandler:
        def __init__(self):
            pygame.init()
            pygame.joystick.init()
            self.joysticks = [pygame.joystick.Joystick(i) for i in range(pygame.joystick.get_count())]
            for joystick in self.joysticks:
                joystick.init()

        def get_axis(self, joystick_index, axis):
            pygame.event.pump()
            if 0 <= joystick_index < len(self.joysticks):
                return self.joysticks[joystick_index].get_axis(axis)
            return 0
except:
    class JoystickHandler:
        def __init__(self):
            self.joystick_count = 0
            self.joystick = None
            self.joysticks = []

        def get_axis(self, joystick_index, axis):
            return 0

# Usage in your main application
joystick_handler = JoystickHandler()
class AnimationEngine:
    def __init__(self):
        self.parameters = {
            'translation_x': 0.0,
            'translation_y': 0.0,
            'translation_z': 0.0,
            'rotation_3d_x': 0.0,
            'rotation_3d_y': 0.0,
            'rotation_3d_z': 0.0
        }
        self.velocity = {
            'translation_x': 0.0,
            'translation_y': 0.0,
            'translation_z': 0.0,
            'rotation_3d_x': 0.0,
            'rotation_3d_y': 0.0,
            'rotation_3d_z': 0.0
        }
        self.acceleration = {
            'translation_x': 0.01,
            'translation_y': 0.01,
            'translation_z': 0.01,
            'rotation_3d_x': 0.003,
            'rotation_3d_y': 0.003,
            'rotation_3d_z': 0.001
        }
        self.max_velocity = 1.0
        self.easing_type = 'exponential'  # Default easing type
        self.key_held_down = {
            'translation_x': False,
            'translation_y': False,
            'translation_z': False,
            'rotation_3d_x': False,
            'rotation_3d_y': False,
            'rotation_3d_z': False
        }
        self.auto_return = {
            'translation_x': True,
            'translation_y': True,
            'translation_z': True,
            'rotation_3d_x': True,
            'rotation_3d_y': True,
            'rotation_3d_z': True
        }

    def ease(self, x):
        # You can also consider more aggressive easing functions here if needed
        if self.easing_type == 'linear':
            return x
        elif self.easing_type == 'easeInSine':
            return 1 - math.cos((x * math.pi) / 2)
        elif self.easing_type == 'easeOutSine':
            return math.sin((x * math.pi) / 2)
        elif self.easing_type == 'easeInOutSine':
            # Increase the impact as the parameter grows larger
            return -(math.cos(math.pi * x) - 1) / 2
        # Adding an exponential decay component for larger values
        elif self.easing_type == 'exponential':
            return x * math.exp(-0.1 * x)
    def update_parameter(self, param, direction):
        if self.key_held_down[param]:
            if direction == 'increase':
                self.velocity[param] += self.acceleration[param]
            elif direction == 'decrease':
                self.velocity[param] -= self.acceleration[param]
        else:
            # Apply easing function to decelerate smoothly to zero
            if abs(self.velocity[param]) > 0:
                self.velocity[param] -= self.ease(abs(self.velocity[param])) * (1 if self.velocity[param] > 0 else -1)
                if abs(self.velocity[param]) < 0.01:
                    self.velocity[param] = 0

        self.velocity[param] = max(min(self.velocity[param], self.max_velocity), -self.max_velocity)
        self.parameters[param] += self.velocity[param]

        # Enhanced auto-return logic
        if self.auto_return[param] and not self.key_held_down[param] and abs(self.velocity[param]) < 0.01:
            # Increase step size dynamically based on distance from zero
            distance = abs(self.parameters[param])
            step_multiplier = 0.1 if distance < 2 else 0.2  # More aggressive when far from zero
            step = self.ease(distance) * step_multiplier * (1 if self.parameters[param] > 0 else -1)
            self.parameters[param] -= step
            # Ensure that it snaps to zero when very close to prevent endless tiny oscillations
            if abs(self.parameters[param]) < 0.05:
                self.parameters[param] = 0

        self.parameters[param] = max(min(self.parameters[param], 10), -10)



    def set_key_held_down(self, param, is_held_down):
        self.key_held_down[param] = is_held_down

    def get_parameter(self, param):
        return self.parameters[param]

    def set_easing_type(self, easing_type):
        if easing_type in ['linear', 'easeInSine', 'easeOutSine', 'easeInOutSine']:
            self.easing_type = easing_type
        else:
            raise ValueError("Invalid easing type specified")
    def set_auto_return(self, param, value):
        self.auto_return[param] = value


class PointerCircle(QGraphicsView):
    def __init__(self, radius=50, parent=None, param_x='translation_x', param_y='translation_y'):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.radius = radius
        self.param_x = param_x
        self.param_y = param_y
        self.invert_x = False
        self.invert_y = False
        self.setFixedSize(radius * 2 + 10, radius * 2 + 10)
        self.setSceneRect(0, 0, radius * 2, radius * 2)

        # Draw the outer circle
        self.circle = QGraphicsEllipseItem(0, 0, radius * 2, radius * 2)
        self.circle.setPen(QPen(QColor(255, 255, 255), 2))
        self.circle.setBrush(QBrush(QColor(255, 255, 255, 100)))
        self.scene.addItem(self.circle)

        # Draw the pointer as a smaller circle
        self.pointer = QGraphicsEllipseItem(radius-5, radius-5, 10, 10)
        self.pointer.setBrush(QBrush(QColor(255, 0, 0)))
        self.scene.addItem(self.pointer)
        # Initialize the rotation arrow
        self.rotationArrow = QGraphicsPathItem()
        self.rotationArrow.setPen(QPen(QColor(0, 255, 0, 128), 2, Qt.SolidLine, Qt.RoundCap))
        self.scene.addItem(self.rotationArrow)

    def update_pointer(self, param_x_value, param_y_value):
        # Apply inversion based on state
        x_value = -param_x_value if self.invert_x else param_x_value
        y_value = -param_y_value if self.invert_y else param_y_value

        # Scale parameter values to fit within the radius (-10 to 10 becomes -radius to radius)
        scale = self.radius / 10.0
        scaled_x = x_value * scale
        scaled_y = y_value * scale

        # Compute the vector length from the center to the pointer
        vector_length = math.sqrt(scaled_x ** 2 + scaled_y ** 2)

        # Clamp the vector length to not exceed the radius
        if vector_length > self.radius:
            scaled_x = (scaled_x / vector_length) * self.radius
            scaled_y = (scaled_y / vector_length) * self.radius

        # Calculate center positions
        center_x = self.radius + scaled_x
        center_y = self.radius + scaled_y

        # Set the pointer's position
        self.pointer.setRect(center_x - 5, center_y - 5, 10, 10)

    def set_inversion(self, axis, state):
        if axis == 'x':
            self.invert_x = state
        elif axis == 'y':
            self.invert_y = state

    def update_rotation_arrow(self, rotation_z_value):
        # Clear the previous path
        path = QPainterPath()
        self.rotationArrow.setPath(path)

        if rotation_z_value != 0:
            # Determine the direction based on the sign of rotation_z_value
            direction = -1 if rotation_z_value > 0 else 1

            # Calculate the angle, adjusting the direction
            angle = direction * abs(rotation_z_value) / 10.0 * 360  # Scale rotation to a full circle

            # Start from the top (90 degrees) and draw the arc
            path.moveTo(self.radius, self.radius)
            path.arcTo(0, 0, self.radius * 2, self.radius * 2, 90, angle)
            self.rotationArrow.setPath(path)

class LiveControlDockWidget(QDockWidget):
    def __init__(self, parent=None, engine=None):
        super().__init__("Live Controls", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.engine = engine
        self.setObjectName('live_control')
        # Main widget and layout
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)

        # Adding circles to display translation and rotation
        self.translationCircle = PointerCircle(50)
        self.translationCircle.invert_y = True
        self.rotationCircle = PointerCircle(50)
        self.rotationCircle.invert_y = True

        layout.addWidget(self.translationCircle)
        layout.addWidget(self.rotationCircle)
        self.joystick_enabled = False
        self.joystick_button = QPushButton("Enable Joystick")
        self.joystick_button.setCheckable(True)
        self.joystick_button.toggled.connect(self.toggle_joystick)
        layout.addWidget(self.joystick_button)
        # Status labels and sliders for translation and rotation
        self.status_labels = {}
        self.sliders = {}
        self.dropdowns = {}
        self.checkboxes = {}
        self.joystick_axis_mapping = {axis: (-1, -1) for axis in self.engine.parameters}
        self.invert_joystick = {axis: False for axis in self.engine.parameters}


        for axis in ['translation_x', 'translation_y', 'translation_z', 'rotation_3d_x', 'rotation_3d_y',
                     'rotation_3d_z']:
            sublayout = QHBoxLayout()
            # Label
            label = QLabel(f"{axis}: 0.0")
            sublayout.addWidget(label)
            self.status_labels[axis] = label

            # Slider for acceleration
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 200)  # acceleration factor range
            slider.setValue(10)  # default value
            slider.valueChanged.connect(lambda value, a=axis: self.adjust_acceleration(a, value))

            sublayout.addWidget(slider)
            self.sliders[axis] = slider
            # Add checkbox for auto return
            auto_return_chk = QCheckBox("Auto Return to Zero")
            auto_return_chk.setChecked(True)
            auto_return_chk.toggled.connect(lambda checked, a=axis: self.toggle_auto_return(a, checked))
            sublayout.addWidget(auto_return_chk)

            axis_dropdown = QComboBox()
            if joystick_handler.joysticks:
                axis_dropdown.addItems(['None'] + [f'Joystick {i} Axis {j}' for i in range(len(joystick_handler.joysticks)) for j in range(joystick_handler.joysticks[i].get_numaxes())])
            else:
                axis_dropdown.addItems(['None'])
            axis_dropdown.currentIndexChanged.connect(lambda index, a=axis: self.set_joystick_axis(a, index-1))
            sublayout.addWidget(axis_dropdown)
            self.dropdowns[axis] = axis_dropdown

            # Checkbox for inverting joystick input
            invert_checkbox = QCheckBox("Invert")
            invert_checkbox.toggled.connect(lambda checked, a=axis: self.set_invert_joystick(a, checked))
            sublayout.addWidget(invert_checkbox)
            self.checkboxes[axis] = invert_checkbox

            layout.addLayout(sublayout)


        # Control visibility button
        self.toggleButton = QPushButton("Toggle Live Control View")
        self.toggleButton.clicked.connect(self.toggle_visibility)
        layout.addWidget(self.toggleButton)

        self.setWidget(widget)
        self.setVisible(False)

    def update_status(self, axis, value):
        self.status_labels[axis].setText(f"{axis}: {value:.2f}")

        # Update pointer positions based on axis
        if axis in ['translation_x', 'translation_y']:
            self.translationCircle.update_pointer(self.engine.get_parameter('translation_x'),
                                                  self.engine.get_parameter('translation_y'))
        if axis in ['rotation_3d_x', 'rotation_3d_y']:
            self.rotationCircle.update_pointer(self.engine.get_parameter('rotation_3d_y'),
                                               self.engine.get_parameter('rotation_3d_x'))
        if axis in ['rotation_3d_z']:
            self.rotationCircle.update_rotation_arrow(self.engine.get_parameter('rotation_3d_z'))

    def adjust_acceleration(self, axis, value):
        # Convert slider value to a suitable acceleration factor
        new_acceleration = value * 0.001
        self.engine.acceleration[axis] = new_acceleration
        # self.status_labels[axis].setText(f"{axis}: {self.engine.parameters[axis]:.2f} (Acc: {new_acceleration:.2f})")

    def toggle_visibility(self):
        self.setVisible(not self.isVisible())
    def toggle_auto_return(self, axis, checked):
        self.engine.set_auto_return(axis, checked)
        # Update the status label to reflect the change
        # self.status_labels[axis].setText(f"{axis}: {self.engine.get_parameter(axis):.2f} (Auto Return: {'On' if checked else 'Off'})")
    def toggle_joystick(self, checked):
        self.joystick_enabled = checked
        self.joystick_button.setText("Disable Joystick" if checked else "Enable Joystick")

    def set_joystick_axis(self, axis, index):
        if index == -1:
            self.joystick_axis_mapping[axis] = (-1, -1)
        else:
            joystick_index = index // max([j.get_numaxes() for j in joystick_handler.joysticks])
            axis_index = index % max([j.get_numaxes() for j in joystick_handler.joysticks])
            self.joystick_axis_mapping[axis] = (joystick_index, axis_index)

    def set_invert_joystick(self, axis, invert):
        self.invert_joystick[axis] = invert

    def update_parameters_from_joystick(self):
        if not self.joystick_enabled:
            return
        for axis, (joystick_index, axis_index) in self.joystick_axis_mapping.items():
            if joystick_index >= 0 and axis_index >= 0:
                raw_input = joystick_handler.get_axis(joystick_index, axis_index)
                inverted = -1 if self.invert_joystick[axis] else 1
                self.engine.parameters[axis] = (self.engine.acceleration[axis] * 1000) * raw_input * inverted
                self.status_labels[axis].setText(f"{axis}: {(self.engine.acceleration[axis] * 1000) * raw_input * inverted:.2f}")


class MainWindow(DeforumCore):
    def __init__(self):
        super().__init__()
        self.defaults()
        self.setupWindowStyle()
        self.setupToolbar()
        self.setupMenu()
        self.setupDynamicUI()
        self.setupPreviewArea()
        self.setupTimeline()
        self.setupBatchControls()
        self.setupVideoPlayer()
        self.create_console_widget()
        self.setupHelpMenu()
        self.tileMdiSubWindows()
        self.timelineDock.hide()
        self.newProject()
        self.initAnimEngine()
        # self.loadWindowState()

        preset_path = os.path.join(self.presets_folder, 'default.txt')
        if os.path.exists(preset_path):
            self.loadPreset(preset_path)


    def defaults(self):
        self.current_job = None  # To keep track of the currently running job
        self.currentTrack = None  # Store the currently selected track
        self.project_replay = False
        self.job_queue = []  # List to keep jobs to be processed
        self.presets_folder = os.path.join(config.root_path, 'presets')
        self.projects_folder = os.path.join(config.root_path, 'projects')
        self.state_file = os.path.join(config.root_path, '.lastview')
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        os.makedirs(self.presets_folder, exist_ok=True)
        os.makedirs(self.projects_folder, exist_ok=True)

    def setupWindowStyle(self):
        self.setWindowTitle("Deforum Engine")

    def setupToolbar(self):
        self.toolbar = self.addToolBar('Main Toolbar')
        self.toolbar.setObjectName('main_toolbar')

        self.renderButton = QAction('Render', self)
        self.renderButton.triggered.connect(self.startBackendProcess)

        self.renderVizButton = QAction('Render Viz', self)
        self.renderVizButton.triggered.connect(self.generateVizData)

        self.stopRenderButton = QAction('Stop Render', self)
        self.stopRenderButton.triggered.connect(self.stopBackendProcess)

        # Add presets dropdown to the toolbar
        # self.presetsDropdown = QMenu()
        # #self.presetsDropdown.currentIndexChanged.connect(self.loadPreset)
        # # self.presetsDropdown.triggered.connect(self.loadPresetsDropdown)
        # self.loadPresetsDropdown()

        # Add actions to load and save presets
        self.loadPresetAction = QAction('Load Preset', self)
        self.loadPresetAction.triggered.connect(self.loadPreset)

        self.tileSubWindowsAction = QAction('Tile Subwindows', self)
        self.tileSubWindowsAction.triggered.connect(self.tileMdiSubWindows)

        self.savePresetAction = QAction('Save Preset', self)
        self.savePresetAction.triggered.connect(self.savePreset)

        self.addJobButton = QAction('Add to Batch', self)
        self.addJobButton.triggered.connect(self.addCurrentParamsAsJob)

        self.toolbar.addAction(self.renderButton)
        self.toolbar.addAction(self.renderVizButton)
        self.toolbar.addAction(self.stopRenderButton)
        # self.toolbar.addWidget(self.presetsDropdown)
        # self.toolbar.addAction(self.loadPresetAction)
        self.toolbar.addAction(self.tileSubWindowsAction)
        self.toolbar.addAction(self.savePresetAction)
        self.toolbar.addAction(self.addJobButton)

        self.renderDropdown = QComboBox()
        self.toolbar.addWidget(self.renderDropdown)

        self.loadRenderButton = QPushButton("Load Render")
        self.loadRenderButton.clicked.connect(self.loadSelectedRender)
        self.toolbar.addWidget(self.loadRenderButton)
        self.controlToggleButton = QAction(QIcon(), 'Enable Controls', self)
        self.controlToggleButton.setCheckable(True)
        self.controlToggleButton.setChecked(False)
        self.controlToggleButton.triggered.connect(self.toggleControlListening)
        self.toolbar.addAction(self.controlToggleButton)
        self.updateRenderDropdown()  # Populate the dropdown

        # Adding the progress bar and status label
        self.progressBar = QProgressBar()
        self.statusLabel = QLabel("Ready")
        self.statusLabel.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        # Create a widget to hold the progress bar and label
        statusWidget = QWidget()
        statusLayout = QHBoxLayout()
        statusLayout.addWidget(self.progressBar)
        statusLayout.addWidget(self.statusLabel)
        statusLayout.setContentsMargins(0, 0, 0, 0)  # Reduce margins to fit the toolbar style
        statusWidget.setLayout(statusLayout)

        # Adding the status widget to the toolbar
        spacer = QWidget()  # This spacer pushes the status to the right
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)
        self.toolbar.addWidget(statusWidget)



    def toggleControlListening(self, checked):
        if checked:
            self.controlToggleButton.setIcon(QIcon())  # Set an icon or change color
            self.controlToggleButton.setText('Disable Controls')
            self.startListening()
        else:
            self.controlToggleButton.setIcon(QIcon())
            self.controlToggleButton.setText('Enable Controls')
            self.stopListening()

    def startListening(self):
        self.listen_to_controls = True
        self.updateStatusBar("Controls Enabled", "green")

    def stopListening(self):
        self.listen_to_controls = False
        self.updateStatusBar("Controls Disabled", "red")

    def updateStatusBar(self, message, color):
        self.statusBar().setStyleSheet(f"background-color: {color};")
        self.statusBar().showMessage(message)


    def updateRenderDropdown(self):
        root_dir = os.path.join(config.root_path, 'output', 'deforum')
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        folders = sorted(next(os.walk(root_dir))[1]) # List directories only
        self.renderDropdown.clear()
        self.renderDropdown.addItems(folders)

    def loadSelectedRender(self):
        selected_folder = self.renderDropdown.currentText()
        root_dir = os.path.join(config.root_path, 'output', 'deforum', selected_folder)
        image_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.png')]
        image_files.sort()  # Ensure the files are sorted correctly

        # Optionally create a temporary video file from images
        video_path = self.createTempVideo(image_files)
        self.playVideo({'video_path': video_path})

        # Check for a settings file and load it
        settings_file = os.path.join(root_dir, 'settings.txt')
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as file:
                settings = json.load(file)
                self.applySettings(settings)
        # Set additional parameters for resuming project
        self.params['resume_path'] = root_dir
        if '_' in selected_folder:
            self.params['resume_timestring'] = selected_folder.split('_')[-1]
        else:
            self.params['resume_timestring'] = 'unknown'
        self.params['resume_from'] = len(image_files) - 1
        # Assuming applySettings updates UI from self.params:
        self.updateUIFromParams()

    def createTempVideo(self, image_files):
        video_path = os.path.join(config.root_path, 'output', 'deforum', 'temp_video.mp4')
        writer = imageio.get_writer(video_path, fps=24, codec='libx264', quality=9,
                                    pixelformat='yuv420p')  # High quality

        for image_file in image_files:
            image = imageio.imread(image_file)
            writer.append_data(image)

        writer.close()
        return video_path

    def applySettings(self, settings):
        # Apply the settings loaded from the text file
        for key, value in settings.items():
            if key in self.params:
                self.params[key] = value
        self.updateUIFromParams()  # Reflect updates


    def setupMenu(self):
        menuBar = self.menuBar()

        fileMenu = menuBar.addMenu('&File')

        newProjectAction = QAction('&New Project', self)
        newProjectAction.triggered.connect(self.newProject)
        fileMenu.addAction(newProjectAction)

        loadProjectAction = QAction('&Load Project', self)
        loadProjectAction.triggered.connect(self.loadProject)
        fileMenu.addAction(loadProjectAction)

        saveProjectAction = QAction('&Save Project', self)
        saveProjectAction.triggered.connect(self.saveProject)
        fileMenu.addAction(saveProjectAction)

        convertProjectAction = QAction('&Convert Project', self)
        convertProjectAction.triggered.connect(self.convertProjectToSingleSettingsFile)
        fileMenu.addAction(convertProjectAction)

        # Add Theme Switcher
        themeMenu = fileMenu.addMenu('&Themes')
        self.loadThemeMenu(themeMenu)

        self.presetsDropdown = menuBar.addMenu('&Presets')
        #self.presetsDropdown.currentIndexChanged.connect(self.loadPreset)
        # self.presetsDropdown.triggered.connect(self.loadPresetsDropdown)
        self.loadPresetsDropdown()
        menuBar.addMenu(self.presetsDropdown)
        layoutsMenu = menuBar.addMenu('&Layouts')
        self.setupLayoutsMenu(layoutsMenu)

    def loadThemeMenu(self, themeMenu):
        """Load theme switcher menu with available QSS files."""
        qss_folder = os.path.join(config.src_path, 'deforum', 'ui', 'qss')

        if os.path.exists(qss_folder):
            for qss_file in os.listdir(qss_folder):
                if qss_file.endswith('.qss'):
                    theme_action = QAction(qss_file, self)
                    theme_action.triggered.connect(lambda checked, qss=qss_file: self.applyTheme(qss))
                    themeMenu.addAction(theme_action)

    def applyTheme(self, qss_file):
        """Apply the selected QSS theme."""
        qss_path = os.path.join(config.src_path, 'deforum', 'ui', 'qss', qss_file)
        with open(qss_path, 'r') as file:
            qss = file.read()
            QApplication.instance().setStyleSheet(qss)
    def setupLayoutsMenu(self, layoutsMenu):
        saveLayoutAction = QAction('&Save Current Layout', self)
        saveLayoutAction.triggered.connect(self.saveCurrentLayout)

        setDefaultLayoutAction = QAction('Set as Default Layout', self)
        setDefaultLayoutAction.triggered.connect(self.saveDefaultLayout)

        # Submenu for available layouts
        self.layoutsSubmenu = QMenu('Load Layout', self)
        self.updateLayoutsSubmenu()

        layoutsMenu.addAction(saveLayoutAction)
        layoutsMenu.addAction(setDefaultLayoutAction)
        layoutsMenu.addMenu(self.layoutsSubmenu)

    def setupDynamicUI(self):
        # Initialize the tabbed control layout docked on the left
        self.controlsDock = self.addDock("Controls", Qt.DockWidgetArea.LeftDockWidgetArea)
        self.tabWidget = DetachableTabWidget(self)
        self.tabWidget.setMinimumSize(QSize(0,0))
        self.controlsDock.setWidget(self.tabWidget)

        # Load UI configuration from JSON
        curr_folder = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(curr_folder, 'ui.json')
        with open(json_path, 'r') as file:
            config = json.load(file)

        for category, settings in config.items():
            print("parsing", category)
            tab = QWidget()  # This is the actual tab that will hold the layout
            layout = QVBoxLayout()
            tab.setLayout(layout)

            for setting, params in settings.items():
                if params['widget_type'] == 'number':
                    self.createSpinBox(params['label'], layout, params['min'], params['max'], 1, params['default'], setting)
                elif params['widget_type'] == 'float':
                    self.createDoubleSpinBox(params['label'], layout, params['min'], params['max'], 0.01, params['default'], setting)

                elif params['widget_type'] == 'dropdown':
                    if setting in known_dropdowns:
                        options = known_dropdowns[setting]
                    else:
                        options = [str(param) for param in params['options']]
                    self.createComboBox(params['label'], layout, options, setting)
                elif params['widget_type'] == 'text input':
                    self.createTextInput(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'text box':
                    self.createTextBox(params['label'], layout, params['default'], setting)
                elif params['widget_type'] == 'slider':
                    self.createSlider(params['label'], layout, params['min'], params['max'], params['default'], setting)
                elif params['widget_type'] == 'checkbox':
                    self.createCheckBox(params['label'], layout, bool(params['default']), setting)
                elif params['widget_type'] == 'file_input':
                    # Add file input handler if needed
                    pass
            scroll = QScrollArea()  # Create a scroll area
            scroll.setWidget(tab)
            scroll.setWidgetResizable(True)  # Make the scroll area resizable

            self.tabWidget.addTabWithDetachButton(scroll, category)  # Add the scroll area to the tab widget

    def saveCurrentLayout(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Layout', os.path.join(config.root_path, 'ui', 'layouts'),
                                                  'Layout Files (*.dlay)')
        if filename:
            if not filename.endswith('.dlay'):
                filename += '.dlay'
            self.saveWindowState(filename)
            self.updateLayoutsSubmenu()

    def saveDefaultLayout(self):
        default_layout_path = os.path.join(config.root_path, 'ui', 'layouts', 'default.dlay')
        self.saveWindowState(default_layout_path)


    def updateLayoutsSubmenu(self):
        self.layoutsSubmenu.clear()
        layouts_dir = os.path.join(config.root_path, 'ui', 'layouts')
        if not os.path.exists(layouts_dir):
            os.makedirs(layouts_dir)

        for filename in os.listdir(layouts_dir):
            if filename.endswith('.dlay'):
                action = QAction(filename[:-5], self)
                action.triggered.connect(
                    lambda checked, path=os.path.join(layouts_dir, filename): self.loadWindowState(path))
                self.layoutsSubmenu.addAction(action)


    def setupPreviewArea(self):
        # Setting up an MDI area for previews
        self.mdiArea = QMdiArea()
        self.setCentralWidget(self.mdiArea)

        self.previewLabel = ResizableImageLabel()
        self.previewSubWindow = AspectRatioMdiSubWindow(self.previewLabel)
        self.mdiArea.addSubWindow(self.previewSubWindow)
        self.previewSubWindow.setWidget(self.previewLabel)
        self.previewSubWindow.show()

    def setupTimeline(self):
        self.timelineDock = TimeLineQDockWidget(self)
        self.timelineDock.setObjectName('timeline_dock')
        # self.timelineDock = QDockWidget("Timeline", self)
        self.timelineDock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.timelineDock)

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

    def setupVideoPlayer(self):
        self.player = QMediaPlayer()  # Create a media player object
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
        # self.videoSlider.valueChanged.connect(self.setResumeFrom)
        # self.player.mediaStatusChanged.connect(self.onMediaStatusChanged)
    def setupHelpMenu(self):
        # Create the Help menu
        help_menu = self.menuBar().addMenu("Help")

        # Create Help action
        help_action = QAction("Help", self)
        help_action.triggered.connect(self.show_help_dialog)
        help_menu.addAction(help_action)

        # Create About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_help_dialog(self):
        help_dialog = HelpDialog()
        help_dialog.exec()

    def show_about_dialog(self):
        about_dialog = AboutDialog()
        about_dialog.exec()
    def create_console_widget(self):
        # Create a text widget for stdout and stderr
        self.text_widget = NodesConsole()
        # Set up the StreamRedirect objects
        self.stdout_redirect = StreamRedirect()
        self.stderr_redirect = StreamRedirect()
        self.stdout_redirect.text_written.connect(self.text_widget.write)
        self.stderr_redirect.text_written.connect(self.text_widget.write)
        sys.stdout = self.stdout_redirect
        sys.stderr = self.stderr_redirect

        self.console = QDockWidget()
        self.console.setObjectName('console_widget')
        self.console.setWindowTitle("Console")
        self.console.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(widget)
        layout.setContentsMargins(5,5,5,5)
        layout.addWidget(self.text_widget)
        self.console.setWidget(widget)
        #layout.addWidget(self.text_widget2)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.console)
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
    def closeEvent(self, event):
        self.saveWindowState()
        self.savePresetToFile('default.txt')
        super().closeEvent(event)

    def saveWindowState(self, file=None):
        if not file:
            file = self.state_file
        with open(file, 'w') as f:
            geometry_base64 = self.saveGeometry().toBase64().data().decode('utf-8')
            state_base64 = self.saveState().toBase64().data().decode('utf-8')
            f.write(geometry_base64 + '\n')
            f.write(state_base64 + '\n')

            for dock in self.findChildren(AutoReattachDockWidget):
                dock_state = {
                    'name': dock.objectName(),
                    'should_detach':dock.should_detach,
                    'floating': dock.isFloating(),
                    'geometry': dock.saveGeometry().toBase64().data().decode('utf-8'),
                    'visible': dock.isVisible(),
                    'is_detached': dock.isFloating(),  # or any other logic you determine
                    'dock_area': str(self.dockWidgetArea(dock)).split('.')[1]
                }

                json.dump(dock_state, f)
                f.write('\n')
            for textbox in self.findChildren(CustomTextBox):
                textbox_state = {
                    'name': textbox.objectName(),
                    'scale': textbox.scale
                }
                json.dump(textbox_state, f)
                f.write('\n')
            # Saving additional states for LiveControlDockWidget
            live_dock = self.liveControlDock
            if live_dock:
                joystick_mappings = live_dock.joystick_axis_mapping
                slider_values = {axis: slider.value() for axis, slider in live_dock.sliders.items()}
                joystick_inversions = live_dock.invert_joystick

                live_dock_state = {
                    'joystick_mappings': joystick_mappings,
                    'slider_values': slider_values,
                    'joystick_inversions': joystick_inversions
                }
                json.dump(live_dock_state, f)
                f.write('\n')

    def loadWindowState(self, file=None):
        if not file:
            file = self.state_file
        try:
            with open(file, 'r') as f:
                self.restoreGeometry(bytes(f.readline().strip(), 'utf-8'))
                self.restoreState(bytes(f.readline().strip(), 'utf-8'))
                while True:
                    line = f.readline().strip()
                    if not line:
                        break
                    dock_state = json.loads(line)

                    if 'joystick_mappings' in dock_state:  # Check if it's the joystick state
                        live_dock = self.liveControlDock
                        if live_dock:
                            live_dock.joystick_axis_mapping = dock_state['joystick_mappings']

                            for key, widget in live_dock.dropdowns.items():
                                widget.setCurrentIndex(widget.findText(f"Axis {dock_state['joystick_mappings'].get(key, '99')}"))

                            for axis, value in dock_state['slider_values'].items():
                                live_dock.sliders[axis].setValue(value)
                            for axis, inverted in dock_state['joystick_inversions'].items():
                                live_dock.invert_joystick[axis] = inverted

                            for key, widget in live_dock.checkboxes.items():
                                widget.setChecked(dock_state['joystick_inversions'].get(key, False))

                    elif 'scale' in dock_state:  # This identifies a CustomTextBox
                        textbox = self.findChild(CustomTextBox, dock_state['name'])
                        if textbox:
                            textbox.scale = dock_state['scale']
                    else:
                        for button in self.tabWidget.buttons:
                            if button.property('title') == dock_state['name']:
                                button.click()

                        dock = self.findChild(AutoReattachDockWidget, dock_state['name'])
                        if dock:
                            area_map = {
                                "TopDockWidgetArea": Qt.DockWidgetArea.TopDockWidgetArea,
                                "BottomDockWidgetArea": Qt.DockWidgetArea.BottomDockWidgetArea,
                                "LeftDockWidgetArea": Qt.DockWidgetArea.LeftDockWidgetArea,
                                "RightDockWidgetArea": Qt.DockWidgetArea.RightDockWidgetArea,
                                "NoDockWidgetArea": Qt.DockWidgetArea.NoDockWidgetArea
                            }
                            dock.restoreGeometry(bytes(dock_state['geometry'], 'utf-8'))
                            self.addDockWidget(area_map[dock_state['dock_area']], dock)
                            dock.setFloating(False)
                # # Second pass: restore tabbing relationships
                # for dock_info in dock_state:
                #     dock = self.findChild(QDockWidget, dock_info['name'])
                #     if dock and dock_info['tabified_with']:
                #         for tab_name in dock_info['tabified_with']:
                #             tab_dock = self.findChild(QDockWidget, tab_name)
                #             if tab_dock:
                #                 self.tabifyDockWidget(dock, tab_dock)
        except FileNotFoundError:
            logger.info("No saved state to load.")
        except Exception as e:
            logger.info(f"Failed to load state: {e}")

    def initAnimEngine(self):
        self.engine = AnimationEngine()

        self.axis_bindings = {
            'translation_x': ('A', 'D'),  # Increase, Decrease
            'translation_y': ('S', 'W'),
            'translation_z': ('F', 'R'),
            'rotation_3d_x': ('I', 'K'),
            'rotation_3d_y': ('J', 'L'),
            'rotation_3d_z': ('U', 'O')
        }
        self.key_to_axis = {key: axis for axis, keys in self.axis_bindings.items() for key in keys}
        self.keys_pressed = {key: False for axis in self.axis_bindings for key in self.axis_bindings[axis]}
        self.listen_to_controls = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(50)  # Update interval in milliseconds
        self.liveControlDock = LiveControlDockWidget(self, engine=self.engine)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.liveControlDock)
        self.controlToggleButton.triggered.connect(self.liveControlDock.toggle_visibility)

    def newProject(self):
        self.project = {
            'params': {},  # Global parameters if any
            'frames': {}  # Dictionary to store parameters per frame
        }

    def loadProject(self):
        # Load project settings and parameters from a file
        fname = QFileDialog.getOpenFileName(self, 'Open Project File', self.projects_folder, 'Deforum Project Files (*.dproj)')
        if fname[0]:
            with open(fname[0], 'r') as file:
                self.project = json.load(file)  # Load the entire project dictionary
                if 'frames' in self.project and self.project['frames']:
                    # Assuming frames are stored with an index key that starts at 0
                    self.params = self.project['frames'].get('1', self.params)
                    self.updateUIFromParams()  # Update UI elements with loaded params

    def saveProject(self):
        # Save current settings and entire project data to a file
        fname = QFileDialog.getSaveFileName(self, 'Save Project File', self.projects_folder, 'Deforum Project Files (*.dproj)')
        if fname[0]:
            if not fname[0].endswith('.dproj'):
                fname = fname[0] + '.dproj'  # Ensure the file has the correct extension
            with open(fname, 'w') as file:
                json.dump(self.project, file, indent=4)  # Save the entire project dictionary
    def saveCurrentProjectAsSettingsFile(self):
        if self.project and 'frames' in self.project:
            # Step 2: Use Frame 1 as a base configuration
            base_params = self.project['frames']['1'].copy()
            max_frame_index = max(map(int, self.project['frames'].keys()))
            # Initialize parameter consolidation
            consolidated_params = {}
            for param in base_params:
                if isinstance(base_params[param], str) and ':' in base_params[param]:
                    consolidated_params[param] = {}

            # Step 3: Collect and consolidate parameter values from all frames
            for frame_index in range(1, max_frame_index + 1):
                frame_key = str(frame_index)
                if frame_key in self.project['frames']:
                    for param in consolidated_params:
                        value = self.project['frames'][frame_key].get(param, "")
                        if value and ':' in value:  # Check if value fits the pattern requiring consolidation
                            # Extract value and append to the schedule
                            try:
                                schedule_part = value.split(':')[-1].strip()
                                consolidated_params[param][frame_index - 1] = schedule_part
                            except IndexError:
                                continue

            # Integrate consolidated parameters into base_params
            for param, schedule in consolidated_params.items():
                if schedule:
                    schedule_str = ", ".join(f"{k}: {v}" for k, v in schedule.items())
                    base_params[param] = schedule_str

            # Step 4: Save the consolidated settings to a .txt file
            file_path = os.path.join(self.params["resume_path"], "settings.txt")
            with open(file_path, 'w') as file:
                json.dump(base_params, file, indent=4)

    def convertProjectToSingleSettingsFile(self):
        # Step 1: Load a .dproj project file
        fname = QFileDialog.getOpenFileName(self, 'Open Project File', '', 'Deforum Project Files (*.dproj)')
        if not fname[0]:
            return
        with open(fname[0], 'r') as file:
            self.project = json.load(file)

        if 'frames' not in self.project or not self.project['frames']:
            QMessageBox.warning(self, "Error", "No frame data found in the project.")
            return

        # Step 2: Use Frame 1 as a base configuration
        base_params = self.project['frames']['1'].copy()
        max_frame_index = max(map(int, self.project['frames'].keys()))

        # Initialize parameter consolidation
        consolidated_params = {}
        for param in base_params:
            if isinstance(base_params[param], str) and ':' in base_params[param]:
                consolidated_params[param] = {}

        # Step 3: Collect and consolidate parameter values from all frames
        for frame_index in range(1, max_frame_index + 1):
            frame_key = str(frame_index)
            if frame_key in self.project['frames']:
                for param in consolidated_params:
                    value = self.project['frames'][frame_key].get(param, "")
                    if value and ':' in value:  # Check if value fits the pattern requiring consolidation
                        # Extract value and append to the schedule
                        try:
                            schedule_part = value.split(':')[-1].strip()
                            consolidated_params[param][frame_index - 1] = schedule_part
                        except IndexError:
                            continue

        # Integrate consolidated parameters into base_params
        for param, schedule in consolidated_params.items():
            if schedule:
                schedule_str = ", ".join(f"{k}: {v}" for k, v in schedule.items())
                base_params[param] = schedule_str

        # Step 4: Save the consolidated settings to a .txt file
        fname = QFileDialog.getSaveFileName(self, 'Save Consolidated Settings File', '', 'Text Files (*.txt)')
        if fname[0]:
            with open(fname[0], 'w') as file:
                json.dump(base_params, file, indent=4)

    def onJobDoubleClicked(self, item):
        widget = self.jobQueueList.itemWidget(item)
        if widget:
            # Create a JobDetailPopup and add it to the MDI area
            popup = JobDetailPopup(widget.job_data)
            subWindow = self.mdiArea.addSubWindow(popup)  # Add popup to MDI area
            popup.setWindowTitle("Job Details - " + widget.job_name)
            subWindow.show()


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
            logger.info(f"Error loading {file_path}: {e}")
            QMessageBox.warning(self, "Loading Error", f"Failed to load {file_path}: {e}")

    def addCurrentParamsAsJob(self, job_name=None, job_params=None):
        if not job_name:
            job_name = f"{self.params.get('batch_name')}_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            job_params = self.params

        item = QListWidgetItem(self.jobQueueList)
        widget = JobQueueItem(job_name, job_params)
        item.setSizeHint(widget.sizeHint())
        self.jobQueueList.addItem(item)
        self.jobQueueList.setItemWidget(item, widget)
        self.job_queue.append(widget)  # Append job widget to the queue

    def loadPresetsDropdown(self, restore=None):
        if not os.path.exists(self.presets_folder):
            os.makedirs(self.presets_folder)

        def list_files(folder):
            items = []
            for entry in os.listdir(folder):
                full_path = os.path.join(folder, entry)
                if os.path.isdir(full_path):
                    sub_items = list_files(full_path)
                    items.append((entry, sub_items))
                elif entry.endswith('.txt'):
                    items.append(entry)
            return items

        preset_items = list_files(self.presets_folder)

        self.presetsDropdown.clear()

        def add_items(menu, items, path=''):
            for item in items:
                if isinstance(item, tuple):
                    sub_menu = QMenu(item[0], menu)
                    menu.addMenu(sub_menu)
                    add_items(sub_menu, item[1], os.path.join(path, item[0]))
                else:
                    full_item_path = os.path.join(path, item)
                    action = QAction(item, menu)
                    action.triggered.connect(
                        lambda _, p=os.path.join(self.presets_folder, full_item_path): self.loadPreset(p))
                    menu.addAction(action)

        add_items(self.presetsDropdown, preset_items)

    def loadPreset(self, preset_path):
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
        self.savePresetToFile(preset_name)
    def savePresetToFile(self, preset_name=None):
        if preset_name:
            preset_name = os.path.splitext(os.path.basename(preset_name))[0] + '.txt'
            preset_path = os.path.join(self.presets_folder, preset_name)
            with open(preset_path, 'w') as file:
                json.dump(self.params, file, indent=4)
            self.loadPresetsDropdown()  # Reload presets dropdown without restoring as it uses QMenu


    def setResumeFrom(self, position):
        # Assuming the frame rate is stored in self.frameRate
        frame_rate = self.params.get('fps', 24)  # you'll need to set this appropriately
        frame_number = int((position / 1000) * frame_rate)
        self.params["resume_from"] = frame_number + 1
        # Update the widget directly if necessary
        # if 'resume_from' in self.widgets:
        #     self.widgets['resume_from'].valueChanged.disconnect()
        #     self.widgets['resume_from'].setValue(frame_number)
        #     self.widgets['resume_from'].valueChanged.connect(lambda val, k='resume_from': self.updateParam(k, val))


    def setPosition(self, position):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self.player.setPosition(position)

    def updatePosition(self, position):
        self.videoSlider.setValue(position)

    def updateDuration(self, duration):
        if duration > 0:
            self.videoSlider.setMaximum(duration)
            # Optionally, you might want to adjust the ticks interval based on the video duration
            self.videoSlider.setTickInterval(duration // 100)  # For example, 100 ticks across the slider

    @Slot(dict)
    def cleanupThread(self):
        pass
        # if self.thread is not None:
        #     self.thread.wait()  # Ensure the thread has finished
        #     self.thread.deleteLater()  # Properly dispose of the thread object
        #     self.thread = None  # Remove the reference, allowing garbage collection

    @Slot(dict)
    def playVideo(self, data):
        self.current_data = data

        # Ensure the player stops properly before setting a new source
        if self.player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
            self.player.mediaStatusChanged.connect(self.on_media_status_changed)
            self.player.stop()
            self.seekToEnd()
            self.player.play()
        else:
            self.loadNewVideo(data)

    def loadNewVideo(self, data):
        if 'video_path' in data:
            # Reinitialize the audio output if it exists
            if hasattr(self, 'audioOutput'):
                self.audioOutput.deleteLater()  # Properly dispose of the old output
                self.audioOutput = QAudioOutput()
                self.player.setAudioOutput(self.audioOutput)  # Set the new audio output

            self.player.setSource(QUrl.fromLocalFile(data['video_path']))
            self.player.play()

        if 'timestring' in data:
            self.updateWidgetValue('resume_timestring', data['timestring'])
            self.updateWidgetValue('resume_path', data['resume_path'])
            self.updateWidgetValue('resume_from', data['resume_from'])
            # Optionally save settings, commented out for now

    def on_media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.player.mediaStatusChanged.disconnect(self.on_media_status_changed)
            self.loadNewVideo(self.current_data)  # Ensure `current_data` is stored as part of the call

    def seekToEnd(self):
        # Ensure the player has loaded the media
        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.LoadedMedia:
            # Seek to a point near the end of the video
            duration = self.player.duration()
            self.player.setPosition(max(0, duration - 1000))

    # QMediaPlayer.MediaStatus.NoMedia
    # QMediaPlayer.MediaStatus.EndOfMedia
    # QMediaPlayer.MediaStatus.InvalidMedia
    # QMediaPlayer.MediaStatus.LoadingMedia
    # QMediaPlayer.MediaStatus.LoadedMedia
    # QMediaPlayer.MediaStatus.BufferedMedia
    # QMediaPlayer.MediaStatus.BufferedMedia
    # QMediaPlayer.MediaStatus.StalledMedia

    # @Slot(dict)
    # def playVideo(self, data):
    #     # Ensure the player stops properly before setting a new source
    #     if self.player.playbackState() != QMediaPlayer.PlaybackState.StoppedState:
    #         self.player.stop()
    #         self.player.waitForStopped(-1)  # Wait for the player to stop if necessary
    #     self.player.setSource(QUrl())
    #
    #     if 'video_path' in data:
    #         # Reinitialize the audio output if it exists
    #         # if hasattr(self, 'audioOutput'):
    #         #     self.audioOutput.deleteLater()  # Properly dispose of the old output
    #         #     self.audioOutput = QAudioOutput()
    #         #     self.player.setAudioOutput(self.audioOutput)  # Set the new audio output
    #         self.player.setSource(QUrl.fromLocalFile(data['video_path']))
    #         self.player.play()
    #     if 'timestring' in data:
    #         self.updateWidgetValue('resume_timestring', data['timestring'])
    #         self.updateWidgetValue('resume_path', data['resume_path'])
    #         self.updateWidgetValue('resume_from', data['resume_from'])
    #         # self.saveCurrentProjectAsSettingsFile()

    # def onMediaStatusChanged(self, status):
    #     pass
    #     # try:
    #     #     if status == QMediaPlayer.MediaStatus.LoadedMedia:
    #     #         self.player.play()
    #     #     elif status in (QMediaPlayer.MediaStatus.NoMedia, QMediaPlayer.MediaStatus.InvalidMedia):
    #     #         print("Failed to load video")
    #     # except Exception as e:
    #     #     print(f"Error in media status change: {e}")
    def startBackendProcess(self):
        #params = {key: widget.value() for key, widget in self.params.items() if hasattr(widget, 'value')}
        # if not self.thread:
        self.statusLabel.setText("Rendering...")

        if self.params['resume_from_timestring']:
            self.params['max_frames'] += 1
            self.updateUIFromParams()

        self.thread = BackendThread(self.params)
        self.thread.imageGenerated.connect(self.updateImage)
        self.thread.finished.connect(self.playVideo)
        self.thread.finished.connect(self.cleanupThread)
        # self.thread.generateViz.connect(self.handleFinishedVizGen)
        self.thread.start()
    def stopBackendProcess(self):
        self.statusLabel.setText("Stopped")
        try:
            from deforum.shared_storage import models
            models["deforum_pipe"].gen.max_frames = len(models["deforum_pipe"].images)
        except:
            pass


    def startBatchProcess(self):
        self.statusLabel.setText("Batch Rendering...")

        if not self.current_job and self.job_queue:
            self.runNextJob()

    def runNextJob(self):
        if self.job_queue:

            self.current_job = self.job_queue.pop(0)
            try:
                self.current_job.markRunning()  # Mark the job as running
                # self.params.update(self.current_job.job_data)
                # self.updateUIFromParams()
                self.thread = BackendThread(self.current_job.job_data)
                self.thread.finished.connect(self.onJobFinished)
                self.thread.imageGenerated.connect(self.updateImage)
                # self.thread.finished.connect(self.playVideo)
                self.thread.start()
                self.statusLabel.setText(f"Batch Rendering. Job: {self.current_job.job_data['batch_name']} {len(self.job_queue)} Items left")

            except:
                self.current_job = None
                self.startBatchProcess()

    @Slot(dict)
    def onJobFinished(self, result):
        #print(f"Job completed with result: {result}")
        if self.current_job:
            self.current_job.markComplete()  # Mark the job as complete
            self.current_job = None  # Reset current job
            self.runNextJob()  # Run next job in the queue
        else:
            self.jobQueueList.clear()
            self.statusLabel.setText(
                f"Batch Render Complete")

    def stopBatchProcess(self):
        if self.current_job:
            # Logic to stop the current thread if it's running
            try:
                from deforum.shared_storage import models
                if 'deforum_pipe' in models:
                    models["deforum_pipe"].gen.max_frames = len(models["deforum_pipe"].images)
            except Exception as e:
                logger.info(repr(e))
        else:
            self.jobQueueList.clear()
        self.current_job = None
        self.statusLabel.setText(
            f"Batch Render Stopped")

    @Slot(dict)
    def updateImage(self, data):
        # Update the image on the label
        # Update progress bar if max_frames is defined
        if 'max_frames' in self.params and 'frame_idx' in data:
            max_frames = self.params['max_frames']
            frame_idx = data['frame_idx']
            if max_frames > 0:
                progress = (frame_idx + 1) / max_frames * 100
                self.progressBar.setValue(int(progress))
                max_frames = self.params['max_frames'] if not (self.params['resume_from_frame'] or self.params['resume_from_timestring']) else self.params['max_frames'] - self.params['resume_from']
                frame_idx = data['frame_idx'] if not (self.params['resume_from_frame'] or self.params['resume_from_timestring']) else data['frame_idx'] - self.params['resume_from']
                self.statusLabel.setText(
                    f"Rendering frame {frame_idx} of {max_frames}")

        if 'image' in data:
            img = copy.deepcopy(data['image'])
            qpixmap = npArrayToQPixmap(np.array(img).astype(np.uint8))  # Convert to QPixmap
            self.previewLabel.setPixmap(qpixmap)
            self.previewLabel.setScaledContents(True)
            if self.project is not None:
                self.project['frames'][str(data['frame_idx'])] = copy.deepcopy(self.params)
                # If there are subsequent frame parameters saved, load them
                if self.project_replay:
                    if data['frame_idx'] + 1 in self.project['frames']:
                        self.params = copy.deepcopy(self.project['frames'][str(data['frame_idx'] + 1)])
                        self.updateUIFromParams()  # Reflect parameter updates in UI

    def updateParam(self, key, value):
        super().updateParam(key, value)
        try:
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
                p = copy.deepcopy(self.params)
                _ = p.pop('max_frames')
                models["deforum_pipe"].live_update_from_kwargs(**p)
        except:
            pass

    def generateVizData(self):
        directory_path = os.path.join(config.root_path, 'temp_viz_images')
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Remove the file or link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Remove the directory recursively
            except Exception as e:
                logger.info('Failed to delete %s. Reason: %s' % (file_path, e))
        self.generateViz({'output_path':directory_path})


    def generateViz(self, data):
        if self.params['generate_viz']:

            if self.params['audio_path'] != "":
                milk = os.path.join(config.root_path, 'milks', self.params["milk_path"])
                # print(milk)
                self.vizGenThread = VisualGeneratorThread(self.params['audio_path'], data['output_path'], milk, self.params['fps'], self.params['width'], self.params['height'])
                self.vizGenThread.preset_path = milk
                self.vizGenThread.finished.connect(self.playVideo)
                self.vizGenThread.start()
    def cleanupVizThread(self):
        if hasattr(self, 'vizGenThread'):
            # self.vizGenThread.finished.disconnect(self.playVideo)
            self.vizGenThread.terminate()  # Ensure the thread has finished
            self.vizGenThread.deleteLater()  # Properly dispose of the thread object


    def update_animation(self):
        if self.listen_to_controls:
            if self.liveControlDock.joystick_enabled:
                self.liveControlDock.update_parameters_from_joystick()
            else:
                for axis, (decrease_key, increase_key) in self.axis_bindings.items():
                    if self.keys_pressed[increase_key]:
                        self.engine.update_parameter(axis, 'increase')
                    elif self.keys_pressed[decrease_key]:
                        self.engine.update_parameter(axis, 'decrease')
                    else:
                        # Gradually return to 0 if no key is pressed
                        current_value = self.engine.get_parameter(axis)
                        if current_value != 0:
                            direction = 'decrease' if current_value > 0 else 'increase'
                            self.engine.update_parameter(axis, direction)
                # print(f"{axis}: {self.engine.get_parameter(axis)}")
            update_params = {}
            for key, value in self.engine.parameters.items():
                update_params[key] = str(f"0: ({value})")
            from deforum.shared_storage import models
            try:
                self.fi = FrameInterpolator(models['deforum_pipe'].gen.max_frames, 1)
                models['deforum_pipe'].gen.keys.translation_x_series = self.fi.get_inbetweens(
                    self.fi.parse_key_frames(update_params['translation_x']))
                models['deforum_pipe'].gen.keys.translation_y_series = self.fi.get_inbetweens(
                    self.fi.parse_key_frames(update_params['translation_y']))
                models['deforum_pipe'].gen.keys.translation_z_series = self.fi.get_inbetweens(
                    self.fi.parse_key_frames(update_params['translation_z']))
                models['deforum_pipe'].gen.keys.rotation_3d_x_series = self.fi.get_inbetweens(
                    self.fi.parse_key_frames(update_params['rotation_3d_x']))
                models['deforum_pipe'].gen.keys.rotation_3d_y_series = self.fi.get_inbetweens(
                    self.fi.parse_key_frames(update_params['rotation_3d_y']))
                models['deforum_pipe'].gen.keys.rotation_3d_z_series = self.fi.get_inbetweens(
                    self.fi.parse_key_frames(update_params['rotation_3d_z']))
            except:
                pass
            if self.liveControlDock.isVisible():
                for axis, label in self.liveControlDock.status_labels.items():
                    label.setText(f"{axis}: {self.engine.get_parameter(axis):.2f}")
                    self.liveControlDock.update_status(axis, self.engine.get_parameter(axis))

                    # models['deforum_pipe'].live_update_from_kwargs(**update_params)

    def keyPressEvent(self, event: QKeyEvent):
        if self.listen_to_controls and not self.liveControlDock.joystick_enabled:
            key = event.text().upper()
            if key in self.keys_pressed:
                self.keys_pressed[key] = True
                axis = self.key_to_axis.get(key)
                if axis:
                    self.engine.set_key_held_down(axis, True)
                event.accept()  # Mark the event as handled
            else:
                super().keyPressEvent(event)  # Pass unhandled key events to the base class

    def keyReleaseEvent(self, event: QKeyEvent):
        if self.listen_to_controls and not self.liveControlDock.joystick_enabled:
            key = event.text().upper()
            if key in self.keys_pressed:
                self.keys_pressed[key] = False
                axis = self.key_to_axis.get(key)
                if axis:
                    self.engine.set_key_held_down(axis, False)
                event.accept()  # Mark the event as handled
            else:
                super().keyReleaseEvent(event)  # Pass unhandled key events to the base class