"""
This module provides functions and classes to generate schedules for Deforum animation based on audio features.
It includes functions for filtering audio, applying Low-Frequency Oscillators (LFOs), and various feature extraction methods.
Additionally, it offers smoothing techniques to refine the extracted features.

Classes:
    DeforumAudioScheduleLab(QMainWindow): A PyQt application for tweaking Deforum schedules.

Functions:
    butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
        Apply a bandpass filter to the data.

    butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
        Apply a lowpass filter to the data.

    butter_highpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
        Apply a highpass filter to the data.

    apply_lfo(t: float, freq: float, amp: float, phase: float, lfo_type: str = 'sine') -> float:
        Apply a Low-Frequency Oscillator (LFO) to the data.

    moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
        Apply a moving average filter to the data.

    gaussian_smoothing(data: np.ndarray, window_size: int, sigma: float) -> np.ndarray:
        Apply Gaussian smoothing to the data.

    savitzky_golay_smoothing(data: np.ndarray, window_size: int, polyorder: int) -> np.ndarray:
        Apply Savitzky-Golay smoothing to the data.

    generate_deforum_schedule(
        y: np.ndarray, sr: int, direction: str = 'x', scale: float = 1.0, smoothness: float = 1.0,
        randomness: float = 0.0, offset: float = 0.0, mode: str = 'beat', filter_type: Optional[str] = None,
        lowcut: float = 300, highcut: float = 3000, lfo1: Optional[Tuple[float, float, float, str]] = None,
        lfo2: Optional[Tuple[float, float, float, str]] = None, lfo3: Optional[Tuple[float, float, float, str]] = None,
        smoothing_methods: Optional[List[Tuple[str, Dict[str, float]]]] = None, fps: int = 24) -> Dict[int, float]:
        Generate a schedule for Deforum animation based on audio features.

"""
import json
import sys
import numpy as np
import librosa
import random

from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, square, sawtooth, savgol_filter
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel,
    QHBoxLayout, QPushButton, QComboBox, QDoubleSpinBox, QCheckBox, QSpinBox, QFileDialog
)
from qtpy.QtCore import Qt, Signal, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from typing import List, Dict, Optional, Tuple

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: int, order: int = 5) -> np.ndarray:
    """
    Apply a bandpass filter to the data.

    Parameters:
        data (np.ndarray): Input audio data.
        lowcut (float): Lower cutoff frequency.
        highcut (float): Upper cutoff frequency.
        fs (int): Sampling rate.
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered audio data.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def butter_lowpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
    """
    Apply a lowpass filter to the data.

    Parameters:
        data (np.ndarray): Input audio data.
        cutoff (float): Cutoff frequency.
        fs (int): Sampling rate.
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered audio data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: int, order: int = 5) -> np.ndarray:
    """
    Apply a highpass filter to the data.

    Parameters:
        data (np.ndarray): Input audio data.
        cutoff (float): Cutoff frequency.
        fs (int): Sampling rate.
        order (int): Order of the filter.

    Returns:
        np.ndarray: Filtered audio data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y


def apply_lfo(t: float, freq: float, amp: float, phase: float, lfo_type: str = 'sine') -> float:
    """
    Apply a Low-Frequency Oscillator (LFO) to the data.

    Parameters:
        t (float): Time value.
        freq (float): Frequency of the LFO.
        amp (float): Amplitude of the LFO.
        phase (float): Phase of the LFO.
        lfo_type (str): Type of the LFO ('sine', 'square', 'sawtooth', 'triangle').

    Returns:
        float: LFO value.
    """
    if lfo_type == 'sine':
        return amp * np.sin(2 * np.pi * freq * t + phase)
    elif lfo_type == 'square':
        return amp * square(2 * np.pi * freq * t + phase)
    elif lfo_type == 'sawtooth':
        return amp * sawtooth(2 * np.pi * freq * t + phase)
    elif lfo_type == 'triangle':
        return amp * sawtooth(2 * np.pi * freq * t + phase, width=0.5)
    else:
        return amp * np.sin(2 * np.pi * freq * t + phase)  # default to sine wave

def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize the data to the range [0, 1].

    Parameters:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Normalized data.
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def invert(data: np.ndarray) -> np.ndarray:
    """
    Invert the data.

    Parameters:
        data (np.ndarray): Input data.

    Returns:
        np.ndarray: Inverted data.
    """
    return -data
def ema_smoothing(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply Exponential Moving Average (EMA) smoothing to the data.

    Parameters:
        data (np.ndarray): Input data.
        window_size (int): Window size for the EMA filter.

    Returns:
        np.ndarray: Smoothed data.
    """
    alpha = 2 / (window_size + 1)
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    return ema

def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Apply a moving average filter to the data.

    Parameters:
        data (np.ndarray): Input data.
        window_size (int): Window size for the moving average.

    Returns:
        np.ndarray: Smoothed data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')


def gaussian_smoothing(data: np.ndarray, window_size: int, sigma: float) -> np.ndarray:
    """
    Apply Gaussian smoothing to the data.

    Parameters:
        data (np.ndarray): Input data.
        window_size (int): Window size for the Gaussian filter.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        np.ndarray: Smoothed data.
    """
    return gaussian_filter1d(data, sigma=sigma)


def savitzky_golay_smoothing(data: np.ndarray, window_size: int, polyorder: int) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to the data.

    Parameters:
        data (np.ndarray): Input data.
        window_size (int): Window size for the Savitzky-Golay filter.
        polyorder (int): Polynomial order for the Savitzky-Golay filter.

    Returns:
        np.ndarray: Smoothed data.
    """
    return savgol_filter(data, window_size, polyorder)


def generate_deforum_schedule(
        y: np.ndarray, sr: int, direction: str = 'x', scale: float = 1.0, smoothness: float = 1.0,
        randomness: float = 0.0, offset: float = 0.0, mode: str = 'beat', filter_type: Optional[str] = None,
        lowcut: float = 300, highcut: float = 3000, lfo1: Optional[Tuple[float, float, float, str]] = None,
        lfo2: Optional[Tuple[float, float, float, str]] = None, lfo3: Optional[Tuple[float, float, float, str]] = None,
        smoothing_methods: Optional[List[Tuple[str, Dict[str, float]]]] = None, fps: int = 24,
        invert_values:bool = False, normalize_values:bool = False, absolute_values:bool = False) -> Dict[int, float]:
    """
    Generate a schedule for Deforum animation based on audio features.

    Parameters:
        y (np.ndarray): Input audio data.
        sr (int): Sampling rate.
        direction (str): Direction of motion ('x' or 'y').
        scale (float): Scale factor for the motion.
        smoothness (float): Smoothness factor for the motion.
        randomness (float): Randomness factor for the motion.
        offset (float): Offset for the motion.
        mode (str): Feature extraction mode.
        filter_type (Optional[str]): Type of filter to apply ('harmonic', 'percussive', 'lowpass', 'highpass', 'bandpass').
        lowcut (float): Lower cutoff frequency for the filter.
        highcut (float): Upper cutoff frequency for the filter.
        lfo1 (Optional[Tuple[float, float, float, str]]): Parameters for the first LFO (frequency, amplitude, phase, type).
        lfo2 (Optional[Tuple[float, float, float, str]]): Parameters for the second LFO (frequency, amplitude, phase, type).
        lfo3 (Optional[Tuple[float, float, float, str]]): Parameters for the third LFO (frequency, amplitude, phase, type).
        smoothing_methods (Optional[List[Tuple[str, Dict[str, float]]]]): List of smoothing methods to apply.
        fps (int): Frames per second for the animation.

    Returns:
        Dict[int, float]: Generated schedule for Deforum animation.
    """
    # Apply harmonic/percussive separation or filtering if specified
    if filter_type == 'harmonic':
        y = librosa.effects.harmonic(y)
    elif filter_type == 'percussive':
        y = librosa.effects.percussive(y)
    elif filter_type == 'lowpass':
        y = butter_lowpass_filter(y, cutoff=highcut, fs=sr)
    elif filter_type == 'highpass':
        y = butter_highpass_filter(y, cutoff=highcut, fs=sr)
    elif filter_type == 'bandpass':
        y = butter_bandpass_filter(y, lowcut=lowcut, highcut=highcut, fs=sr)

    y = np.nan_to_num(y)

    # Select feature extraction method
    if mode == 'beat':
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        event_times = librosa.frames_to_time(beats, sr=sr)
        event_values = [y[int(time * sr)] for time in event_times]
    elif mode == 'onset':
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        event_times = librosa.frames_to_time(onset_frames, sr=sr)
        event_values = onset_env[onset_frames]

    elif mode == 'amplitude':
        envelope = librosa.onset.onset_strength(y=y, sr=sr)
        event_frames = np.arange(len(envelope))
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = envelope
    elif mode == 'spectral_centroid':
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        event_frames = np.arange(len(spectral_centroids))
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = spectral_centroids
    elif mode == 'spectral_bandwidth':
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        event_frames = np.arange(len(spectral_bandwidth))
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = spectral_bandwidth
    elif mode == 'spectral_contrast':
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        event_frames = np.arange(spectral_contrast.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = spectral_contrast.mean(axis=0)
    elif mode == 'spectral_flatness':
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        event_frames = np.arange(len(spectral_flatness))
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = spectral_flatness
    elif mode == 'zero_crossing_rate':
        zero_crossings = librosa.feature.zero_crossing_rate(y=y)[0]
        event_frames = np.arange(len(zero_crossings))
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = zero_crossings
    elif mode == 'rms':
        rms = librosa.feature.rms(y=y)[0]
        event_frames = np.arange(len(rms))
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = rms
    elif mode == 'mfcc':
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        event_frames = np.arange(mfccs.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = mfccs.mean(axis=0)
    elif mode == 'chroma':
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        event_frames = np.arange(chroma.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = chroma.mean(axis=0)
    elif mode == 'tonnetz':
        tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
        event_frames = np.arange(tonnetz.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = tonnetz.mean(axis=0)
    elif mode == 'tempogram':
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
        event_frames = np.arange(tempogram.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = tempogram.mean(axis=0)
    elif mode == 'mel_spectrogram':
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        event_frames = np.arange(mel_spectrogram.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = mel_spectrogram.mean(axis=0)
    elif mode == 'cqt':
        cqt = librosa.core.cqt(y=y, sr=sr)
        event_frames = np.arange(cqt.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = np.abs(cqt).mean(axis=0)
    elif mode == 'stft':
        stft = librosa.stft(y)
        event_frames = np.arange(stft.shape[1])
        event_times = librosa.frames_to_time(event_frames, sr=sr)
        event_values = np.abs(stft).mean(axis=0)
    elif mode == 'fft':
        # Perform FFT and consider only the positive frequencies
        # Perform FFT and consider only the positive frequencies
        fft = np.fft.fft(y)
        fft_magnitude = np.abs(fft[:len(fft) // 2])  # Only take the positive frequencies
        freqs = np.fft.fftfreq(len(fft), 1 / sr)[:len(fft) // 2]

        # Generate time values based on the desired fps
        fps = fps
        total_duration = len(y) / sr
        frame_times = np.linspace(0, total_duration, int(total_duration * fps))

        # Interpolate FFT values to match the frame times
        # Scale the fft_magnitude to match the time length
        fft_time_values = np.linspace(0, total_duration, len(fft_magnitude))
        event_values = np.interp(frame_times, fft_time_values, fft_magnitude * freqs)
        event_times = frame_times


    if normalize_values:
        event_values = normalize(event_values)
    # Initialize the schedule dictionary
    schedule = {}

    # Generate values for each event
    for event_time, event_value in zip(event_times, event_values):
        frame_number = int(event_time * fps)  # Convert to frame number, assuming 30 FPS
        t = frame_number / fps  # Convert frame number to time in seconds

        # Calculate the value based on amplitude, scale, and randomness
        value = event_value * scale + random.gauss(0, randomness)

        # Apply LFOs if they are defined
        if lfo1 is not None:
            freq, amp, phase, lfo1_type = lfo1
            value += apply_lfo(t, freq, amp, phase, lfo1_type)
        if lfo2 is not None:
            freq, amp, phase, lfo2_type = lfo2
            value += apply_lfo(t, freq, amp, phase, lfo2_type)
        if lfo3 is not None:
            freq, amp, phase, lfo3_type = lfo3
            value += apply_lfo(t, freq, amp, phase, lfo3_type)

        # Add the value to the schedule
        schedule[frame_number] = value
    # Apply smoothing
    if smoothing_methods:
        frames = np.array(list(schedule.keys()))
        values = np.array(list(schedule.values()))

        for method, params in smoothing_methods:
            if method == 'moving_average':
                values = moving_average(values, params['window_size'])
            elif method == 'gaussian':
                values = gaussian_smoothing(values, params['window_size'], params['sigma'])
            elif method == 'savitzky_golay':
                values = savitzky_golay_smoothing(values, params['window_size'], params['polyorder'])
            elif method == 'ema':
                values = ema_smoothing(values, params['window_size'])
        schedule = dict(zip(frames, values))

    for frame, value in schedule.items():

        v = -value if invert_values else value
        v = abs(v) if absolute_values else v

        schedule[frame] = v + offset

    return schedule



def load_dviz_file(file_path: str) -> dict:
    """
    Load parameters from a .dviz file.

    Args:
        file_path (str): Path to the .dviz file.

    Returns:
        dict: The parameters loaded from the .dviz file.
    """
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params

def generate_schedule_from_dviz(mp3_file: str, dviz_file: str, fps: int = 24) -> Dict[int, float]:
    """
    Generate a Deforum schedule based on the parameters from a .dviz file.

    Args:
        mp3_file (str): Path to the MP3 file to be loaded.
        dviz_file (str): Path to the .dviz file containing parameters.

    Returns:
        Dict[int, float]: The generated Deforum schedule.
    """
    # Load the audio file
    y, sr = librosa.load(mp3_file, sr=None)

    # Load parameters from the .dviz file
    params = load_dviz_file(dviz_file)

    # Extract parameters
    sliders = params.get('sliders', {})
    lfo_controls = params.get('lfo_controls', {})
    combo_boxes = params.get('combo_boxes', {})
    smoothing_methods = params.get('smoothing_methods', [])

    # Prepare LFOs
    lfo1 = (
        lfo_controls['lfo1']['freq'], lfo_controls['lfo1']['amp'],
        lfo_controls['lfo1']['phase'], lfo_controls['lfo1']['type']
    ) if lfo_controls.get('lfo1', {}).get('enabled', False) else None
    lfo2 = (
        lfo_controls['lfo2']['freq'], lfo_controls['lfo2']['amp'],
        lfo_controls['lfo2']['phase'], lfo_controls['lfo2']['type']
    ) if lfo_controls.get('lfo2', {}).get('enabled', False) else None
    lfo3 = (
        lfo_controls['lfo3']['freq'], lfo_controls['lfo3']['amp'],
        lfo_controls['lfo3']['phase'], lfo_controls['lfo3']['type']
    ) if lfo_controls.get('lfo3', {}).get('enabled', False) else None

    # Prepare smoothing methods
    smoothing_methods_list = [
        (method['type'], {
            'window_size': method['window_size'],
            'sigma': method['sigma'],
            'polyorder': method['polyorder']
        }) for method in smoothing_methods
    ]

    # Generate the schedule
    schedule = generate_deforum_schedule(
        y=y,
        sr=sr,
        direction='x',
        scale=sliders.get('scale', 100) / 10.0,
        smoothness=sliders.get('smoothness', 2),
        randomness=sliders.get('randomness', 0) / 10.0,
        offset=sliders.get('offset', 0),
        mode=combo_boxes.get('mode', 'beat'),
        filter_type=combo_boxes.get('filter_type', None),
        lowcut=sliders.get('lowcut', 300),
        highcut=sliders.get('highcut', 3000),
        lfo1=lfo1,
        lfo2=lfo2,
        lfo3=lfo3,
        smoothing_methods=smoothing_methods_list,
        fps=fps  # Default FPS value
    )

    return schedule

class UpdateSignal(QObject):
    """
    A custom signal class for updating the plot in the DeforumAudioScheduleLab.
    """
    update = Signal()

class DeforumAudioScheduleLab(QMainWindow):
    """
    A PyQt application for tweaking Deforum schedules based on audio features.

    Attributes:
        y (np.ndarray): Audio time series.
        sr (int): Sampling rate of y.
        sliders (dict): Dictionary of QSliders for various parameters.
        lfo_controls (dict): Dictionary of LFO controls.
        combo_boxes (dict): Dictionary of QComboBox for mode and filter_type.
        smoothing_methods (list): List of tuples containing smoothing methods and their parameters.
        update_signal (UpdateSignal): Custom signal for updating the plot.
        schedule (dict): Dictionary containing the generated schedule.
        figure (Figure): Matplotlib figure for plotting.
        canvas (FigureCanvas): Matplotlib canvas for the figure.
        smoothing_list_layout (QVBoxLayout): Layout for the list of smoothing methods.

    Methods:
        __init__(mp3_file: str): Initialize the DeforumAudioScheduleLab.
        initUI(): Initialize the user interface.
        add_smoothing(): Add a smoothing method to the list.
        remove_smoothing(smoothing_layout: QHBoxLayout): Remove a smoothing method from the list.
        create_labeled_combobox(label: str, combo_box: QComboBox) -> QHBoxLayout: Create a labeled QComboBox.
        create_labeled_spinbox(label: str, spin_box: QDoubleSpinBox) -> QHBoxLayout: Create a labeled QDoubleSpinBox.
        refresh_visualization(): Refresh the visualization by recalculating the schedule.
        calculate_schedule() -> dict: Calculate the schedule based on the current parameters and smoothing methods.
        update_plot(): Update the plot with the current schedule.
    """
    def __init__(self):
        """
        Initialize the DeforumAudioScheduleLab.

        Args:
            mp3_file (str): Path to the MP3 file to be loaded.
        """
        super().__init__()
        self.setWindowTitle('Deforum Schedule Tweaker')
        # Prompt the user to select an MP3 file
        options = QFileDialog.Option.DontUseNativeDialog
        mp3_file, _ = QFileDialog.getOpenFileName(self, "Open MP3 File", "", "MP3 Files (*.mp3);;All Files (*)", options=options)
        if not mp3_file:
            sys.exit("No file selected. Exiting.")
        # Load and cache the audio file
        self.y, self.sr = librosa.load(mp3_file, sr=None)

        # Initialize the UI
        self.initUI()

        # Create a signal for updating the plot
        self.update_signal = UpdateSignal()
        self.update_signal.update.connect(self.update_plot)

        # Initial schedule calculation
        self.schedule = self.calculate_schedule()

        # Initial plot update
        self.update_plot()

    def initUI(self):
        """
        Initialize the user interface.
        """
        # Create a central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add Matplotlib canvas
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Create sliders
        self.sliders = {}
        params = {
            'scale': (1, 200, 100),
            'smoothness': (1, 100, 2),
            'randomness': (0, 500, 0),
            'offset': (-500, 500, 0),
            'lowcut': (1, 20000, 300),
            'highcut': (1, 20000, 3000)
        }
        for param, (min_val, max_val, default_val) in params.items():
            slider_layout = QHBoxLayout()
            label = QLabel(f'{param.capitalize()}:')
            slider_layout.addWidget(label)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            slider.setObjectName(param)
            slider_layout.addWidget(slider)
            layout.addLayout(slider_layout)
            self.sliders[param] = slider

        # Add LFO controls with checkboxes and combo boxes for LFO type
        self.lfo_controls = {}
        lfo_types = ['sine', 'square', 'sawtooth', 'triangle']
        for i in range(1, 4):
            lfo_layout = QVBoxLayout()
            checkbox = QCheckBox(f'Enable LFO{i}')
            lfo_layout.addWidget(checkbox)
            self.lfo_controls[f'lfo{i}_enabled'] = checkbox
            for param in ['freq', 'amp', 'phase']:
                spin_box = QDoubleSpinBox()
                spin_box.setDecimals(3)
                spin_box.setRange(-100.0, 100.0 if param != 'phase' else 2 * np.pi)
                spin_box.setSingleStep(0.1)
                spin_box.setValue(0.1 if param == 'freq' else (10.0 if param == 'amp' else 0.0))
                lfo_layout.addLayout(self.create_labeled_spinbox(f'LFO{i} {param.capitalize()}: ', spin_box))
                self.lfo_controls[f'lfo{i}_{param}'] = spin_box

            lfo_type_combo = QComboBox()
            lfo_type_combo.addItems(lfo_types)
            lfo_layout.addLayout(self.create_labeled_combobox(f'LFO{i} Type: ', lfo_type_combo))
            self.lfo_controls[f'lfo{i}_type'] = lfo_type_combo

            layout.addLayout(lfo_layout)


        # Add combo boxes for mode and filter_type
        self.combo_boxes = {}
        options = {
            'mode': ['beat', 'onset', 'amplitude', 'spectral_centroid', 'spectral_bandwidth', 'spectral_contrast',
                     'spectral_flatness', 'zero_crossing_rate', 'rms', 'mfcc', 'chroma', 'tonnetz', 'tempogram',
                     'mel_spectrogram', 'cqt', 'stft', 'fft'],
            'filter_type': [None, 'harmonic', 'percussive', 'lowpass', 'highpass', 'bandpass']
        }
        for param, items in options.items():
            combo_box = QComboBox()
            combo_box.addItems([str(item) for item in items])
            combo_box.setCurrentText(str(items[0]))
            combo_box.setObjectName(param)
            layout.addWidget(QLabel(f'{param.capitalize()}:'))
            layout.addWidget(combo_box)
            self.combo_boxes[param] = combo_box
        # Add smoothing list
        self.smoothing_methods = []
        self.smoothing_list_layout = QVBoxLayout()
        layout.addLayout(self.smoothing_list_layout)
        self.add_smoothing_button = QPushButton('Add Smoothing')
        self.add_smoothing_button.clicked.connect(self.add_smoothing)
        layout.addWidget(self.add_smoothing_button)
        # Add refresh button
        self.refresh_button = QPushButton('Refresh Visualization')
        self.refresh_button.clicked.connect(self.refresh_visualization)
        layout.addWidget(self.refresh_button)

        self.normalize_checkbox = QCheckBox('Normalize')
        layout.addWidget(self.normalize_checkbox)
        self.invert_checkbox = QCheckBox('Invert')
        layout.addWidget(self.invert_checkbox)
        self.absolute_checkbox = QCheckBox('Absolute')
        layout.addWidget(self.absolute_checkbox)


        self.save_button = QPushButton('Save Parameters')
        self.save_button.clicked.connect(self.save_params)
        layout.addWidget(self.save_button)

        self.load_button = QPushButton('Load Parameters')
        self.load_button.clicked.connect(self.load_params)
        layout.addWidget(self.load_button)

    def add_smoothing(self):
        """
        Add a smoothing method to the list.
        """
        smoothing_layout = QHBoxLayout()
        smoothing_type_combo = QComboBox()
        smoothing_type_combo.addItems(['moving_average', 'gaussian', 'savitzky_golay', 'ema'])
        smoothing_layout.addWidget(smoothing_type_combo)

        window_size_spin = QSpinBox()
        window_size_spin.setRange(1, 500)
        window_size_spin.setValue(5)
        smoothing_layout.addWidget(window_size_spin)

        sigma_spin = QDoubleSpinBox()
        sigma_spin.setDecimals(3)
        sigma_spin.setRange(0.1, 100.0)
        sigma_spin.setValue(1.0)
        smoothing_layout.addWidget(sigma_spin)

        polyorder_spin = QSpinBox()
        polyorder_spin.setRange(1, 50)
        polyorder_spin.setValue(3)
        smoothing_layout.addWidget(polyorder_spin)

        remove_button = QPushButton('Remove')
        remove_button.clicked.connect(lambda: self.remove_smoothing(smoothing_layout))
        smoothing_layout.addWidget(remove_button)

        self.smoothing_list_layout.addLayout(smoothing_layout)
        self.smoothing_methods.append((smoothing_type_combo, window_size_spin, sigma_spin, polyorder_spin))

    def remove_smoothing(self, smoothing_layout: QHBoxLayout):
        """
        Remove a smoothing method from the list.

        Args:
            smoothing_layout (QHBoxLayout): The layout of the smoothing method to be removed.
        """
        for i in reversed(range(smoothing_layout.count())):
            widget = smoothing_layout.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        self.smoothing_list_layout.removeItem(smoothing_layout)
        self.smoothing_methods = [
            (s, w, g, p) for s, w, g, p in self.smoothing_methods if s.parent() is not None
        ]

    def create_labeled_combobox(self, label: str, combo_box: QComboBox) -> QHBoxLayout:
        """
        Create a labeled QComboBox.

        Args:
            label (str): The label text.
            combo_box (QComboBox): The QComboBox widget.

        Returns:
            QHBoxLayout: The layout containing the labeled QComboBox.
        """
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(label))
        hbox.addWidget(combo_box)
        return hbox
    def create_labeled_spinbox(self, label: str, spin_box: QDoubleSpinBox) -> QHBoxLayout:
        """
        Create a labeled QDoubleSpinBox.

        Args:
            label (str): The label text.
            spin_box (QDoubleSpinBox): The QDoubleSpinBox widget.

        Returns:
            QHBoxLayout: The layout containing the labeled QDoubleSpinBox.
        """
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel(label))
        hbox.addWidget(spin_box)
        return hbox
    def refresh_visualization(self):
        """
        Refresh the visualization by recalculating the schedule.
        """
        self.schedule = self.calculate_schedule()
        self.update_signal.update.emit()

    def calculate_schedule(self):
        """
        Calculate the schedule based on the current parameters and smoothing methods.

        Returns:
            dict: The generated schedule.
        """
        scale = self.sliders['scale'].value() / 10.0
        smoothness = self.sliders['smoothness'].value()
        randomness = self.sliders['randomness'].value() / 10.0
        offset = self.sliders['offset'].value()
        lowcut = self.sliders['lowcut'].value()
        highcut = self.sliders['highcut'].value()
        mode = self.combo_boxes['mode'].currentText()
        filter_type = self.combo_boxes['filter_type'].currentText()
        filter_type = None if filter_type == 'None' else filter_type

        lfo1 = None if not self.lfo_controls['lfo1_enabled'].isChecked() else (
            self.lfo_controls['lfo1_freq'].value(), self.lfo_controls['lfo1_amp'].value(), self.lfo_controls['lfo1_phase'].value(), self.lfo_controls['lfo1_type'].currentText()
        )
        lfo2 = None if not self.lfo_controls['lfo2_enabled'].isChecked() else (
            self.lfo_controls['lfo2_freq'].value(), self.lfo_controls['lfo2_amp'].value(), self.lfo_controls['lfo2_phase'].value(), self.lfo_controls['lfo2_type'].currentText()
        )
        lfo3 = None if not self.lfo_controls['lfo3_enabled'].isChecked() else (
            self.lfo_controls['lfo3_freq'].value(), self.lfo_controls['lfo3_amp'].value(), self.lfo_controls['lfo3_phase'].value(), self.lfo_controls['lfo3_type'].currentText()
        )

        smoothing_methods = [
            (s.currentText(), {'window_size': w.value(), 'sigma': g.value(), 'polyorder': p.value()})
            for s, w, g, p in self.smoothing_methods
        ]

        schedule = generate_deforum_schedule(
            self.y, self.sr, direction='x', scale=scale, smoothness=smoothness, randomness=randomness,
            offset=offset, mode=mode, filter_type=filter_type, lowcut=lowcut, highcut=highcut,
            lfo1=lfo1, lfo2=lfo2, lfo3=lfo3, smoothing_methods=smoothing_methods, invert_values=self.invert_checkbox.isChecked(),
            normalize_values=self.normalize_checkbox.isChecked(), absolute_values=self.absolute_checkbox.isChecked()
        )
        return schedule

    def update_plot(self):
        """
        Update the plot with the current schedule.
        """
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_title('Deforum Schedule')
        ax.set_xlabel('Frame')
        ax.set_ylabel('Value')

        frames = list(self.schedule.keys())
        values = list(self.schedule.values())

        ax.plot(frames, values, marker='o')
        self.canvas.draw()
    def save_params(self):
        """
        Save current parameters to a .dviz file.
        """
        options = QFileDialog.Option.DontUseNativeDialog
        file, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "DVIZ Files (*.dviz)", options=options)
        if file:
            if not file.endswith('.dviz'):
                file += '.dviz'
            params = self.serialize_params()
            with open(file, 'w') as f:
                json.dump(params, f, indent=4)

    def load_params(self):
        """
        Load parameters from a .dviz file.
        """
        options = QFileDialog.Option.DontUseNativeDialog
        file, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "DVIZ Files (*.dviz)", options=options)
        if file:
            with open(file, 'r') as f:
                params = json.load(f)
                self.deserialize_params(params)

    def serialize_params(self) -> dict:
        """
        Serialize current parameters to a dictionary.

        Returns:
            dict: The serialized parameters.
        """
        params = {
            'sliders': {key: slider.value() for key, slider in self.sliders.items()},
            'lfo_controls': {
                f'lfo{i}': {
                    'enabled': self.lfo_controls[f'lfo{i}_enabled'].isChecked(),
                    'freq': self.lfo_controls[f'lfo{i}_freq'].value(),
                    'amp': self.lfo_controls[f'lfo{i}_amp'].value(),
                    'phase': self.lfo_controls[f'lfo{i}_phase'].value(),
                    'type': self.lfo_controls[f'lfo{i}_type'].currentText()
                } for i in range(1, 4)
            },
            'combo_boxes': {key: combo.currentText() for key, combo in self.combo_boxes.items()},
            'smoothing_methods': [
                {
                    'type': s.currentText(),
                    'window_size': w.value(),
                    'sigma': g.value(),
                    'polyorder': p.value()
                } for s, w, g, p in self.smoothing_methods
            ],
            'normalize': self.normalize_checkbox.isChecked(),
            'invert': self.invert_checkbox.isChecked(),
            'absolute': self.absolute_checkbox.isChecked()
        }
        return params

    def deserialize_params(self, params: dict):
        """
        Deserialize parameters from a dictionary.

        Args:
            params (dict): The dictionary containing parameters.
        """
        for key, value in params['sliders'].items():
            self.sliders[key].setValue(value)

        for i in range(1, 4):
            self.lfo_controls[f'lfo{i}_enabled'].setChecked(params['lfo_controls'][f'lfo{i}']['enabled'])
            self.lfo_controls[f'lfo{i}_freq'].setValue(params['lfo_controls'][f'lfo{i}']['freq'])
            self.lfo_controls[f'lfo{i}_amp'].setValue(params['lfo_controls'][f'lfo{i}']['amp'])
            self.lfo_controls[f'lfo{i}_phase'].setValue(params['lfo_controls'][f'lfo{i}']['phase'])
            self.lfo_controls[f'lfo{i}_type'].setCurrentText(params['lfo_controls'][f'lfo{i}']['type'])

        for key, value in params['combo_boxes'].items():
            self.combo_boxes[key].setCurrentText(value)

        for _ in range(len(self.smoothing_methods)):
            self.remove_smoothing(self.smoothing_list_layout.itemAt(0))

        for smoothing in params['smoothing_methods']:
            self.add_smoothing()
            self.smoothing_methods[-1][0].setCurrentText(smoothing['type'])
            self.smoothing_methods[-1][1].setValue(smoothing['window_size'])
            self.smoothing_methods[-1][2].setValue(smoothing['sigma'])
            self.smoothing_methods[-1][3].setValue(smoothing['polyorder'])

        self.normalize_checkbox.setChecked(params.get('normalize', False))
        self.invert_checkbox.setChecked(params.get('invert', False))
        self.absolute_checkbox.setChecked(params.get('absolute', False))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = DeforumAudioScheduleLab()
    main_window.show()
    sys.exit(app.exec())
