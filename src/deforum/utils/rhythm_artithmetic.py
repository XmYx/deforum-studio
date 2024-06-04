from typing import Any, List


def frame_to_beat(frame: int, fps: float, bpm: float) -> float:
    return frame / ((fps * 60) / bpm)

def frame_to_sec(frame: int, fps: float) -> float:
    return frame / fps

def sec_to_frame(sec: float, fps: float) -> int:
    return int(sec_to_frame_exact(sec, fps))

def sec_to_frame_exact(sec: float, fps: float) -> float:
    return sec * fps

def beat_to_frame(beat: float, fps: float, bpm: float) -> int:
    return int(beat_to_frame_exact(beat, fps, bpm))

def beat_to_frame_exact(beat: float, fps: float, bpm: float) -> float:
    return beat * ((fps * 60) / bpm)

def beat_to_sec(beat: float, bpm: float) -> float:
    return beat / bpm * 60

def sec_to_beat(sec: float, bpm: float) -> float:
    return sec * bpm / 60

def frames_per_beat(fps: float, bpm: float) -> float:
    return fps / (bpm / 60)

def beats_per_frames(fps: float, bpm: float) -> float:
    return 1 / frames_per_beat(fps, bpm)

def count_until_criteria(lst : List[Any], criteria) -> int:
    count = 0
    for item in lst:
        if criteria(item):
            break
        count += 1
    return count