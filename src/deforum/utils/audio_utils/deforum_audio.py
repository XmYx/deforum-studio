
def get_audio_duration(audio_file):
    import mutagen
    audio = mutagen.File(audio_file)
    return audio.info.length