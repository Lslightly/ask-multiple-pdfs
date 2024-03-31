import os
import streamlit as st
from TTS.api import TTS


def text_to_speech(tts: TTS, text: str, id: int):
    text += "."
    audio_loc = f"audio/ai_{id}.wav"
    if not os.path.exists(audio_loc):
        tts.tts_to_file(text, file_path=audio_loc)
    with open(audio_loc, "rb") as audiof:
        st.audio(audiof)
    
