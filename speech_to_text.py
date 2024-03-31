#!/usr/bin/env python3
import streamlit as st
from whisper import Whisper

def speech_to_text(stt: Whisper, audio_bytes) -> str:
    audio_loc = "audio_file.wav"
    result = stt.transcribe(audio_loc)["text"]
    if result:
        return str(result)
    else:
        return "unable to convert to text"