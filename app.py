import streamlit as st
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav
from openai import OpenAI
import base64

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Live English ↔ Spanish Translator")

duration = st.slider("Recording duration", 1, 10, 5)

if st.button("Start Speaking"):

    st.write("Listening...")

    fs = 16000

    recording = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1
    )

    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    wav.write(
        temp_file.name,
        fs,
        recording
    )

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=open(temp_file.name,"rb")
    )

    text = transcript.text

    st.write("You said:", text)

    translated = client.responses.create(
        model="gpt-4o-mini",
        input=f"Translate English to Spanish or Spanish to English: {text}"
    )

    translated_text = translated.output_text

    st.write("Translated:", translated_text)

    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=translated_text
    )

    audio_bytes = speech

    st.audio(audio_bytes)
