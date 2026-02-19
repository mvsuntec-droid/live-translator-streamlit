import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    WebRtcMode,
    AudioProcessorBase,
    RTCConfiguration,
)
import av
import numpy as np
import tempfile
import scipy.io.wavfile as wav
from openai import OpenAI
import base64
import os

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Live Meeting Translator",
    layout="centered"
)

st.title("🌍 Live English ↔ Spanish Translator")

st.info("Allow microphone access and start speaking.")

# -------------------------
# OPENAI CLIENT
# -------------------------

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# -------------------------
# WEBRTC CONFIG (Production Safe)
# -------------------------

RTC_CONFIGURATION = RTCConfiguration({

    "iceServers": [

        {"urls": ["stun:stun.l.google.com:19302"]},

        {"urls": ["stun:stun1.l.google.com:19302"]},

        {"urls": ["stun:stun2.l.google.com:19302"]},

    ]
})


# -------------------------
# AUDIO PROCESSOR
# -------------------------

class Translator(AudioProcessorBase):

    def recv(self, frame):

        audio = frame.to_ndarray()

        temp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        )

        wav.write(
            temp.name,
            16000,
            audio
        )

        try:

            # Speech to Text

            transcript = client.audio.transcriptions.create(

                model="gpt-4o-mini-transcribe",

                file=open(temp.name, "rb")

            )

            original = transcript.text

            if len(original.strip()) < 2:
                return frame

            st.write("🗣 Original:", original)

            # Translation

            translation = client.responses.create(

                model="gpt-4o-mini",

                input=f"""
Translate this speech.

If English → Spanish
If Spanish → English

Speech:
{original}
"""
            )

            translated = translation.output_text

            st.write("🌎 Translated:", translated)

            # Text to Speech

            speech = client.audio.speech.create(

                model="gpt-4o-mini-tts",

                voice="alloy",

                input=translated

            )

            audio_bytes = speech

            audio_base64 = base64.b64encode(audio_bytes).decode()

            audio_html = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}">
            </audio>
            """

            st.markdown(audio_html, unsafe_allow_html=True)

        except Exception as e:

            st.error(e)

        return frame


# -------------------------
# START WEBRTC
# -------------------------

webrtc_streamer(

    key="live-translator",

    mode=WebRtcMode.SENDONLY,

    rtc_configuration=RTC_CONFIGURATION,

    audio_processor_factory=Translator,

    media_stream_constraints={

        "audio": True,

        "video": False,

    },

)
