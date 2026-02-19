import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import numpy as np
import tempfile
import scipy.io.wavfile as wav
from openai import OpenAI

# ------------------------

st.set_page_config(page_title="Live Translator")

st.title("🎤 Live Meeting Translator")

st.write("Click START and speak")

# ------------------------

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)

# ------------------------

class AudioProcessor:

    def recv(self, frame):

        audio = frame.to_ndarray()

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

        wav.write(temp.name, 16000, audio)

        try:

            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=open(temp.name, "rb"),
            )

            text = transcript.text

            if text:

                st.write("You:", text)

                translation = client.responses.create(
                    model="gpt-4o-mini",
                    input=f"Translate to Spanish: {text}",
                )

                st.write("Translated:", translation.output_text)

        except Exception as e:

            st.error(e)

        return frame


# ------------------------

webrtc_streamer(

    key="speech",

    mode=WebRtcMode.SENDONLY,

    rtc_configuration=RTC_CONFIGURATION,

    audio_processor_factory=AudioProcessor,

    media_stream_constraints={

        "audio": True,

        "video": False,

    },
)
