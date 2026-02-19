import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
from openai import OpenAI
import numpy as np
import av
import tempfile

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Live English ↔ Spanish Translator")

class AudioProcessor(AudioProcessorBase):

    def recv(self, frame):

        audio = frame.to_ndarray()

        temp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        import scipy.io.wavfile as wav
        wav.write(temp.name, 16000, audio)

        transcript = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=open(temp.name,"rb")
        )

        text = transcript.text

        translated = client.responses.create(
            model="gpt-4o-mini",
            input=f"Translate English to Spanish or Spanish to English: {text}"
        )

        translated_text = translated.output_text

        speech = client.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice="alloy",
            input=translated_text
        )

        st.write("Original:", text)
        st.write("Translated:", translated_text)

        return frame


webrtc_streamer(

    key="translator",

    mode=WebRtcMode.SENDRECV,

    audio_processor_factory=AudioProcessor,

    media_stream_constraints={"audio":True,"video":False},

)
