import streamlit as st
from openai import OpenAI
import tempfile

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Live English ↔ Spanish Translator")

audio_file = st.file_uploader("Record or upload audio", type=["wav","mp3","m4a"])

if audio_file:

    st.audio(audio_file)

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file
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

    st.audio(speech)
