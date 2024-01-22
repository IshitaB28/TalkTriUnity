import streamlit as st
import sys
import subprocess

from clarifai.client.model import Model
# from pydub import AudioSegment
# from scipy.io import wavfile
# from pathlib import PurePath
import numpy as np
from dotenv import load_dotenv
import streamlit as st
import wave
import io
from audio_recorder_streamlit import audio_recorder
from scipy.signal import resample
import os
import cv2


clarifai_pat =  os.getenv('CLARIFAI_PAT')

load_dotenv()

clarifai_pat =  os.getenv('CLARIFAI_PAT')



def get_sample_rate(audio_bytes):
    with wave.open(io.BytesIO(audio_bytes), 'rb') as wave_file:
        print(wave_file)
        sample_rate = wave_file.getframerate()
    return sample_rate

# with open(wav_file, "rb") as f:
#     file_bytes = f.read()

def get_video_input():
    vid = cv2.VideoCapture(0)

def whisper(audio_bytes):
    samplerate = get_sample_rate(audio_bytes)
    inference_params = dict(task="translate", sample_rate = samplerate )
    whisper_model = Model("https://clarifai.com/openai/whisper/models/whisper-large-v2", pat=clarifai_pat)
    model_prediction = whisper_model.predict_by_bytes(audio_bytes, "audio", inference_params=inference_params)
    return model_prediction.outputs[0].data.text.raw

def assembly(audio_bytes):
    samplerate = get_sample_rate(audio_bytes)
    inference_params = dict(sample_rate = samplerate )
    transcription_model = Model("https://clarifai.com/assemblyai/speech-recognition/models/audio-transcription", pat=clarifai_pat)
    model_prediction = transcription_model.predict_by_bytes(audio_bytes, "audio", inference_params=inference_params)
    return model_prediction.outputs[0].data.text.raw

def fbasr(audio_bytes):
    samplerate = get_sample_rate(audio_bytes)
    inference_params = dict(sample_rate = samplerate )
    chirp_model = Model("https://clarifai.com/facebook/asr/models/asr-wav2vec2-base-960h-english", pat=clarifai_pat)
    model_prediction = chirp_model.predict_by_bytes(audio_bytes, "audio", inference_params=inference_params)
    return model_prediction.outputs[0].data.text.raw

def chirpasr(audio_bytes):
    samplerate = get_sample_rate(audio_bytes)
    inference_params = dict(sample_rate = samplerate )
    chirp_model = Model("https://clarifai.com/gcp/speech-recognition/models/chirp-asr", pat=clarifai_pat)
    model_prediction = chirp_model.predict_by_bytes(audio_bytes, "audio", inference_params=inference_params)
    return model_prediction.outputs[0].data.text.raw




def text_to_speech(input_text):
    inference_params = dict(voice="alloy", speed=0.9)
    model_prediction = Model(
        "https://clarifai.com/openai/tts/models/openai-tts-1"
    ).predict_by_bytes(
        input_text.encode(), input_type="text", inference_params=inference_params
    )
    audio_base64 = model_prediction.outputs[0].data.audio.base64
    return audio_base64


def gpt4(prompt):
    from clarifai.client.model import Model

    inference_params = dict(temperature=0.2, max_tokens=100)

    # Model Predict
    model_prediction = Model("https://clarifai.com/openai/chat-completion/models/gpt-4-turbo").predict_by_bytes(prompt.encode(), input_type="text", inference_params=inference_params)

    return model_prediction.outputs[0].data.text.raw


if __name__ == "__main__":

    if st.button('Input Sign Language'):


        sign_script_path = "/home/ishita-wicon/Hackathon/Sign-Language-To-Text-Conversion/Application.py"

        output = subprocess.check_output(["python3", sign_script_path])

        with open("/home/ishita-wicon/Hackathon/sentence.txt", "r") as file:
            captured_result = file.read()

        #Pass this result to Text to speech/chatbot etc

        st.write(captured_result)

        gpt_op_text = gpt4(captured_result)
        st.write(gpt_op_text)

    # if st.button('Input Audio'):

    audio_bytes = audio_recorder(pause_threshold=100,sample_rate=44100)
    if audio_bytes:
        st.text(whisper(audio_bytes))

        gpt_op_text = gpt4(whisper(audio_bytes))

        gpt_op_sp = text_to_speech(gpt_op_text)

        st.write(gpt_op_text)
        st.audio(gpt_op_sp)
                

    # if st.button('Input Text'):

    t_i = st.text_input("Put your Text here")

    if t_i:
        gpt_op_text = gpt4(t_i)

        gpt_op_sp = text_to_speech(gpt_op_text)
        st.write(gpt_op_text)
        st.audio(gpt_op_sp)