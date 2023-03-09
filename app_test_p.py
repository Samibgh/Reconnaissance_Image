import cv2
import streamlit as st
import speech_recognition as sr
import time
import face_recognition
import cv2
import os, sys
import numpy as np
import math
import pickle
from PIL import Image 
import main_thread


##############################################################################################
#                                 Fonctions appli                                            #
##############################################################################################

def start_camera():
    st.title("Webcam Live Feed")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set width to 640 pixels
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # set height to 480 pixels

    # Pour enregistrer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 5, (640, 480))

    out.release()  # fin de l'enregistrement mais ça enregistre rien du tout

    while True:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        if not st.session_state.run:
            st.write('Stopped')
            camera.release()
            out.release()  # fin de l'enregistrement mais ça enregistre rien du tout
            break

        if st.session_state.listen_for_commands:    
        # Listen for voice commands while webcam is running
            with sr.Microphone() as source:
                audio = r.listen(source, phrase_time_limit=5)
                try:
                    text = r.recognize_google(audio, language='fr-FR')
                    if text.lower() == "arrêt":
                        st.session_state.run = False
                        st.sidebar.warning('Arrêt de la webcam')
                    else:
                        st.sidebar.warning(f"Je n'ai pas reconnu la commande: {text}")   
                except sr.UnknownValueError:
                    st.sidebar.warning("Je n'ai pas entendu")
                except sr.RequestError as e:
                    st.error(f"Erreur; {e}")
                time.sleep(4)
    st.session_state.listen_for_commands = False       


st.title("Voice-controlled Webcam")
st.sidebar.write("Barre des messages : ")
st.session_state.listen_for_commands = False
r = sr.Recognizer()
run_button = st.checkbox("Activate Voice Control")


if run_button:
    with sr.Microphone() as source:
        st.write("Dites 'tarte' pour démarrer la webcam")
        r.adjust_for_ambient_noise(source)
        while True:
            audio = r.listen(source, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio, language='fr-FR')
                if text.lower() == "tarte":
                    st.write("Demarrage de la webcam en cours")
                    st.session_state.run = True
                    fr = main_thread.run_recognition()
                    st.image(fr, channels='BGR', use_column_width=True)
                else:
                    st.sidebar.warning(f"Je n'ai pas reconnu la commande: {text}")
            except sr.UnknownValueError:
                st.sidebar.warning("Je n'ai pas entendu")
            except sr.RequestError as e:
                st.sidebar.error(f"Erreur; {e}")

