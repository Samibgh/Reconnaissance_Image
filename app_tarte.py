import cv2
import streamlit as st
import speech_recognition as sr

def start_camera():
    st.title("Webcam Live Feed")
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    while True:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
        if not st.session_state.run:
            st.write('Stopped')
            break

st.title("Voice-controlled Webcam")
r = sr.Recognizer()
run_button = st.button("Activate Voice Control")

if run_button:
    with sr.Microphone() as source:
        st.write("Dites 'tarte' pour démarrer")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='fr-FR')
            if text.lower() == "tarte":
                st.session_state.run = True
                start_camera()
            else:
                st.warning(f"Je n'ai pas reconnu la commande: {text}")
        except sr.UnknownValueError:
            st.warning("Je n'ai pas entendu")
        except sr.RequestError as e:
            st.error(f"Erreur; {e}") 
