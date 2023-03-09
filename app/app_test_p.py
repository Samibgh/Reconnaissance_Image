import cv2
import streamlit as st
import speech_recognition as sr
import time

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
            camera.release()
            break

        # Listen for voice commands while webcam is running
        with sr.Microphone() as source:
            audio = r.listen(source, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio, language='fr-FR')
                if text.lower() == "arrêt":
                    st.session_state.run = False
                else:
                    st.warning(f"Je n'ai pas reconnu la commande: {text}")
                    # Add query parameter containing current time
                    st.experimental_set_query_params(message_time=str(time.time()))
            except sr.UnknownValueError:
                st.warning("Je n'ai pas entendu")
            except sr.RequestError as e:
                st.error(f"Erreur; {e}")
            
            # Remove messages displayed more than 3 seconds ago
            query_params = st.experimental_get_query_params()
            message_time = query_params.get("message_time", [None])[0]
            if message_time is not None and time.time() - float(message_time) > 3:
                st.empty()  # Remove the last element from the output stream

st.title("Voice-controlled Webcam")
r = sr.Recognizer()
run_button = st.button("Activate Voice Control")

if run_button:
    with sr.Microphone() as source:
        st.write("Dites 'tarte' pour démarrer la webcam")
        r.adjust_for_ambient_noise(source)
        while True:
            audio = r.listen(source, phrase_time_limit=5)
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
