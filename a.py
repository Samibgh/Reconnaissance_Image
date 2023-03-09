import main_thread
import streamlit as st
import cv2
import numpy as np
import speech_recognition as sr

# Streamlit initialization
st.title("Face Recognition")

def main():
    # Créer une instance de la classe FaceRecognition
    fr = main_thread.FaceRecognition()
    cap = cv2.VideoCapture(0)
    # Boucle principale de l'application
    while True:
        # Lire le flux vidéo de la webcam
        ret, frame = cap.read()

        # Si le flux vidéo est lu correctement
        if ret:
            # Exécuter la reconnaissance faciale sur le flux vidéo
            fr.run_recognition()

        # Si l'utilisateur appuie sur la touche "q", quitter l'application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Fermer la fenêtre de la vidéo et libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

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
                    st.session_state.run = True
                    main()
                else:
                    st.sidebar.warning(f"Je n'ai pas reconnu la commande: {text}")
            except sr.UnknownValueError:
                st.sidebar.warning("Je n'ai pas entendu")
            except sr.RequestError as e:
                st.sidebar.error(f"Erreur; {e}")

#if __name__ == "__main__":
 #   main()