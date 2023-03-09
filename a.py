import main_thread
import streamlit as st
import cv2
import numpy as np

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

if __name__ == "__main__":
    main()