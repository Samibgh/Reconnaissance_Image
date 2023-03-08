
import pickle 
import cv2 
import dlib
import face_recognition
import numpy as np
import streamlit as st
import time
import os
# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
tmp = []

##_________

from keras.models import load_model
from keras_preprocessing.image import load_img


face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
#Fonctions
def easy_face_reco(frame, known_face_encodings, known_face_names):
    process_this_frame = True
    try:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    except:
        small_frame = frame

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray,1.2,5)
        
        label =""
        gender = ""
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            tmp.append(name)

    
    process_this_frame = not process_this_frame
    
    face_img = frame[0:20,0:20]
    blob = cv2.dnn.blobFromImage(face_img, 1.0,size=(227, 227),
				swapRB=False)
    
    # Display the results
    for i, (top, right, bottom, left), name in zip(range(len(face_locations)), face_locations, face_names):
        label = i + 1
    


    #for (top, right, bottom, left), name, label in zip(face_locations, face_names):
        
        face_img = frame[max(0,left-20): min(top+20,frame.shape[0]-1),
                   max(0,right-20):min(bottom+20, frame.shape[1]-1)]
        
        if len(face_img) >0 :
    
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)      

        
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom + 90), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
        cv2.putText(frame, str(label), (left + 6, bottom + 25), font, 1, (0,0,255), 1)
        #cv2.putText(frame, label_age, (left + 6, bottom + 55), font, 1.0, (19, 69, 139), 1)
        #cv2.putText(frame, gender, (left + 6, bottom + 85), font, 1.0, (250, 0, 0), 1)

    return frame, np.unique(tmp)

known_dir = "C:/Users/USER/Documents/Master_SISE/challengeweb/Reconnaissance_Image/individuelle"

# Liste des noms de fichiers d'images connus
known_face_files = os.listdir(known_dir)

# Liste des noms de personnes connues
known_face_names = [os.path.splitext(file)[0] for file in known_face_files]

# Liste des chemins complets des images connues
known_face_paths = [os.path.join(known_dir, file) for file in known_face_files]

# Liste des encodages de visage des images connues
known_face_encodings = []
for path in known_face_paths:
    image = face_recognition.load_image_file(path)
    if len(face_recognition.face_encodings(image)) > 0:
                face_encoding = face_recognition.face_encodings(image)[0]

    else:
                print(f"No face found in {image}")
                continue
    #encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encodings)



# Enregistrer les noms des personnes connues dans un fichier texte
with open("known_face_names.txt", "w") as f:
    for name in known_face_names:
        f.write(name + "\n")


# Convertir les tableaux numpy en listes
known_face_encodings_list = [enc.tolist() for enc in known_face_encodings]

# Enregistrer les encodages faciaux dans un fichier texte
with open("known_face_encodings.txt", "w") as f:
    for enc in known_face_encodings_list:
        f.write(' '.join([str(e) for e in enc]) + "\n")


fichier = open('known_face_encodings.txt', 'rb')
known_face_encodings = pickle.load(fichier)

fichier = open('known_face_names.txt', 'rb')
known_face_names = pickle.load(fichier)



@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
Vid = None
if 'vidOn' not in st.session_state:
    st.session_state['vidOn'] = 'intial'

#title of the App
st.title("Reconnaissance faciale en temps réel")

def LancerButton():
    st.session_state.vidOn = "ok"

def ArretButton():
    st.session_state.vidOn = "no"
    st.session_state.video_capture.release()
    st.session_state.out.release()
    cv2.destroyAllWindows()
    st.session_state.videoButton = False
    if Vid==True:
        st.experimental_memo.clear()

videoButton = st.button('Lancer la camera', on_click=LancerButton)


mp4Vid = st.checkbox("Enregistrer la video", True)
if mp4Vid ==True:
    nomFich = st.text_input('Nom du fichier video: ', 'enregistrementSISE')


videoArret = st.empty()

if st.session_state.vidOn=="ok":
     Vid =True
     st.session_state["FRAME_WINDOW"] = st.image([])
     st.session_state["video_capture"] = cv2.VideoCapture(0)
     prev_frame_time = 0
     new_frame_time = 0

     videoArret.button('Arrêter la camera', on_click= ArretButton)
     
     if mp4Vid ==True:
         width = int(st.session_state.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
         heigth= int(st.session_state.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
         #fps_input= int(st.session_state.video_capture.get(cv2.CAP_PROP_FPS))
        
         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
         st.session_state['out'] = cv2.VideoWriter(nomFich + '.mp4',fourcc, 3.7, (width,heigth))
     
     while (st.session_state.video_capture.isOpened()) : 
        ret, frame = st.session_state.video_capture.read()
        
        if not ret:
            continue
    
        frame = cv2.flip(frame,1)
        frame, autre = easy_face_reco(frame, known_face_encodings, known_face_names)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ## Calcul FPS
        new_frame_time = time.time() 
        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        fps =int(np.around(fps +0.5 ,0))
        fps = str(fps) 
        cv2.putText(frame , "FPS : "+fps, (7, 25), font, 1, (255, 255, 0), 1, cv2.LINE_AA)

        st.session_state.FRAME_WINDOW.image(frame)
        

        if mp4Vid ==True:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            st.session_state.out.write(frame)    
            
            with open(nomFich + '.txt', 'w') as f:
                for item in autre:
                    f.write("%s\n" % item)
    
    
if videoArret==True:
    st.session_state.video_capture.release()
    st.session_state.out.release()
    cv2.destroyAllWindows()
    st.session_state.videoButton = False
    if Vid==True:
        st.experimental_memo.clear()
        st.experimental_rerun()


























