# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:37:30 2022

@author: Administrateur
"""

import pickle 
import cv2 
import face_recognition
import numpy as np
import streamlit as st
import time
from tensorflow.keras.preprocessing.image import img_to_array

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
tmp = []


##_________

from keras.models import load_model


face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
classifier =load_model('models/Emotion_detector.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
#__________


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_INTERVALS = ['(0, 2)', '(4, 6)','(25, 32)', '(8, 12)', '(15, 20)',
                  '(38, 43)', '(48, 53)', '(60, 100)']

genderList = ['Male', 'Female']

models_path = r"models\\"
images_path = r"photos\\"
encoding_path = r"encoding\\"

#Model age
AGE_MODEL = models_path  + 'deploy_age.prototxt'
AGE_PROTO = models_path + 'age_net.caffemodel'
age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL, AGE_PROTO)
#__________

#Model gender
GENDER_MODEL = models_path + 'gender_net1.caffemodel'
GENDER_PROTO = models_path + 'deploy_gender1.prototxt'
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)
#__________


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

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            tmp.append(name)

    
        list_emotions =[]
        list_emotions_positions =[]
        for (x,y,w,h) in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            try:
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            except:
                continue
    
    
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                
            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            
            list_emotions.append(label)
            list_emotions_positions.append(label_position)
            
            
    
    process_this_frame = not process_this_frame
    
    label_age =""
    
    face_img = frame[0:20,0:20]
    blob = cv2.dnn.blobFromImage(face_img, 1.0,size=(227, 227),
				swapRB=False)
    
    # Display the results
    for (top, right, bottom, left), name, label in zip(face_locations, face_names, list_emotions):
        
        face_img = frame[max(0,left-20): min(top+20,frame.shape[0]-1),
                   max(0,right-20):min(bottom+20, frame.shape[1]-1)]
        
        if len(face_img) >0 :
    
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

        
        # Predict Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        
        i = age_preds[0].argmax()
        age = AGE_INTERVALS[i]
        label_age = f"Age:{age} "
        
         # Predict Gender
        gender_net.setInput(blob)
        genderPreds = gender_net.forward()
        gender = genderList[genderPreds[0].argmax()]
        
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom + 90), (0, 255, 0), cv2.FILLED)

        # Draw a box around the face
        #cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 0), 1)
        cv2.putText(frame, label, (left + 6, bottom + 25), font, 1, (0,0,255), 1)
        cv2.putText(frame, label_age, (left + 6, bottom + 55), font, 1.0, (19, 69, 139), 1)
        cv2.putText(frame, gender, (left + 6, bottom + 85), font, 1.0, (250, 0, 0), 1)

    return frame, np.unique(tmp)

#fichier = open('known_face_encodings.txt', 'rb')
#known_face_encodings = pickle.load(fichier)

#fichier = open('known_face_names.txt', 'rb')
#known_face_names = pickle.load(fichier)

with open(f"{encoding_path}known_face_encodings.pickle", "rb") as f:
    known_face_encodings = pickle.load(f)

with open(f"{encoding_path}known_face_names.pickle", "rb") as f:
    known_face_names= pickle.load(f)


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
Vid = None
if 'vidOn' not in st.session_state:
    st.session_state['vidOn'] = 'intial'

#title of the App
st.title("Application de reconnaissance faciale en temps réel")

def LancerButton():
    st.session_state.vidOn = "ok"

def ArretButton():
    st.session_state.vidOn = "no"
    st.session_state.video_capture.release()
    st.session_state.out.release()
    cv2.destroyAllWindows()
    st.session_state.videoButton = False
    if Vid==True:
        st.cache_data.clear()

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
        st.cache_data.clear()
        st.experimental_rerun()




















