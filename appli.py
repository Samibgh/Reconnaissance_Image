import face_recognition
import cv2
import os, sys
import numpy as np
import math
import pickle
import streamlit as st
import time 
from PIL import Image 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.models import load_model
#from keras.preprocessing.image import img_to_array
from keras_preprocessing.image import img_to_array

face_classifier = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
classifier =load_model('models/Emotion_detector.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_INTERVALS = ['(0, 3)', '(4, 6)','(7, 12)', '(13,15)','(16, 20)', '(21, 25)', '(26,29)','(30, 35)', '(36,39)','(40, 45)','(46,49)','(50, 55)','(56, 59)','(60, 65)','(66, 70)','(71, 75)','(76, 80)','(81, 85)','(86, 90)','(91, 95)','(96, 100)']

listgender = ['M', 'F']

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



for img_filename in os.listdir(images_path):
    
    img = Image.open(images_path + img_filename )
    new_image = img.resize((1200,1600))
    new_image.save(os.path.join(images_path, str(img_filename)))

# Helper
def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    process_current_frame = True

    with open(f"{encoding_path}known_face_encodings.pickle", "rb") as f:
        known_face_encodings = pickle.load(f)

    with open(f"{encoding_path}known_face_names.pickle", "rb") as f:
        known_face_names= pickle.load(f)

    def __init__(self):
        self.known_face_encodings
        self.known_face_names

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()

            # Only process every other frame of video to save time
            if self.process_current_frame:

                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


                # Resize frame of video to 1/4 size for faster face recognition processing
                rgb_small_frame = cv2.resize(small_frame, (0, 0), fx=0.25, fy=0.25)

                
                # Find all the faces and face encodings in the current frame of video
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                faces = face_classifier.detectMultiScale(gray,1.2,5)

                
                self.face_names = []
                for face_encoding in self.face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    # Calculate the shortest distance to face
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                     
                    # Get the index of the face with the shortest distance
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')


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
                 
            self.process_current_frame = not self.process_current_frame
            face_img = frame[0:20,0:20]
            blob = cv2.dnn.blobFromImage(face_img, 1.0,size=(227, 227),
				swapRB=False)

            # Display the results
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
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


                 # Predict gender
                gender_net.setInput(blob)
                genderPreds = gender_net.forward()
                gender = listgender[genderPreds[0].argmax()]
                
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Create the frame with the name
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left - 100, bottom - 180), font, 0.8, (255, 255, 255), 1)
                cv2.putText(frame, label, (left + 6, bottom + 25), font, 1, (0,0,255), 1)
                cv2.putText(frame, label_age, (left + 6, bottom + 30), font, 1.0, (255, 255, 0), 1)
                cv2.putText(frame, gender, (left + 6, bottom + 70), font, 1.0, (0, 0, 255), 1)

            # Display the resulting image
            cv2.imshow('Face Recognition', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) == ord('q'):
                break
        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 
Vid = None
if 'vidOn' not in st.session_state:
    st.session_state['vidOn'] = 'intial'
#title of the App
st.title("Reconnaissance faciale  SISE en temps réel")

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
        frame, autre =FaceRecognition(frame,known_face_encodings,known_face_names)
        #frame, autre = easy_face_reco(frame, known_face_encodings, known_face_names)
        
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


