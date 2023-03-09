import face_recognition
import os
import pickle

known_face_encodings = []
known_face_names = []

for image in os.listdir('individuelle'):
    face_image = face_recognition.load_image_file(f"individuelle/{image}")
    if len(face_recognition.face_encodings(face_image)) > 0:
        face_encoding = face_recognition.face_encodings(face_image)[0]
    else:
        print(f"No face found in {image}")
        continue
    known_face_encodings.append(face_encoding)
    known_face_names.append(image.split(".")[0])

with open("known_face_encodings.pickle", "wb") as f:
    pickle.dump(known_face_encodings, f)

with open("known_face_names.pickle", "wb") as f:
    pickle.dump(known_face_names, f)
