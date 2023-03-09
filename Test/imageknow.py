import cv2
import os
import numpy as np
import face_recognition
# data_path = "./individuelle"  
# KNOWN_FACE_ENCODINGS = "./known_face_encodings.npy"  
# KNOWN_FACE_NANE = "./known_face_name.npy"  

# new_images = []  
# processd_images = [] 
# known_face_names = []  
# known_face_encodings = []  
# name_and_encoding = "./face_encodings.txt"

# image_thread = 0.15




# def encoding_images(path):
#     with open(name_and_encoding, 'w') as f:
#         subdirs = [os.path.join(path, x) for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
#         for subdir in subdirs:
#             print('process image name :', subdir)
#             person_image_encoding = []
#             for y in os.listdir(subdir):
#                 print("image name is ", y)
#                 _image = face_recognition.load_image_file(os.path.join(subdir, y))
#                 face_encodings = face_recognition.face_encodings(_image)
#                 name = os.path.split(subdir)[-1]
#                 if face_encodings and len(face_encodings) == 1:
#                     if len(person_image_encoding) == 0:
#                         person_image_encoding.append(face_encodings[0])
#                         known_face_names.append(name)
#                         continue
#                     for i in range(len(person_image_encoding)):
#                         distances = face_recognition.compare_faces(person_image_encoding, face_encodings[0], tolerance=image_thread)
#                         if False in distances:
#                             person_image_encoding.append(face_encodings[0])
#                             known_face_names.append(name)
#                             print(name, " new feature")
#                             f.write(name + ":" + str(face_encodings[0]) + "\n")
#                             break
#                     # face_encoding = face_recognition.face_encodings(_image)[0]
#                     # face_recognition.compare_faces()
#             known_face_encodings.extend(person_image_encoding)
#             bb = np.array(known_face_encodings)
#             print("--------")
#     np.save(KNOWN_FACE_ENCODINGS, known_face_encodings)
#     np.save(KNOWN_FACE_NANE, known_face_names) 
import face_recognition
from PIL import Image
import os

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
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        encoding = encodings[0]
        known_face_encodings.append(encoding)
    else:
        print(f"No face found in {path}")

print(f"Number of face encodings: {len(known_face_encodings)}")




# Enregistrer les noms des personnes connues dans un fichier texte
with open("C:/Users/USER/Documents/Master_SISE/challengeweb/Reconnaissance_Image/known_face_names.txt", "w") as f:
    for name in known_face_names:
        f.write(name + "\n")


# Convertir les tableaux numpy en listes
known_face_encodings_list = [enc.tolist() for enc in known_face_encodings]

# Enregistrer les encodages faciaux dans un fichier texte
with open("C:/Users/USER/Documents/Master_SISE/challengeweb/Reconnaissance_Image/known_face_encodings.txt", "w") as f:
    for enc in known_face_encodings_list:
        f.write(' '.join([str(e) for e in enc]) + "\n")
