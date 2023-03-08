import cv2

# ouvrir la webcam
cap = cv2.VideoCapture(0)

# boucle infinie pour capturer les images de la webcam
while True:
    # lire les images de la webcam
    ret, frame = cap.read()
    
    # afficher les images dans une fenêtre
    cv2.imshow('Webcam', frame)
    
    # attendre une touche clavier pour sortir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# libérer les ressources
cap.release()
cv2.destroyAllWindows()