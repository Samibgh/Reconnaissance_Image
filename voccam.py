import cv2
import speech_recognition as sr

r = sr.Recognizer()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    with sr.Microphone() as source:
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("Commande vocale : {}".format(text))
            if "start video" in text.lower():
                cap = cv2.VideoCapture(0)
            elif "stop video" in text.lower():
                cap.release()
            elif "exit" in text.lower():
                break
        except:
            pass

    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
