import speech_recognition as sr

r = sr.Recognizer()

with sr.Microphone() as source:
    print("Dites quelque chose !")
    audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        print("Vous avez dit : {}".format(text))
    except:
        print("Impossible de transcrire la parole")