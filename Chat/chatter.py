"""
pip install chatterbot
# pip install spacy==2.1
pip install SpeechRecognition
pip install pyaudio
#pip install pipwin
#pipwin install pyaudio
python -m spacy download en
python -m spacy link en_core_web_sm en #rodar como admin
"""

import speech_recognition as sr
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

#apenas para quem usa windows
import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
speak.Rate=3

#para quem usa MacOS ou Linux
#import os

def recognize_speech_from_mic(recognizer, microphone):
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")

    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    try:
        response["transcription"] = recognizer.recognize_google(audio, language="pt-BR")
    except sr.RequestError:
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        response["error"] = "Unable to recognize speech"

    return response

recognizer = sr.Recognizer()
microphone = sr.Microphone()
with microphone as source:
    recognizer.adjust_for_ambient_noise(source)
        
name = "Robozitcha"
bot = ChatBot('{} Bot'.format(name), read_only=True)
#bot.storage.drop()

trainer = ChatterBotCorpusTrainer(bot)
trainer.train(
    "C:\\Users\\jasmi\\OneDrive\\Área de Trabalho\\PLN RV\\Chat\\meuCorpus.yml"
)

greets = 'Vamos começar!'
print(greets)
speak.Speak(greets)

while True:
    guess = recognize_speech_from_mic(recognizer, microphone)
    if guess["transcription"]:
        pergunta = guess["transcription"]
    else:
        continue

    print('Você: ', pergunta)
    resposta = bot.get_response(pergunta)
    #Windows
    speak.Speak(resposta)
    print('{}: '.format(name), resposta)
    
    #comando = re.search("&&(.+?)&&", resposta)
    #resposta = re.sub("&&(.+?)>&&","",resposta)
    #MacOS
    #os.system("say '{}'".format(resposta))
    #Linux
    #os.system("spd-say '{}'".format(resposta))

    

