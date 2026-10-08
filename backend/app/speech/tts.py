"""
File: tts.py

Purpose:
Converts translated English text into speech
using the computer's built-in voice.
"""
import pyttsx3

class Speaker:
    def speak(self,text):
        if text.strip()=="":#empty text
            return 
        engine=pyttsx3.init()#turn on the voice 

        engine.setProperty("rate", 170)
        engine.setProperty("volume", 1.0)

        engine.say(text)
        engine.runAndWait()
        engine.stop()

if __name__=="__main__": 
    speaker=Speaker()
    speaker.speak("Hello, yes.")
