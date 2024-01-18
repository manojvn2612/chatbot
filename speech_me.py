import speech_recognition as sr
import pyttsx3
def speech_to_text():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Capture audio from the microphone
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

    try:
        # Use Google Web Speech API to convert speech to text
        text = recognizer.recognize_google(audio)
        print(text)
        return text
    except sr.UnknownValueError:
        return "error"
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")

if __name__ == "__main__":
    speech_to_text()