import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from tts import speak
import time

print("Proton TTS Test Starting...")

speak("Hello. Proton voice system is now active.")
time.sleep(3)

speak("This is a second sentence to test stability and clarity.")
time.sleep(3)

speak("Final test. No clipped words should occur now.")

time.sleep(5)
print("Test finished.")