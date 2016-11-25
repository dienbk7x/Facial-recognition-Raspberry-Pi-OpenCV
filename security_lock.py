#!/usr/bin/env python
"""Raspberry Pi Face Recognition Treasure Box
Treasure Box Script
Copyright 2013 Tony DiCola 
Copyright 2016 Monsieur Vechai
"""

import time
from time import sleep, strftime
from datetime import datetime
import cv2
import config, face
import Adafruit_CharLCD as LCD
import RPi.GPIO as GPIO


#Setup the GPIO relay pin
Relay_Pin = config.LOCK_RELAY_PIN
GPIO.setmode(GPIO.BCM)
GPIO.setup(Relay_Pin, GPIO.OUT)

# Initialize the LCD 
lcd = LCD.Adafruit_CharLCDPlate()

#Load the training image
lcd.message('Load data...'+ '\n')
model = cv2.createEigenFaceRecognizer()
model.load(config.TRAINING_FILE)
lcd.message('Data loaded!'+ '\n')
time.sleep(1)
lcd.clear()

def face_check():
    #Only run face_check() for a certain amount of time according to the config file
    time_out_start = time.time()
    while time.time() < time_out_start + config.TIME_OUT:
        # Initialize camera'
        camera = cv2.VideoCapture(config.WEBCAM_ID)
        ret1, frame1 = camera.read()
        image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        result = face.detect_single(image)
        #Check if any face presents
        if result is None:
            lcd.clear()
            lcd.message('NO FACE DETECTED' + '\n')
            GPIO.output(Relay_Pin, GPIO.LOW)
        else:
            lcd.clear()
            x, y, w, h = result
            crop = face.resize(face.crop(image, x, y, w, h))
            label, confidence = model.predict(crop)
            if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
                lcd.clear()
                lcd.message(str(label) + ' OK. OPEN LOCK!' + '\n' + str(confidence) + '\n')
                GPIO.output(Relay_Pin, GPIO.HIGH)
            else:
                lcd.clear()
                lcd.message(str(label) + ' NOT RECOGNIZED' + '\n' + str(confidence)+ '\n')
                GPIO.output(Relay_Pin, GPIO.LOW)
        camera.release()

        
if __name__ == '__main__':
    GPIO.output(Relay_Pin, GPIO.LOW)
    lcd.clear()
    while True:
        lcd.clear()
        lcd.message(datetime.now().strftime('%b %d %H:%M:%S')+ '\n')
        lcd.message('Press Select' + '\n')
        time.sleep(1)
        if lcd.is_pressed(LCD.SELECT):
            lcd.clear()
            face_check()
            lcd.clear()
            lcd.message('Time out' + '\n')
            time.sleep(5)

