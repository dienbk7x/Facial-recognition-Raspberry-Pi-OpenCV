#!/usr/bin/env python
"""
Demo script for Watson IBM facial recognition 
Copyright 2016 Monsieur Vechai
"""
import cv2
import config, face
import os, time
import json
from watson_developer_cloud import VisualRecognitionV3

camera = cv2.VideoCapture(0)

if __name__ == '__main__':
    # Initialize camer and box.'	
    ret, frame = camera.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = face.detect_single(image)
    
    visual_recognition = VisualRecognitionV3('2016-05-20', api_key='a4f1c9fe8d9015cedbac7e3ca2c4dd8b82677008')
    photo = cv2.imwrite('capture'+'.png',frame)
    path_of_photo = str(os.path.abspath('capture.png'))
    with open(path_of_photo, 'rb') as image_file:
        #Uncomment to see general image recognition with different classifier
        #print(json.dumps(visual_recognition.classify(images_file=image_file, threshold=0.1, classifier_ids=['CarsvsTrucks_1675727418', 'default']), indent=2))
        print(json.dumps(visual_recognition.detect_faces(images_file=image_file), indent=2))
    if result is None:
        print 'Could not detect single face!'
    else:
        print 'Face recognized'
    camera.release()
    cv2.destroyAllWindows()
	