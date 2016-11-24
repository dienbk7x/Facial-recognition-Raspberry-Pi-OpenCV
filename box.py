#!/usr/bin/env python
"""Raspberry Pi Face Recognition Treasure Box
Treasure Box Script
Copyright 2013 Tony DiCola 
"""
import cv2
import config, face

camera = cv2.VideoCapture(config.WEBCAM_ID)

if __name__ == '__main__':
	# Load training data into model
	print 'Loading training data...'
	model = cv2.createEigenFaceRecognizer()
	model.load(config.TRAINING_FILE)
	print 'Training data loaded!'

	# Initialize camer and box.'	
	ret1, frame1 = camera.read()
	image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
	result = face.detect_single(image)
	if result is None:
	    print 'Could not detect single face! Check capture image.'
	x, y, w, h = result
        # Crop and resize image to face.
	crop = face.resize(face.crop(image, x, y, w, h))
	# Test face against model.
	label, confidence = model.predict(crop)
	print 'Predicted {0} face with confidence {1} (lower is more confident).'.format(
					'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', 
					confidence)
	if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
	   print 'Recognized face!'
	else:
	   print 'Did not recognize face!'
	   
        camera.release()
        cv2.destroyAllWindows()
	