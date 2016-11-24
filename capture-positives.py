#!/usr/bin/env python
"""Raspberry Pi Face Recognition Treasure Box
Positive Image Capture Script
Copyright 2013 Tony DiCola 
Copyright 2016 Monsieur Vechai
Run this script to capture positive images for training the face recognizer.
"""

import glob, os, sys, select, time
import numpy as np
import cv2
import config, face

# Prefix for positive training image filenames.
POSITIVE_FILE_PREFIX = 'positive_'
camera = cv2.VideoCapture(config.WEBCAM_ID)


if __name__ == '__main__':
	# Create the directory for positive training images if it doesn't exist.
	if not os.path.exists(config.POSITIVE_DIR):
		os.makedirs(config.POSITIVE_DIR)
	# Find the largest ID of existing positive images.
	# Start new images after this ID value.
	files = sorted(glob.glob(os.path.join(config.POSITIVE_DIR,
	        POSITIVE_FILE_PREFIX + '[0-9][0-9][0-9].pgm')))
	        
	count = 0
	number_of_detected_faces = 0
	if len(files) > 0:
		# Grab the count from the last filename.
		count = int(files[-1][-7:-4])+1
		
	print 'Capturing positive training images.'
	print 'Press button or type c (and press enter) to capture an image.'
	print 
	'Press Ctrl-C to quit.'
	while number_of_detected_faces < config.MAX_NUMBER_OF_IMAGES:	        
                print 'Capturing image...'
                ret1, frame1 = camera.read()
                image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) #Convert to grayscale
                #Uncomment the next 3 lines for to see webcam view
                #cv2.imshow('PRESS ANY KEY TO TAKE PHOTO',image)
                #cv2.waitKey(0)
                #time.sleep(1.0/config.CAPTURE_HZ)
                result = face.detect_single(image)
                if result is None:
                    print 'NO FACE DETECTED! Try to move to different positions.'
                    continue
                x, y, w, h = result
                # Crop image as close as possible to desired face aspect ratio.
                # Might be smaller if face is near edge of image.
                crop = face.crop(image, x, y, w, h)
                # Save image to file.
                filename = os.path.join(config.POSITIVE_DIR, POSITIVE_FILE_PREFIX + '%03d.pgm' % count)
                cv2.imwrite(filename, crop)
                print 'FACE DETECTED. Wrote to training image.', filename
                print '%s more frames to go!' % (config.MAX_NUMBER_OF_IMAGES - number_of_detected_faces)
                number_of_detected_faces +=1
                count += 1
        print 'DONE CAPTURING IMAGES FOR TRAINING.'
        camera.release()
        cv2.destroyAllWindows()


