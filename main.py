import cv
import cv2
import time
from PIL import Image
import numpy as np
import csv
import logistic
import mouthdetection as m

WIDTH, HEIGHT = 28, 10 # all mouth images will be resized to the same size
dim = WIDTH * HEIGHT # dimension of feature vector

"""
pop up an image showing the mouth with a blue rectangle
"""
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

def show(area): 
    cv.Rectangle(img,(area[0][0],area[0][1]),
                     (area[0][0]+area[0][2],area[0][1]+area[0][3]),
                    (255,0,0),2)
    cv.NamedWindow('Face Detection', cv.CV_WINDOW_NORMAL)
    cv.ShowImage('Face Detection', img) 
    cv.WaitKey()

"""
given an area to be cropped, crop() returns a cropped image
"""
def crop(area): 
    crop = img[area[0][1]:area[0][1] + area[0][3], area[0][0]:area[0][0]+area[0][2]] #img[y: y + h, x: x + w]
    return crop

"""
given a jpg image, vectorize the grayscale pixels to 
a (width * height, 1) np array
it is used to preprocess the data and transform it to feature space
"""
def vectorize(filename):
    size = WIDTH, HEIGHT # (width, height)
    im = Image.open(filename) 
    resized_im = im.resize(size, Image.ANTIALIAS) # resize image
    im_grey = resized_im.convert('L') # convert the image to *greyscale*
    im_array = np.array(im_grey) # convert to np array
    oned_array = im_array.reshape(1, size[0] * size[1])
    return oned_array


if __name__ == '__main__':
    """
    load training data
    """
    # create a list for filenames of smiles pictures
    smilefiles = []
    with open('smiles.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            smilefiles += rec

    # create a list for filenames of neutral pictures
    neutralfiles = []
    with open('neutral.csv', 'rb') as csvfile:
        for rec in csv.reader(csvfile, delimiter='	'):
            neutralfiles += rec

    # N x dim matrix to store the vectorized data (aka feature space)       
    phi = np.zeros((len(smilefiles) + len(neutralfiles), dim))
    # 1 x N vector to store binary labels of the data: 1 for smile and 0 for neutral
    labels = []

    # load smile data
    PATH = "C:/emotion/data/smile/"
    for idx, filename in enumerate(smilefiles):
        phi[idx] = vectorize(PATH + filename)
        labels.append(1)

    # load neutral data    
    PATH = "C:/emotion/data/neutral/"
    offset = idx + 1
    for idx, filename in enumerate(neutralfiles):
        phi[idx + offset] = vectorize(PATH + filename)
        labels.append(0)

    """
    training the data with logistic regression
    """
    lr = logistic.Logistic(dim)
    lr.train(phi, labels)
    

    """
    open webcam and capture images
    """
    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    print ("\n\n\n\n\npress space to take picture; press ESC to exit")
    while 1:
        ret, img = vc.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('img',img)

        #to exit on pressing escape
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(40)
        ret, img = vc.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        cv2.imshow('img',img)
        if key == 27: # exit on ESC
            break
        if key == 32: # press space to save images
            cv.SaveImage("webcam.jpg", cv.fromarray(frame))
            img = cv.LoadImage("webcam.jpg") # input image
            mouth = m.findmouth(img)
            # show(mouth)
            if mouth != 2: # did not return error
                mouthimg = crop(mouth)
                cv.SaveImage("webcam-m.jpg", mouthimg)
                # predict the captured emotion
                result = lr.predict(vectorize('webcam-m.jpg'))
                if result == 1:
                    print ("Happy Face :-) ")
                else:
                    print ("Neutral Face:-| ")
            else:
                print ("failed to detect mouth. Try hold your head straight and make sure there is only one face.")
    
    cv2.destroyWindow("preview")