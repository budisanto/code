#importing Libraries
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
import pytesseract

#=====================================================
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,help="path to the input image")
args = vars(ap.parse_args())

img = cv2.imread(args["image"])
#=====================================================


lic_data = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

def plt_show(image, title="", gray = False, size =(100,100)):
    temp = image
    if gray == False:
        temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        plt.title(title)
        plt.imshow(temp, cmap='gray')
        plt.show()


temp = img
gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
number = lic_data.detectMultiScale(img,1.2)
print("number plate detected:"+str(len(number)))
for numbers in number:
    (x,y,w,h) = numbers
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+h]
    cv2.rectangle(temp, (x,y), (x+w,y+h), (0,255,0), 3)
        
#plt_show(temp)
#plt_show(roi_gray)
cropped = roi_gray
    
#Take input of car image with number plate
#img = cv2.imread()
#plt_show(img)

cropped = cv2.resize(cropped,(1000,400))
#========= untuk background putih/hitam beda =================
#retval, image = cv2.threshold(image,150,255, cv2.THRESH_BINARY)
#=============================================================
image = cv2.GaussianBlur(cropped,(11,11),0)
image = cv2.medianBlur(image,9)
#=====================================================
grayPlate = cv2.equalizeHist(image)

cv2.imshow('crop',image)
cv2.imshow('crop2',grayPlate)
cv2.waitKey(0)
cv2.destroyAllWindows()
#=====================================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
text = pytesseract.image_to_string(image, config='--psm 3')
print("Detected Number is:",text)

