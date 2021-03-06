import cv2
import imutils
import numpy as np
import pytesseract
import argparse
from PIL import Image
import  urllib

#=====================================================
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
#=====================================================

#image = cv2.imread('6.jpg')

#=====================================================
cv2.imshow('raw',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#=====================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grey scale
gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]

screenCnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
    detected = 1
if detected == 1:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(image,image,mask=mask)
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]
Cropped = cv2.resize(Cropped,(1000,400))

#=====================================================
cv2.imshow('crop',Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
#=====================================================

#========= untuk background putih/hitam beda =================
retval, image = cv2.threshold(image,150,255, cv2.THRESH_BINARY)
#=============================================================
image = cv2.GaussianBlur(image,(11,11),0)
image = cv2.medianBlur(image,9)

#=====================================================
cv2.imshow('crop',Cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()
#=====================================================

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
text = pytesseract.image_to_string(Cropped, config='--psm 3')
print("Detected Number is:",text)


