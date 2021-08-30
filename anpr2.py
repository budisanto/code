import cv2
import imutils
import numpy as np
import pytesseract
import argparse
import  urllib
from PIL import Image, ImageEnhance, ImageFilter

#=====================================================
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True,help="path to the input image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
#=====================================================

#image = cv2.imread('6.jpg')

#=====================================================
cv2.imshow('input',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#=====================================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grey scale

#blur = cv2.GaussianBlur(gray, (3,3), 0)
#thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

#=====================================================
#cv2.imshow('gray-blur',blur)
#cv2.imshow('threshold',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
#=====================================================

gray = cv2.bilateralFilter(gray, 11, 17, 17) #Blur to reduce noise
edged = cv2.Canny(gray, 30, 200) #Perform Edge detection
cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
#=====================================================
cv2.imshow('gray-blur',gray)
cv2.imshow('edge',edged)
cv2.waitKey(0)
cv2.destroyAllWindows()
#=====================================================
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
cv2.imshow('gray-blur',gray)
cv2.imshow('edge',edged)
cv2.imshow('Cropped-resize',Cropped)
cv2.imshow('Contour-bitwise',new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#=====================================================

#im = Image.open("temp.jpg") # the second one 
#im = Cropped
##im = im.filter(ImageFilter.MedianFilter())
#enhancer = ImageEnhance.Contrast(im)
#im = enhancer.enhance(2)
#im = im.convert('1')

text = pytesseract.image_to_string(Cropped, config='--psm 3')
print("Detected Number is:",text)

#im.save('temp2.jpg')
#text = pytesseract.image_to_string(Image.open('temp2.jpg'))
#print(text)