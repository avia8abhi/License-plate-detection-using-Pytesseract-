#!/usr/bin/env python
# coding: utf-8

# In[1]:



import cv2
import numpy as np
import re
import os
from matplotlib import pyplot as plt
import pytesseract
import imutils




# In[1]:


image = cv2.imread('OneDrive/Desktop/numplate_6.jpg')

# Resize the image - change width to 500 using imutils
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("1:Original Image", image)
cv2.waitKey(0)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("2: Grayscale Conversion", gray)
cv2.waitKey(0)

# Noise removal with iterative bilateral filter(removes noise along with edges marked)
bilat= cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("3:  Bilateral Filter", gray)
cv2.waitKey(0)

# Find Edges of the grayscale image
edged = cv2.Canny(bilat, 100, 400)
cv2.imshow("4: Canny Edges", edged)
cv2.waitKey(0)

# Find contours based on Edges detected
cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create copy of original image to draw all contours
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("5: All Contours", img1)
cv2.waitKey(0)

#sort contours based on their area keeping minimum required area as '20' (anything smaller than this will not be considered)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
NumberPlateCnt = None #we currently have no Number plate contour
# Top 20 Contours to find number plate
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("6: Top 20 Contours", img2)
cv2.waitKey(0)

# loop over our contours to find the best possible approximate contour of number plate
#To find the topmost area that is rectangular and that area is the largest of all areas with rectangular in shape
count = 0
increment =7
for c in cnts:
        perimeter = cv2.arcLength(c, True) #Getting perimeter of each contour
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            break


# Drawing the selected contour on the original image
#print(NumberPlateCnt)
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 0)
cv2.imshow("7: Final Image With Number Plate Detected", image)
cv2.waitKey(0)

#Masking entire image except the plate , convert it into zeros array
masked = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(masked,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=masked)
#new_img= Image.open('plate.jpg')
cv2.imshow("8: Final_output",new_image)     #The final image showing only the number plate.
cv2.waitKey(0)
print(new_image.shape)

(x,y) = np.where(masked==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))


# In[6]:



def recognize_plate(folder):
    # separate coordinates from box
    # grayscale region within bounding box
    img=cv2.imread(folder)
    # grayscale region within bounding box
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # resize image to three times as large as original for better readability
    #box=cv2.resize(box,(400,150))
    gray = cv2.resize(gray, None, fx = 3, fy = 3, interpolation = cv2.INTER_CUBIC)
    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imshow("Otsu Threshold", thresh)
    cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
    #cv2.imshow("Dilation", dilation)
    #cv2.waitKey(0)
    # find contours of regions of interest within license plate
    try:
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = img.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 0.2: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 28: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        # perfrom bitwise not to flip image to black text on white background
        roi = cv2.bitwise_not(roi)
        # perform another blur on character region
       
        try:
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            plate_num += clean_text
        except: 
            text = None
    if plate_num != None:
        print("License Plate #: ", plate_num)
    cv2.imshow("Character's Segmented", im2)
    cv2.waitKey(0)

recognize_plate("OneDrive/Desktop/t.png")


# In[4]:


image = cv2.imread('OneDrive/Desktop/numplate_6.jpg')

# Resize the image - change width to 500 using imutils
image = imutils.resize(image, width=500)

# Display the original image
cv2.imshow("1:Original Image", image)
cv2.waitKey(0)

# RGB to Gray scale conversion
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("2: Grayscale Conversion", gray)
cv2.waitKey(0)

# Noise removal with iterative bilateral filter(removes noise along with edges marked)
bilat= cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("3:  Bilateral Filter", gray)
cv2.waitKey(0)

# Find Edges of the grayscale image
edged = cv2.Canny(bilat, 100, 400)
cv2.imshow("4: Canny Edges", edged)
cv2.waitKey(0)

# Find contours based on Edges detected
cnts, new  = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create copy of original image to draw all contours
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0,255,0), 3)
cv2.imshow("5: All Contours", img1)
cv2.waitKey(0)

#sort contours based on their area keeping minimum required area as '20' (anything smaller than this will not be considered)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:20]
NumberPlateCnt = None #we currently have no Number plate contour
# Top 20 Contours to find number plate
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("6: Top 20 Contours", img2)
cv2.waitKey(0)

# loop over our contours to find the best possible approximate contour of number plate
#To find the topmost area that is rectangular and that area is the largest of all areas with rectangular in shape
count = 0
increment =7
for c in cnts:
        perimeter = cv2.arcLength(c, True) #Getting perimeter of each contour
        approx = cv2.approxPolyDP(c, 0.01 * perimeter, True)
        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx #This is our approx Number Plate Contour
            break


# Drawing the selected contour on the original image
#print(NumberPlateCnt)
cv2.drawContours(image, [NumberPlateCnt], -1, (0,255,0), 0)
cv2.imshow("7: Final Image With Number Plate Detected", image)
cv2.waitKey(0)

#Masking entire image except the plate , convert it into zeros array
masked = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(masked,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=masked)
#new_img= Image.open('plate.jpg')
cv2.imshow("8: Final_output",new_image)     #The final image showing only the number plate.
cv2.waitKey(0)
print(new_image.shape)

(x,y) = np.where(masked==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+1, y1:y2+1]

plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
 
    
def recognize_plate(box):    
    l=list()
    box=cv2.resize(box,(400,150))

    # perform gaussian blur to smoothen image
    blur = cv2.GaussianBlur(box, (5,5), 0)
    # perform gaussian blur to smoothen image
    #cv2.imshow("Gray", gray)
    #cv2.waitKey(0)
    # threshold the image using Otsus method to preprocess for tesseract
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    cv2.imshow("Otsu Threshold", thresh)
    cv2.waitKey(0)
    # create rectangular kernel for dilation
    rect_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # apply dilation to make regions more clear
    dilation = cv2.dilate(thresh, rect_kern, iterations = 1)
 
    cv2.imshow("preprocessed img", thresh)
    cv2.waitKey(0)

    try:
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        ret_img, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours left-to-right
    sorted_contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])
    # create copy of gray image
    im2 = box.copy()
    # create blank string to hold license plate number
    plate_num = ""
    # loop through contours and find individual letters and numbers in license plate
    for cnt in sorted_contours:
        x,y,w,h = cv2.boundingRect(cnt)
        height, width = im2.shape
        # if height of box is not tall enough relative to total height then skip
        if height / float(h) > 6: continue

        ratio = h / float(w)
        # if height to width ratio is less than 1.5 skip
        if ratio < 1: continue

        # if width is not wide enough relative to total width then skip
        if width / float(w) > 28: continue

        area = h * w
        # if area is less than 100 pixels skip
        if area < 100: continue

        # draw the rectangle
        rect = cv2.rectangle(im2, (x,y), (x+w, y+h), (0,255,0),2)
        # grab character region of image
        roi = thresh[y-5:y+h+5, x-5:x+w+5]
        roi = cv2.bitwise_not(roi)
        cv2.imshow("Character's Segmented", roi)
        cv2.waitKey(0)

        #perfrom bitwise not to flip image to black text on white background
        # perform another blur on character region

        try:
            text = pytesseract.image_to_string(roi, config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            # clean tesseract text by removing any unwanted blank spaces
            clean_text = re.sub('[\W_]+', '', text)
            l.append(clean_text)
        except:
            text = None
    if l != None:
        print("License Plate #: ", l)
    cv2.imshow("Character's Segmented", im2)
    cv2.waitKey(0)
    
recognize_plate(cropped_image)    

