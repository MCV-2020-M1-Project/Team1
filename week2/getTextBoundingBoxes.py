import imageThresholding as imThresh
import glob
import os
import cv2
import imutils
import numpy as np
import argparse
import time
import masks
import math
from PIL import Image

# THIS FILE GENERATES THE OUTPUT IMAGES. call imThresh.imageThresholding(image), returns 3 images (h,l,s).
# Compute getRectContours(img) on a specific channel to get the mask. Running getTextBoundingBoxes.py 
#  generates the output files in '../resources/w2_out'



def getRectContours(img):
    
    contours,h = cv2.findContours(img,cv2.RETR_TREE   ,cv2.CHAIN_APPROX_SIMPLE  ) 
    rectContours = []
    idx = 0
    resultImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  
    hull_list = []
    

    for cnt in contours: 
        x,y,w,h = cv2.boundingRect(cnt)
        resultImg = cv2.rectangle(resultImg,(x,y),(x+w,y+h),(255,255,255),-1)
        
        hull = cv2.convexHull(cnt)
        hull_list.append(hull)
        approx = cv2.approxPolyDP(cnt,0.05*cv2.arcLength(cnt,True),True) 
        if len(approx)==4: 
            idx = idx + 1
            #print ("square") 
            #print( cnt )
            cv2.drawContours(img,[cnt],0,(0,0,255),-1) 
            rectContours.append(cnt)

    # Draw
    finalList = rectContours
    for i in range(len(finalList)):
        #cv2.drawContours(resultImg, contours, i, (0,255,0), 2)
        cv2.drawContours(resultImg, finalList, i, (255,255,255), -1)
    
    resultImg=cv2.cvtColor(resultImg, cv2.COLOR_BGR2GRAY)
    
    resultImg = cv2.morphologyEx(resultImg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (10,5)))
    resultImg = cv2.morphologyEx(resultImg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5,10)))
    resultImg = cv2.cvtColor(resultImg, cv2.COLOR_GRAY2BGR)      


    return  resultImg

if __name__ == '__main__':
    impath = '../resources/qsd1_w2'
    extension = 'jpg'
    images = sorted(glob.glob(os.path.join(impath, '*.'+extension)))

    print(f'number of images in dataset is {len(images)}')

    i = 0

    for im in images:
        image = cv2.imread(im)
        h, l, s = imThresh.imageThresholding(image)
        
        width = 600
        height = 600

        imgResult_h = getRectContours(h)
        h = cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)        
        #result_h = cv2.drawContours(h, contours_h, -1, (0,255,0), 1)

        imgResult_l = getRectContours(l)
        l = cv2.cvtColor(l, cv2.COLOR_GRAY2BGR)
        #result_l = cv2.drawContours(l, contours_l, -1, (0,255,0), 1)

        imgResult_s = getRectContours(s)
        s = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        #result_s = cv2.drawContours(s, contours_s, -1, (0,255,0), 1)

        concat_img = np.concatenate((imThresh.resize_mantain_ratio(
                                        image, width, height),
                                    imThresh.resize_mantain_ratio(
                                        imgResult_h, width, height), 
                                    imThresh.resize_mantain_ratio(
                                        imgResult_l, width, height),
                                    imThresh.resize_mantain_ratio(
                                        imgResult_s, width, height)), axis=1)

        concat_img_noResize = np.concatenate( (image, imgResult_h, imgResult_l, imgResult_s), axis=1)

        print("Image {}".format(i))
        cv2.imwrite('../resources/w2_out/out{}.jpg'.format(i), concat_img_noResize)
        #cv2.imshow('Image', concat_img)
        #cv2.waitKey(0)
        i = i+1
        

    #cv2.destroyAllWindows()
