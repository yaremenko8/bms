from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import zipfile


def clean_image(img, alpha=10, beta1=20, beta2=450, pr_mode=1, kernel_size=2, return_cntrs=False):
    
    """
     
    Parameters
    ----------
    img : np.array
        an image of a chemical compound
    alpha : float
        the maximum allowed difference between bouding rectangle sides ,0 differnce means all sides are equal i.e square
    beta1 : float
        the minimum area of the circle which bound the contour found 
    beta2 : int
        the maximum area of the circle which bound the contour found 
    pr_mode : int
        preprocessing preset value can be 1,2 or 3 
    kernel_size : int
        kernel size for morphological operations in preprocessing  
        
    """
    
    t1 =img.copy().astype(np.uint8)
    #preprocess the image so better contours can be detected
    if pr_mode ==1 :
        t1 = cv2.erode(t1,np.ones((kernel_size,kernel_size)))
    elif pr_mode == 2:
        t1 = cv2.morphologyEx(t1,cv2.MORPH_OPEN,np.ones((kernel_size,kernel_size)))
    else:
        t1 = cv2.morphologyEx(t1,cv2.MORPH_GRADIENT,np.ones((kernel_size,kernel_size)))


    t1 = t1 + 255
    
    #create canvas to draw contours on
    img = cv2.cvtColor(np.zeros_like(t1),cv2.COLOR_GRAY2RGB)
    
    #find contours
    contours, hierarchy = cv2.findContours(t1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )

    cnts, recs = [], []
    
    #function for box area calculation
    barea = lambda x: np.linalg.norm(x[0]-x[1])*np.linalg.norm(x[0]-x[3])
    #function for box side difference calculation
    bsd = lambda x: np.abs(np.linalg.norm(x[0]-x[1])-np.linalg.norm(x[0]-x[3]))

    #preprocess contours
    for cnt in contours:
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        
        #if contour bonding box sides are not more different then alpha
        #and the bounding circle area is between beta1 and beta 2
        #save contour else ignore it
        if (beta1<np.pi *radius**2 < beta2) and bsd(box)<alpha:
            cnts.append(cnt)
            x1 = x - radius
            x2 = x + radius
            y1 = y - radius
            y2 = y + radius
            recs.append(np.array([x1, x2, y1, y2], dtype=np.uint8))
            
    if return_cntrs:
        return recs
    wc = cv2.drawContours(img, cnts, -1, (0,255,0), 2)
    wc =  cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    
    
    return cv2.bitwise_and(t1,t1,mask=wc)


def get_letters(processed_img, scope=2):
        contours, hierarchy = cv2.findContours(processed_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
        letters = []
        for cts in contours:
            rect = cv2.minAreaRect(cts)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            W = rect[1][0]
            H = rect[1][1]
            Xs = [i[0] for i in box]
            Ys = [i[1] for i in box]
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)
            y2 = max(Ys)

            rotated = False
            angle = rect[2]

            if angle < -45:
                angle+=90
                rotated = True
                
            center = (int((x1+x2)/2), int((y1+y2)/2))
            size = (int(scope*(x2-x1)),int(scope*(y2-y1)))
            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)
            cropped = cv2.getRectSubPix(processed_img, size, center)    
            cropped = cv2.warpAffine(cropped, M, size)
            croppedW = W if not rotated else H 
            croppedH = H if not rotated else W
            croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW*scope), int(croppedH*scope)), (size[0]/2, size[1]/2))
            croppedRotated = np.rot90(croppedRotated)

            #print(box)
            letters.append(croppedRotated)
        return letters
        
        
def get_image(bigzip, filename):
    with bigzip.open(filename) as raw:
        im = io.imread(raw)
    return im
    
    
def erode_image(image, kernel=(2,2)):
    return cv2.erode(image, np.ones(kernel))