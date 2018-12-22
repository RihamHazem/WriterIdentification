import cv2
import math
import numpy as np

def enclosed_regions(line):
    ret, labels = cv2.connectedComponents(line)
   # l=0
   # for label in range(1,ret):
    #    mask = np.array(labels, dtype=np.uint8)
     #   mask[labels == label] = 255
      #  t=np.sum(labels == label)
       # if t >20 and t < 1200:
        #    cv2.imshow("com",mask)
         #   cv2.waitKey()
          #  l=l+1
   # print(l)

    # Map component labels to hue val
    label_hue = np.uint8(179 * labels / np.max(labels))
    label_hue[label_hue<20] = 255
    label_hue[label_hue!=255]=0

    image, contours, hierarchy = cv2.findContours(label_hue,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    f=0
    r=0
    num=0
    size=0
    for contour in contours:
        area=cv2.contourArea(contour)
        l=cv2.arcLength(contour,1)
        if area > 20 and area < 1200:
            num+=1
            f+=4*area*math.pi/(l*l)
            r+=(l*l)/area
            size+=area
    num = max(num, 1)
    return [f/num, r/num, size/num]

