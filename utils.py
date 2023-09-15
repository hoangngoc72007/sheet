import cv2
import numpy as np

def splitBox(img):
    rows = np.vsplit(img,10)
    print(rows)

def rectCountour(coutours):
    rectCon = []
    for i in coutours:
        area = cv2.contourArea(i)
        #print(area)
        if area > 100:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i,0.03*peri, True) # true la tim duong khep kin
            #neu la hinh chu nhat
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True) # sap xep theo dien tich giam dan
    return rectCon

def rectCountourid(coutours):
    rectCon = []
    for i in coutours:
        area = cv2.contourArea(i)
        if area > 500:
            print(area)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i,0.03*peri, True) # true la tim duong khep kin
            #neu la hinh chu nhat
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True) # sap xep theo dien tich giam dan
    return rectCon


def getConnerPoints(cont):
    #tu contour chuyen ve toa do 4 dinh
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.03 * peri, True)  # true la tim duong khep kin
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape(4,2)
    myNewPoints = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    myNewPoints[0] = myPoints[np.argmin(add)]
    myNewPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myNewPoints[1] = myPoints[np.argmin(diff)]
    myNewPoints[2] = myPoints[np.argmax(diff)]
    #print(f'mynewpoint',myNewPoints)
    return myNewPoints


def sort_contours(cnts, method="left-to-right"):
    return cnts




def splitBoxes(img, top_left_x=0,top_left_y = 0 ):
    print("x,y", top_left_x, top_left_y)
    '''

    :param img:
    :return:
     rows = (np.vsplit(img,10))
    boxes = []
    for r in rows:
        cols = np.hsplit(r,4)
        for box in cols:
            boxes.append(box)
    return boxes
    '''
    rows = (np.vsplit(img,10))
    boxes = []
    boxes1 = []
    coords = []
    for i, row in enumerate(rows) :
        cols = np.hsplit(row,4)
        for j, col in enumerate(cols):
            boxes.append(col)
            # Calculate the coordinates of the part
            x = j * col.shape[1] + top_left_x
            y = i * col.shape[0] + top_left_y
            coords.append((x, y))
            col = cv2.resize(col, (28, 28), cv2.INTER_AREA)
            boxes1.append(col)

    return boxes,coords, boxes1

def splitBoxesIdTest(img):
    rows = (np.vsplit(img,10))
    boxes = []
    for r in rows:
        cols = np.hsplit(r,3)
        for box in cols:
            boxes.append(box)
    return boxes

def splitBoxesIdStudent(img):
    rows = (np.vsplit(img,10))
    boxes = []
    for r in rows:
        cols = np.hsplit(r,6)
        for box in cols:
            boxes.append(box)
    return boxes



def showAnswer(img, myIndex , answer, mypixcelCoords ):
    secW = int (img.shape[1]/4)
    secH = int (img.shape[0]/10)
    for x in range(0,10):
        numb = myIndex[x]
        an = answer[x]
        for i in range(0,4):
            c_x = mypixcelCoords[x][i][0]+20
            c_y = mypixcelCoords[x][i][1]+20
            if numb[i] == 1 and an == i and sum(numb) ==1:
                #dung
                cX = (i * secW) + secW // 2
                cY = (x * secH) + secH // 2
                cv2.circle(img, (c_x,c_y ), 10, (0, 255, 0), 3)
            elif numb[i] == 1:
                #sai
                cX = (i * secW) + secW // 2
                cY = (x * secH) + secH // 2
                #cv2.circle(img, (cX, cY), 10, (0, 0, 255), 3)
                cv2.circle(img, (c_x, c_y), 10, (0, 0, 255), 3)
            elif numb[i] == 1 and an == i:
                cX = (i * secW) + secW // 2
                cY = (x * secH) + secH // 2
                #cv2.circle(img, (cX, cY), 10, (0, 0, 255),3)
                cv2.circle(img, (c_x, c_y), 10, (0, 0, 255), 3)
    return img



