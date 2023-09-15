import json
import os
from random import random
import uuid

import cv2
import numpy as np
import math
import utils
from flask import Flask, request, Response
#from model import CNN_Model

def ResizeImage(img, height=800):
    rat = height / img.shape[0]
    width = int(rat * img.shape[1])
    dst = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return dst

'''
def FindAnchors(copy_img):
    src_hsv = cv2.cvtColor(copy_img, cv2.COLOR_BGR2HSV)
    src_hsv = cv2.blur(src_hsv,(5,5))
    edges = cv2.inRange(src_hsv,(0,0,0),(180,255,50))  # lam bien mat tat ca noi dung phia trong chi hien thi cac cham vuong tren man hinh
    cv2.imshow("hsv", edges)
    #Find contours
    contours , hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    if len(contours) > 0:
        anchors = []
        for cnt in contours:
            perimeter =0.03 * cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,perimeter, True)
            area = cv2.contourArea(cnt)
            #print(f"gia tri approx {approx}")
            #print(f"gia tri area {area}")
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if len(approx) == 4 and aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                anchors.append(approx)

    return anchors
'''

def find_anchor2(img):
    anchors = []
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(img, contours, -1, (225,0,0), 3)
    #cv2.imshow("ancho", img)
    # Iterate through each contour and find the rectangular contour
    for cnt in contours:
        # Find the perimeter of the contour
        perimeter = cv2.arcLength(cnt, True)
        # Approximate the contour to a polygon
        approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)
        # Check if the polygon is a square or rectangle
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)

            aspect_ratio = float(w) / h
            # Check if the aspect ratio is close to 1
            if aspect_ratio >= 0.9 and aspect_ratio <= 1.1:
                # Check if the square is dark
                roi = gray[y:y + h, x:x + w]
                mean = cv2.mean(roi)[0]
                if mean < 100:
                    print(mean)
                    anchors.append(approx.reshape(-1,2))
                    # Draw the square on the image
                    #cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)

    # Display the image
    #cv2.imshow('image', img)
    #cv2.waitKey()
    return anchors



def FindAnchors(img, Area=np.inf, deltaArea=np.inf):
    anchors = []
    src_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    src_HSV = cv2.blur(src_HSV, (5, 5))
    edges = cv2.inRange(src_HSV, (0, 0, 0), (255, 255, 50))
    cv2.imshow("edt", edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        area = cv2.contourArea(contour)
        print(area, "dien tihch")
        if area > 10:
            if Area == np.inf or deltaArea == np.inf:
                ok = len(approx) == 4 and cv2.isContourConvex(approx)
            else:
                ok = len(approx) == 4 and (area > Area - deltaArea) and (
                            area < Area + deltaArea) and cv2.isContourConvex(
                    approx)
            if ok:
                anchors.append(approx.reshape(-1, 2))

    print(len(anchors))
    return anchors


def PointIntersecsion(points):
    M = cv2.moments(points)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # giao nhau
    return (cx,cy)

def ListPointIntersection(contours):
    dst = []
    for i in contours:
        point = PointIntersecsion(i)
        dst.append(point)
    return dst
def Distance(p1,p2) -> float:
    dis  = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return math.sqrt(dis)

def ClusterPoints(points, distance = 3):
    #gom nhom cac diem lai va tinh khoang cach trung binh sau do tra ve lai cac diem
    dst =[]
    while points:
        count = 1
        p = points[0]
        sumX = p[0]
        sumY = p[1]
        points.pop(0)
        i = 1
        while i<len(points):
            if p == points[i]:
                points.pop(i)
            elif Distance(p, points[i]) < distance:
                sumY += points[i][0]
                sumY += points[i][1]
                count += 1
                points.pop(i)
            else:
                i +=1
        dst.append((int(sumX / count), int(sumY / count)))

    return dst

def find_position_min(array):
    vt = 0
    min_value = array[0]
    for i in range(1, len(array)):
        if array[i] < min_value:
            min_value = array[i]
            vt = i
    return vt

def find_position_max(array):
    vt = 0
    max_value = array[0]
    for i in range(1, len(array)):
        if array[i] > max_value:
            max_value = array[i]
            vt = i
    return vt


def SortCornerPoints(points):
    print(f'point ',(points))
    _pts = []
    _sum = np.zeros(len(points))
    _diff = np.zeros(len(points))
    for i in range(len(points)):
        _sum[i] = points[i][0] + points[i][1]
        _diff[i] = points[i][0] - points[i][1]
    x1 = find_position_min(_sum)
    x2 = find_position_min(_diff)
    print(_sum)
    print(_diff)
    print(f'gia tri nho nhat {x1}')
    print(f'gia tri nho nhat {x2}')
    x3 = find_position_max(_sum)
    x4 = find_position_max(_diff)
    print(f'gia tri lon nhat {x3}')
    print(f'gia tri lon nhat {x4}')
    '''
    /*the top-left point will have the smallest sum, whereas
	the bottom - right point will have the largest sum
	now, compute the difference between the points, the
	top - right point will have the smallest difference,
	 whereas the bottom - left will have the largest difference*/
    '''
    _pts.append(points[x1])
    _pts.append(points[x2])
    _pts.append(points[x3])
    _pts.append(points[x4])
    print(_pts)
    return _pts

def find_Point_InRegion(points, condition1,condition2,axis):
    point = np.array([0, 0], dtype=np.float32)
    #axit 0 va 1, 0 theo y , 1 theo truc x
    for i in range(0,len(points)):
        if axis==1:
            #tim point trong khoang denta y
            average = (condition1[1] + condition2[1])/ 2.0
            delta = abs(condition1[1] - condition2[1]) + 5
            print(f'gia tri trung binh',average)
            print(f'dental',delta)
            if points[i][1] == average:
                point = points[i]
                break
            else:
                ok = points[i][1] > (average - delta) and points[i][1] < (average+delta)
                if ok:
                    #lay duoc diem dau tien thoa man va thoat vong lap
                    point = points[i]
                    break
        else:
            #tìm point trong khoảng delta x
            average = (condition1[0] + condition2[0]) / 2.0
            delta = abs(condition1[0] - condition2[0]) + 5
            print(f'gia tri trung binh theo truc ngang', average)
            print(f'dental theo ngang', delta)
            if points[i][0] == average:
                point = points[i]
                break
            else:
                ok = points[i][0] > (average - delta) and points[i][0] < (average + delta)
                if ok:
                    point = points[i]
                    print(point)
                    break
    return point


def SortPoints(points,  axis = 0):
    def x_compare(point):
        return point[0]

    def y_compare(point):
        return point[1]

    if axis == 0:
        points = sorted(points, key=y_compare)
        points = sorted(points, key=x_compare)
    elif axis == 1:
        points = sorted(points, key=x_compare)
        points = sorted(points, key=y_compare)
    return points



def SortAnchors(points, img_src):
    dst = []
    conners = []
    '''
    conners = SortCornerPoints(points)  # lay 4 diem xung quanh
    for i in range(0, len(conners)):
        points.remove(conners[i])
        '''
    #conner 1
    conners.append((0,0))
    conners.append ((0 , img_src.shape[0]))
    conners.append((img_src.shape[1],0))
    conners.append((img_src.shape[1], img_src.shape[0]))
    print(f'truoc khi tim', points)
    dst.append(conners[0])
    print('goc 0', conners[0])
    print('goc 1', conners[1])
    for i in range(0,12):
        point = find_Point_InRegion(points, conners[0], conners[1], 1)
        dst.append(point)
        points.remove(point)
    #corner 3
    '''
    dst.append(conners[1])

    for i in range(0,6):
        point = find_Point_InRegion(points, conners[1], conners[2], 0)
        dst.append(point)
        points.remove(point)

    print(f'truoc khi tim conner5', points)
    '''
    dst.append(conners[1])
    dst.append(conners[2])
    dst.append(conners[3])
    return dst

def TranformPoints(points, offset,rat):
    for i in range(len(points)):
        points[i] = (points[i][0]*rat +offset[0], points[i][1]*rat + offset[1])
    return points

def sub_rect_image(img_src,indentityRegion):
    rect = cv2.boundingRect(np.array(indentityRegion,dtype=np.int32))
    sub_image = img_src[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]].copy()
    return sub_image

def FindRectangles(img, fromThreshold, toThreshold, ratioArea=0):
    rectangles = []
    src_gray = cv2.GaussianBlur(img, (3, 3), 0)
    src_gray = cv2.cvtColor(src_gray, cv2.COLOR_BGR2GRAY)
    dst1 = cv2.bilateralFilter(src_gray, 9, 75, 75)
    cv2.adaptiveThreshold(dst1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 2)
    cv2.medianBlur(dst1, 11)
    edges = cv2.Canny(src_gray, fromThreshold, toThreshold, 3)
    cv2.imshow("edg", edges)
    Max_Area_Find = (img.shape[1] - 10) * (img.shape[0] - 10)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.03 * perimeter, True)
        area = cv2.contourArea(contours[i])
        ok = False
        if ratioArea == 0:
            ok = len(approx) == 4 and cv2.isContourConvex(approx)
        else:
            ok = len(approx) == 4 and cv2.isContourConvex(approx)  and area >= ratioArea * Max_Area_Find
        if ok:
            rectangles.append(approx)
    return rectangles

def SortRectangles(rect , axis = 0):
    rectangles = []
    centerPoints = []
    for i in range(len(rect)):
        p = PointIntersecsion(rect[i])
        print(p)
        centerPoints.append(p)
    _copy = centerPoints.copy()
    _copy = ClusterPoints(_copy)
    _copy = SortPoints(_copy,axis)
    print(f"gia tri sau khi sort", centerPoints)
    print(f"gia tri sau khi sort",_copy)
    for i in range(len(_copy)):
        for j in range(len(centerPoints)):
            if (_copy[i] == centerPoints[j]):
                rectangles.append(rect[j])
                centerPoints.pop(j)
                break
            else:
                if (Distance(_copy[i],centerPoints[j]) <3):
                    rectangles.append(rect[j])
                    centerPoints.pop(j)
                    break

    return rectangles

def SortCornerPoints2(points):
    # Tính trung bình của các điểm để xác định điểm giữa
    center = np.mean(points, axis=0)
    sorted_points = []
    for point in points:
        if point[0] < center[0] and point[1] < center[1]:
            sorted_points.append(point)  # Top-left corner
        elif point[0] > center[0] and point[1] < center[1]:
            sorted_points.append(point)  # Top-right corner
        elif point[0] > center[0] and point[1] > center[1]:
            sorted_points.append(point)  # Bottom-right corner
        elif point[0] < center[0] and point[1] > center[1]:
            sorted_points.append(point)  # Bottom-left corner
    return sorted_points

def TransformImage(img, points):
    print(f"point trong tranform",points[0])
    newHeight = max(np.linalg.norm(points[0][1] - points[3][1]), np.linalg.norm(points[1][1] - points[2][1]))
    newWidth = max(np.linalg.norm(points[0][0] - points[1][0]), np.linalg.norm(points[2][0] - points[3][0]))
    _dst = np.array([(0, 0), (newWidth - 1, 0), (newWidth - 1, newHeight - 1), (0, newHeight - 1)], dtype=np.float32)
    _M = cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), _dst)
    _Warp = cv2.warpPerspective(img, _M, (int(newWidth), int(newHeight)))
    return _Warp




def SeriImage(src, rects):
    serious = []
    for rect in rects:
        lst = [tuple(coord[0]) for coord in rect.tolist()]
        cornners = SortCornerPoints(lst)
        M = TransformImage(src, cornners)
        serious.append(M)
    return serious

def pro_answer(subAnserImg, imgfinal):

    imgContor = subAnserImg.copy()
    imgBiggestCon = subAnserImg.copy()
    #cv2.imshow("khuvutraloi", subAnserImg)
    width = subAnserImg.shape[1]
    height = subAnserImg.shape[0]
    print(f"chieu rong va cao", width, height)

    imgGray = cv2.cvtColor(subAnserImg, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 20)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContor,contours, -1 , (0,255,0),2)
    cv2.imshow("imgcontoir", imgContor)
    cv2.waitKey()
    rectCont = utils.rectCountour(contours)
    # duyet qua tung hinh vuong lay duoc toa do 4 diem cua hinh vuong
    # sort lai theo thu tu tu trai sang
    list_Mybiggescontour = []
    index_big = []
    for i in range(len(rectCont)):
        biggestContour = utils.getConnerPoints(rectCont[i])  # get dc contour lon nha va chuyen ve toa do 4 diem
        # print(f'chieu dai cua big',len(biggestContour))
        list_Mybiggescontour.append(biggestContour)
        index_big.append(biggestContour[0][0][0])

    sorted_list = sorted(index_big)
    print(sorted_list)
    img_listwrap = []
    for x in range(0,4):
        # cv2.drawContours(imgBiggestCon, list_Mybiggescontour[index_big.index(value)], -1, (255, 0, 0), 10)
        # cv2.imshow("bigg", imgBiggestCon)
        biggestContour = utils.reorder(list_Mybiggescontour[index_big.index(sorted_list[x])])

        biggestContour[0][0]=[biggestContour[0][0][0]+4,biggestContour[0][0][1]+4]
        print(f'big', biggestContour[0][0])
        pt1 = np.float32(biggestContour)
        pt2 = np.array([[0, 0], [300, 0], [0, 600], [300, 600]], dtype=np.float32)
        # Tính toán ma trận biến đổi phối cảnh
        M = cv2.getPerspectiveTransform(pt1, pt2)
        warped_img = cv2.warpPerspective(subAnserImg, M, (300, 600))
        #cv2.imshow("wrap" + str(x), warped_img)
        findChoiceAndAnswer(warped_img,"wrap" + str(x+1), pt1,pt2, imgfinal)

def findIdTest(src):
    y = 20
    x = 15
    h = 350
    w = 120
    src = src[y:y + h, x:x + w]
    print("w", src.shape[1], src.shape[0])
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("th", imgThresh)
    boxes = utils.splitBoxesIdTest(imgThresh)
    mypixcelVal = np.zeros((10, 3))
    countC = 0
    countR = 0
    # cv2.waitKey()
    for image in boxes:
        totalPixcel = cv2.countNonZero(image)
        mypixcelVal[countR, countC] = totalPixcel
        countC += 1
        if countC == 3:
            countR += 1
            countC = 0
    print(mypixcelVal)
    idTest = ""
    print(mypixcelVal)
    for cols in range(0, 3):
        for rows in range(0,10):
            arr = mypixcelVal[rows][cols]
            if arr>400:
                idTest = idTest+str(rows)+"_"
    return idTest


def findIdStudent(src):
    y = 20
    x = 15
    h = 350
    w = 228
    src = src[y:y + h, x:x + w]
    #print("w", src.shape[1], src.shape[0])
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("th", imgThresh)
    #cv2.waitKey()
    boxes = utils.splitBoxesIdStudent(imgThresh)
    mypixcelVal = np.zeros((10, 6))
    countC = 0
    countR = 0
    # cv2.waitKey()
    for image in boxes:
        totalPixcel = cv2.countNonZero(image)
        mypixcelVal[countR, countC] = totalPixcel
        countC += 1
        if countC == 6:
            countR += 1
            countC = 0
    print("st",mypixcelVal)
    idStudent = ""
    for cols in range(0, 6):
        for rows in range(0,10):
            arr = mypixcelVal[rows][cols]
            if arr>450:
                idStudent = idStudent+str(rows)+"_"
    print(idStudent)
    return idStudent




def findChoiceAndAnswer(kv, src):
    y = 20
    x = 15
    h = 350
    w = 152
    src = src[y:y + h, x:x + w]
    print("w", src.shape[1] , src.shape[0])
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(gray,140,255,cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("th", imgThresh)
    boxes = utils.splitBoxes(imgThresh)
    mypixcelVal = np.zeros((10,4))
    countC = 0
    countR = 0
    #cv2.waitKey()
    for image in boxes:
        totalPixcel = cv2.countNonZero(image)
        mypixcelVal[countR,countC] = totalPixcel
        countC +=1
        if countC == 4:
            countR +=1
            countC =0

    myIndex = []
    #print(mypixcelVal)
    for x in range(0,10):
        arr = mypixcelVal[x]
        index=[]
        for val in arr:
            if val >500:
                index.append(1)
            else:
                index.append(0)
        myIndex.append(index)
    print(myIndex)
    answer = []
    right_answer=0
    for i in range(0, 10):
        answer.append(1)
    print(answer)
    for ls in range(len(myIndex)):
        if sum(myIndex[ls])==1:
            if answer[ls] == np.argmax(myIndex[ls]):
                print('cau dung',ls+1)
                right_answer +=1
        elif sum(myIndex[ls])==0 :
            pass
            #print("khong chon d a")

        else:
            pass
            #print("chon hai dap an")
    #cv2.waitKey()
    m = utils.showAnswer(src,myIndex,answer)
    return m
    imgdraw = np.zeros_like(src)
    #imgdraw = utils.showAnswer(imgdraw, myIndex, answer)
    #M = cv2.getPerspectiveTransform(pt2, pt1)
    #warped_img = cv2.warpPerspective(imgdraw, M, (300, 600))
    #imgfinal = cv2.addWeighted(imgfinal,1,warped_img,1,0)

    #cv2.imshow(index_str, imgfinal)



def find_select_choice(imgsrc, anchor, right_answer, answer, positon):
    y = int(20+anchor[1])
    print("y ", y)
    x = int(15+anchor[0])
    h = 350
    w = 152
    roi = imgsrc[y:y + h,  x:x + w]
    print("w", roi.shape[0] , roi.shape[1])
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(gray,140,255,cv2.THRESH_BINARY_INV)[1]
    #cv2.imshow("th", imgThresh)
    boxes, coords, boxes1 = utils.splitBoxes(imgThresh,x ,y)
    mypixcelVal = np.zeros((10,4))
    mypixcelCoords = np.zeros((10, 4),dtype=list)
    countC = 0
    countR = 0


    list_ans = np.array(boxes1)
    #scores = model.predict_on_batch(list_ans / 255.0)
    #cv2.waitKey()

    for i, image in enumerate(boxes) :
        #totalPixcel = cv2.countNonZero(image)
        #mypixcelVal[countR,countC] = totalPixcel
        x,y = coords[i]
        mypixcelCoords[countR,countC] = x,y
        countC +=1
        if countC == 4:
            countR +=1
            countC =0


    myIndex = []
    #print("mypicxecll",mypixcelVal)
    #print("mypicxecll", mypixcelCoords)

    for x in range(0,10):
        arr = mypixcelVal[x]
        index=[]
        for val in arr:
            if val >500:
                index.append(1)
            else:
                index.append(0)
        myIndex.append(index)
    print(myIndex)


    ''' 
    for x in range(0,10):
        index = []
        for row in range(x*4,(x*4)+4):
            if scores[row][1]>0.95:
                index.append(1)
            else:
                index.append(0)
        myIndex.append(index)
    '''

    print(myIndex)
    for ls in range(len(myIndex)):
        if sum(myIndex[ls])==1:
            if answer[ls+positon] == np.argmax(myIndex[ls]):
                print('cau dung',ls+1+positon)
                right_answer +=1
        elif sum(myIndex[ls])==0 :
            pass
            #print("khong chon d a")

        else:
            pass
            #print("chon hai dap an")
    m = utils.showAnswer(imgsrc,myIndex,answer,mypixcelCoords)
    return right_answer

    #imgdraw = np.zeros_like(src)
    #imgdraw = utils.showAnswer(imgdraw, myIndex, answer)
    #M = cv2.getPerspectiveTransform(pt2, pt1)
    #warped_img = cv2.warpPerspective(imgdraw, M, (300, 600))
    #imgfinal = cv2.addWeighted(imgfinal,1,warped_img,1,0)

    #cv2.imshow(index_str, imgfinal)

def SubCircleImage(src, center, radius):
    x_offset = int(center[0] - radius)
    y_offset = int(center[1] - radius)
    width = int(2 * radius)
    height = int(2 * radius)
    subImage = src[y_offset:y_offset + height, x_offset:x_offset + width].copy()
    return subImage

def FindRatioWhiteRegion(src, threshold=185):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_gray = cv2.blur(src_gray, (3, 3))
    _, src_gray = cv2.threshold(src_gray, threshold, 255, cv2.THRESH_BINARY)
    white = cv2.countNonZero(src_gray)
    ratio = white / (src_gray.shape[0] * src_gray.shape[1])
    return ratio



def find_choice(src, tickcontours, img):
    correct = 0
    for (q, i) in enumerate(np.arange(0, len(tickcontours), 4)):

        # Dinh nghia mau rieng cho tung cau hoi
        color = ((100,100,100), (0,255,0), (255,255,100))
        # Sap xep cac contour theo cot
        cnts = utils.sort_contours(tickcontours[i:i + 4])[0]
        #cv2.drawContours(src, cnts, -1, color[0], 3)
        choice = (0, 0)
        total = 0
        # Duyet qua cac contour trong hang
        for (j, c) in enumerate(cnts):

            # Tao mask de xem muc do to mau cua contour
            mask = np.zeros(img.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(img, img, mask=mask)
            total = cv2.countNonZero(mask)
            # Lap de chon contour to mau dam nhat
            if total > choice[0]:
                choice = (total, j)

        # Lay dap an cua cau hien tai
        current_right = 0
        # Kiem tra voi lua chon cua nguoi dung
        if current_right == choice[1]:
            # Neu dung Thi to mau xanh
            color = (0, 255, 0)
            correct += 1
        else:
            # Neu sai Thi to mau do
            color = (0, 0, 255)
        # Ve ket qua len anh
        cv2.drawContours(src, [cnts[current_right]], -1, color, 3)
        cv2.imshow("hieu",src)

def find_circle(img, adaptiveThreshold, Area=np.inf, deltaArea=np.inf):
    circle = []
    y = 10
    x = 10
    h = img.shape[0]-20
    w = img.shape[1]-20
    img = img[y:y + h, x:x + w]
    temp = img.copy()
    img = cv2.GaussianBlur(img, (3, 3), 0)
    dst = cv2.bilateralFilter(img, 7, 75, 75)
    img = dst
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 65, 10)
    img = cv2.bitwise_not(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(temp,contours,-1, (0,255,0), 2)

    for i in range(len(contours)):
        perimeter = cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], 0.03 * perimeter, True)
        area = cv2.contourArea(contours[i])
        if np.isinf(Area) or np.isinf(deltaArea):
            ok = len(approx) >= 6 and cv2.isContourConvex(approx)
        else:
            ok = len(approx) >= 6 and (area > Area - deltaArea) and (area < Area + deltaArea) and cv2.isContourConvex(approx)
        if ok:
            circle.append(contours[i])

    CenterPoints = ListPointIntersection(circle)
    CenterPoints = ClusterPoints(CenterPoints, 3)
    CenterPoints = SortPoints(CenterPoints, 1)

    print(len(CenterPoints))
    for cent in CenterPoints:
        cv2.rectangle(temp, cent,(cent[0]+1, cent[1]+1),(255,255,0),5 )
    cv2.imshow("find", temp)
    print(CenterPoints)
    return CenterPoints


def pro_id(subIdImg):
    #cv2.imshow("id", subIdImg)
    imgContor = subIdImg.copy()
    width = subIdImg.shape[1]
    height = subIdImg.shape[0]
    imgGray = cv2.cvtColor(subIdImg, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 10, 20)
    contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rectCont = utils.rectCountourid(contours)
    list_Mybiggescontour = []
    index_big = []
    for i in range(0,2):
        biggestContour = utils.getConnerPoints(rectCont[i])  # get dc contour lon nha va chuyen ve toa do 4 diem
        # print(f'chieu dai cua big',len(biggestContour))
        list_Mybiggescontour.append(biggestContour)
        index_big.append(biggestContour[0][0][0])
    print(index_big)
    sorted_list = sorted(index_big)
    for value in sorted_list:
        #cv2.drawContours(imgContor, list_Mybiggescontour[index_big.index(value)], -1, (255, 0, 0), 10)
        #cv2.imshow("bigg", imgContor)
        biggestContour = utils.reorder(list_Mybiggescontour[index_big.index(value)])
        pt1 = np.float32(biggestContour)
        # print(f'big', biggestContour)
        pt2 = np.array([[0, 0], [300, 0], [0, 600], [300, 600]], dtype=np.float32)
        # Tính toán ma trận biến đổi phối cảnh
        M = cv2.getPerspectiveTransform(pt1, pt2)
        warped_img = cv2.warpPerspective(subIdImg, M, (300, 600))



def detect_buble(img):

    img = ResizeImage(img, 800)
    print("chieu rong", img.shape[1], "chieu cao", img.shape[0])
    imgfinal = img.copy()
    #cv2.imshow("file goc", img)
    #tim cac hinh vuong trong hinh
    contours= find_anchor2(img)
    #contours  = FindAnchors(img)
    listPoint = ListPointIntersection(contours)
    print(listPoint)  # lay tat ca cac hinh vuong trong hin
    print(sorted(listPoint))
    '''
    for cnt in listPoint:
        cv2.rectangle(img,(cnt),(cnt[0]+1, cnt[1]+1),(255,255,255),5)
    cv2.imshow("list", img)
    cv2.waitKey()
    '''
    # cam
    listPoint =  ClusterPoints(listPoint,3)
    print("list p ", listPoint)
    print("chieudai", len(listPoint))
    sorted_points = sorted(listPoint, key=lambda x: (x[0]))
    # add answers
    answer = []
    for i in range(0, 40):
        answer.append(2)  # 0 dap an a , 1 dap an b, 2 dap an c, 3 dap an d
    right_answer = 0

    if len(sorted_points) == 12:
        anchors = SortAnchors(sorted_points, img)  # sort tat ca cac hinh vuong theo thu tu
        for cnt in anchors:
            cv2.rectangle(img, (cnt), (cnt[0] + 1, cnt[1] + 1), (0, 255, 255), 5)
        #cv2.imshow("sortconner", img)
        anchors = TranformPoints(anchors, (0, 0), img.shape[0] / 800.0)  # height 0 width 1
        print('anchor', anchors)
        print(f'ham sau khi chuyen doi', len(anchors))
        for i in range(0, len(anchors) - 1):
            if anchors[i][1] > anchors[i + 1][1] and abs(anchors[i][1] - anchors[i + 1][1]) > 10 and abs(anchors[i][0] - anchors[i + 1][0]) < 10:
                temp = anchors[i]
                anchors[i] = anchors[i + 1]
                anchors[i + 1] = temp
        print('anchor', (anchors))
        kv1 = [anchors[4], anchors[1]]
        kv2 = [anchors[6], anchors[3]]
        kv3 = [anchors[8], anchors[5]]
        kv4 = [anchors[10], anchors[7]]
        kv5 = [anchors[12], anchors[9]]
        kv6 = [(anchors[15][0], anchors[12][1]), (anchors[11][0] - 5, anchors[11][1])]

        img_kv1 = sub_rect_image(img, kv1)
        img_kv2 = sub_rect_image(img, kv2)
        img_kv3 = sub_rect_image(img, kv3)
        img_kv4 = sub_rect_image(img, kv4)
        img_kv5 = sub_rect_image(img, kv5)
        img_kv6 = sub_rect_image(img, kv6)
        img_answer = []
        img_answer.append(img_kv3)
        img_answer.append(img_kv4)
        img_answer.append(img_kv5)
        img_answer.append(img_kv6)
            ### xu li idde
        id_test = findIdTest(img_kv1)

            ### xu li sbd
        id_student = findIdStudent(img_kv2)
        cv2.putText(imgfinal, id_test, (int(anchors[1][0]), int(anchors[1][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(imgfinal, id_student, (int(anchors[3][0]), int(anchors[3][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 0, 0), 2, cv2.LINE_AA)
        r1 = find_select_choice(imgfinal, anchors[5], right_answer, answer, 0)
        r2 = find_select_choice(imgfinal, anchors[7], right_answer, answer, 10)
        r3 = find_select_choice(imgfinal, anchors[9], right_answer, answer, 20)
        r4 = find_select_choice(imgfinal, anchors[11], right_answer, answer, 30)
        '''
        r1 = find_select_choice(imgfinal, anchors[5], right_answer, answer,0, model)
        r2 = find_select_choice(imgfinal, anchors[7], right_answer, answer,10, model)
        r3 = find_select_choice(imgfinal, anchors[9], right_answer, answer,20, model)
        r4 = find_select_choice(imgfinal, anchors[11], right_answer, answer,30, model)
        '''
        cv2.putText(imgfinal, str(r1+r2+r3+r4), (800,100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)

        path = 'static/'+str(uuid.uuid4())+'.jpg'
        cv2.imwrite(path, imgfinal)

        return json.dumps(path)
    else:
        return json.dumps("false")






app = Flask(__name__)
@app.route('/upload', methods  = ['POST'])
def upload():
    img = cv2.imdecode(np.fromstring(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #result = detect_buble(img, model)
    result = detect_buble(img)
    return Response(response=result,status=200,mimetype="application/json")

if __name__ == '__main__':
   #model = CNN_Model('weight.h5').build_model(rt=True)
   #app.debug = False
   #app.run(host="0.0.0.0", port=7000)
   app.run()



















