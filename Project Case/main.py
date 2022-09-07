#SALMA LAILATUL ALIFAH
#2101713235
#BA05 - COMPUTER VISION (QUIZ)


import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

#var global
maxMatch = 0
folderFinal = "" #untuk define jadi str

img = cv.imread('object.jpg') #Read image object
img_gray =  cv.cvtColor(img, cv.COLOR_BGR2GRAY) #convert ke gray

#gaussian bluring
gaussian = cv.GaussianBlur(img_gray, (7,7), 150)

#listdir
for folder in os.listdir("Data"): #menggunakan os.listdir untuk memproses gambar difolder data
    print(folder)
    
    imgScene = cv.imread("Data/" + folder) #untuk membaca gambar yang ada difolder data
    imgScene_gray =  cv.cvtColor(imgScene, cv.COLOR_BGR2GRAY) #convert ke gray

    #gaussian bluring
    gaussian1 = cv.GaussianBlur(imgScene_gray, (3,3), 150)

    #Melakukan matching terhadap gambar dalam folder
    surf_obj = cv.xfeatures2d.SURF_create()

    key_obj, des_obj = surf_obj.detectAndCompute(gaussian, None)
    key_scene_obj, des_scene_obj = surf_obj.detectAndCompute(gaussian1, None)

    index_param = dict(algorithm = 0) #KDTREE = 0
    search_param = dict(checks = 50)

    flann_obj = cv.FlannBasedMatcher(index_param, search_param)
    matches = flann_obj.knnMatch(des_obj, des_scene_obj, 2)

    matchesMask = []

    for i in range(len(matches)) :
            matchesMask.append([0,0])

    totalMatch = 0 #define total match
    
    for index, (first_best, second_best) in enumerate(matches) :
        if first_best.distance < 0.7 * second_best.distance :
            matchesMask[index] = [1, 0]
            totalMatch += 1

    if maxMatch < totalMatch :
        folderFinal = folder
        maxMatch = totalMatch

#Hasil maxMatch dan untuk menampilkan hasilnya
        
imgScene = cv.imread("Data/" + folderFinal) #untuk membaca gambar yang ada difolder data
imgScene_gray =  cv.cvtColor(imgScene, cv.COLOR_BGR2GRAY) #convert ke gray

#gaussian bluring
gaussian1 = cv.GaussianBlur(imgScene_gray, (3,3), 150)

#Melakukan matching terhadap gambar dalam folder
surf_obj = cv.xfeatures2d.SURF_create()

key_obj, des_obj = surf_obj.detectAndCompute(gaussian, None)
key_scene_obj, des_scene_obj = surf_obj.detectAndCompute(gaussian1, None)

index_param = dict(algorithm = 0) #KDTREE = 0
search_param = dict(checks = 50)

flann_obj = cv.FlannBasedMatcher(index_param, search_param)
matches = flann_obj.knnMatch(des_obj, des_scene_obj, 2)

matchesMask = []

for i in range(len(matches)) :
    matchesMask.append([0,0])

totalMatch = 0
    
for index, (first_best, second_best) in enumerate(matches) :
    if first_best.distance < 0.7 * second_best.distance :
        matchesMask[index] = [1, 0]
        totalMatch += 1

result_img = cv.drawMatchesKnn(
            gaussian,
            key_obj,
            gaussian1,
            key_scene_obj,
            matches,
            None,
            matchColor = [0, 255, 0],
            singlePointColor = [0, 0 , 255],
            matchesMask = matchesMask
            )


plt.imshow(result_img, "gray")
plt.title("Sister's Requested Fruit")
plt.show()

cv.waitKey(0)
cv.destroyAllWindows()
