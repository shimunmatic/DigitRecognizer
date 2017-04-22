import base64
from binascii import a2b_base64

import cv2
import keras
import numpy as np
import tensorflow as tf
from  django.http import HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt


#load pretrained model

def data_uri_to_cv2_img(uri):
        encoded_data = uri.split(',')[1]
        # img = base64.b64decode(uri)
        # nparr = np.fromstring(img, np.uint8)
        # img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # cv2.IMREAD_COLOR in OpenCV 3.1
        data = encoded_data
        binary_data = a2b_base64(data)
        fd = open('imagelol.png', 'wb')
        fd.write(binary_data)
        fd.close()

        return cv2.imread('imagelol.png')

def add_border_to_image(im):
    WHITE = [255,255,255]

    bordersize = 100
    border = cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                borderType=cv2.BORDER_CONSTANT, value=WHITE)
    return border



def cnn_predict(im):
    model = keras.models.load_model("/home/shimun/PycharmProjects/untitled/WebApp/mnist_cnn_model.h5")

    im = cv2.resize(im,(28,28))
    im=im/255
    predictedImage = np.resize(im,(1,28,28,1))
    prediction = model.predict_classes(predictedImage)
    return prediction

def predict(data):
    # Read the input image
    with tf.Session() as sess:
        model = keras.models.load_model("/home/shimun/PycharmProjects/untitled/WebApp/mnist_cnn_model.h5")

        im = data_uri_to_cv2_img(data)

        im = add_border_to_image(im)
        im = cv2.resize(im,(500,500))
        # Convert to grayscale and apply Gaussian filtering

        # cv2.imwrite("base.png", im)

        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        # For each rectangular region, calculate HOG features and predict
        # the digit using Linear SVM.
        cnt=0
        print(len(rects))
        for rect in rects:
            cnt+=1
            print(rect)
            # Draw the rectangles

            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            leng = int(rect[3]*1.4)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]    # Make the rectangular region around the digit

            print(roi.shape)
            x,y= roi.shape
            #if( x>= 28 & y>= 28):
            roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            roi = cv2.dilate(roi, (3, 3))

            #predict rectangle
            nbr = cnn_predict(roi)
            cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 144, 30), 3)

        cv2.imwrite("hello.png",im)
        with open("hello.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        return "data:image/png;base64,"+(encoded_string).decode('utf-8')





def hello_world(request):

    return render(request, 'WebApp/home.html')

def index(request):
    return HttpResponse("You are in the INDEX")

@csrf_exempt
def classify(request):
    data = request.POST.get('data', '')
    classifiedData = predict(data)
    print(classifiedData)


    return HttpResponse(classifiedData)

