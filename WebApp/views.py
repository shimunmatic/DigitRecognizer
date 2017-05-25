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
        data = uri.split(',')[1]
        binary_data = a2b_base64(data)
        fd = open('image_buff.png', 'wb')
        fd.write(binary_data)
        fd.close()

        return cv2.imread('image_buff.png')

def add_border_to_image(im):
    WHITE = [255,255,255]

    bordersize = 100
    border = cv2.copyMakeBorder(im, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                borderType=cv2.BORDER_CONSTANT, value=WHITE)
    return border


def cnn_predict(model, im):
    # model = keras.models.load_model("/home/shimun/PycharmProjects/DigitRecognizer/WebApp/mnist_cnn_model.h5")

    im = cv2.resize(im,(28,28))
    im=im/255
    predictedImage = np.resize(im,(1,28,28,1))
    prediction = model.predict_classes(predictedImage)
    return prediction

def predict(data):
    # Read the input image
    with tf.Session() as sess:
        model = keras.models.load_model("/home/shimun/PycharmProjects/DigitRecognizer/WebApp/mnist_cnn_model.h5")
        import datetime
        time = datetime.datetime.now()
        im = data_uri_to_cv2_img(data)
        print(datetime.datetime.now() - time)
        im = add_border_to_image(im)
        # im = cv2.resize(im,(500,500))
        # Convert to grayscale and apply Gaussian filtering

        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

        # Threshold the image
        ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the image
        _,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get rectangles contains each contour
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        cnt=0
        print(len(rects))
        for rect in rects:
            cnt+=1
            # Draw the rectangles
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            leng = int(rect[3]*1.4)
            pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
            pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
            roi = im_th[pt1:pt1 + leng, pt2:pt2 + leng]    # Make the rectangular region around the digit
            print(roi.shape)
            x,y= roi.shape
            if (x >= 10 and y >= 10):
                roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
                roi = cv2.dilate(roi, (3, 3))
                time = datetime.datetime.now()
                nbr = cnn_predict(model, roi)
                print(datetime.datetime.now() - time)
                cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (255, 144, 30), 3)

        time = datetime.datetime.now()
        cv2.imwrite("hello.png",im)
        with open("hello.png", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        print(datetime.datetime.now() - time)
        return "data:image/png;base64,"+encoded_string.decode('utf-8')





def hello_world(request):

    return render(request, 'WebApp/home.html')



@csrf_exempt
def classify(request):

    data = request.POST.get('data', '')
    import datetime
    time = datetime.datetime.now()
    classifiedData = predict(data)
    print(classifiedData)

    print(datetime.datetime.now() - time)
    return HttpResponse(classifiedData)


def about(request):
    return render(request, 'WebApp/about.html')
