"""This program can be used for running inference on individual classes of the test data.
Predicted class along with confidence score will be embedded on the input image and saved to a defined location."""


#Import the necessary libraries
import os
import cv2
from keras.models import load_model, Model  # TensorFlow is required for Keras to work
from PIL import Image, ImageDraw, ImageOps  # Install pillow instead of PIL
import numpy as np
import tensorflow as tf

#Give the path of the folder where you want to save the output images of a particular class
os.chdir("/home/Vehicular_Collision_Image_Classification/Inference/Prediction_Collided/") 
logfile = open("/home/Vehicular_Collision_Image_Classification/log_inf.txt","a+")

np.set_printoptions(suppress=True)

#Load the keras model downloaded from teachable machine
model = load_model("/home/Vehicular_Collision_Image_Classification/Model/keras_model.h5", compile=False)
msg1 = "\n-------------------------------- Model Loaded ---------------------------------"
logfile.write(msg1)

#Read and Store the label names from the label file
class_names = open("/home/Vehicular_Collision_Image_Classification/Model/labels.txt", "r").readlines()

#Provide the path for the test data - classwise folder
test_data_path = "/home/Vehicular_Collision_Image_Classification/Test_data/Collided/"

count = 0           #count variable
correct_pred = 0    #variable to store the no. of correction predictions
incorrect_pred = 0  #variable to store the no. of incorrection predictions
size = (224, 224)   #size of the input image to the model

#For every image in the test data folder do the following:
for imgname in os.listdir(test_data_path):
    image_path = os.path.join(test_data_path,imgname)   #Create path for individual test image
    image = cv2.imread(image_path)                      #read the test image
    if image is not None:       #if the image is not None, resize it and store it as array
        image_resized = cv2.resize(image,size)
        image_array = np.asarray(image_resized)
    
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1   #Normalize the input test image
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array    #extract the image content from the nd array

        prediction = model.predict(data)    #prediction for the input test image by the model
        index = np.argmax(prediction)       #returns the index position of the maximum prediction probability
        class_name = class_names[index]     #fetch the class name corresponding to the index position
        confidence_score = prediction[0][index]     #obtain the confidence score for the prediction made

        """ 
        check the predicted class name. Create a text containing the class name and the confidence score of prediction.
        Overlay the text on the input test image using putText method.
        Green color for No collision (0,255,0)
        Red color for Collision (0,0,255)
        Brown color for Collided (0,0,128) for the color of the text.
        """

        if class_name[2:] == 'No_Collision\n':
            #print("noclsn")
            text = "No Collision {:.4f}".format(confidence_score)
            image_out = cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0,255,0), 2, cv2.LINE_AA)
        elif class_name[2:] == 'Collision\n':
            #print("clsn")
            text = "Collision {:.4f}".format(confidence_score)
            image_out = cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0,0,255), 2, cv2.LINE_AA)
        elif class_name[2:] == 'Collided\n':
            #print("colveh")
            text = "Collided {:.4f}".format(confidence_score)
            image_out = cv2.putText(image, text, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                                (0,0,128), 2, cv2.LINE_AA)
        else:
            pass;
            
            
        #increment the count of correct and incorrect predicions according to the output prediction     
        if text.replace(text[-7:],'') == 'Collided':
            correct_pred +=1
            msg2 = "\nCorrect Prediction: {}".format(imgname)
            logfile.write(msg2)
            print(msg2)
        else:
            incorrect_pred +=1
            msg3 = "\n*************Incorrect Prediction: {}".format(imgname)
            logfile.write(msg3)
            print(msg3)
    
        count+=1    #increment the image count
    
        cv2.imwrite(imgname,image_out) #write the output image with text overlaid on it to the folder

        print("\nPredicted Class:", class_name[2:], end="") #Print predicted class and confidence score for every image
        print("\nConfidence Score:", confidence_score)
        
#print and log the total no.of correct and incorrect predictions and total no. of images tested on the model
        
msg5 = "\n\nNo. of Correct Predictions: {}".format(correct_pred)
logfile.write(msg5)
print(msg5)

msg6 = "\nNo. of Incorrect Predictions: {}".format(incorrect_pred)
logfile.write(msg6)
print(msg6)


msg4 = "\nTotal no.of images: {}\n ------------------------------".format(count)
logfile.write(msg4)
print(msg4)

