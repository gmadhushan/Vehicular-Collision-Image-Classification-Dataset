"""This program can be used for running inference on the complete test data.
Confusion Matrix, an image classification evaluation metric is computed."""

#Import all the necessary libraries
import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras.models import load_model, Model  # TensorFlow is required for Keras to work
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

#%%Load the keras model from the stored directory

model = load_model("/home/Vehicular_Collision_Image_Classification/Model/keras_model.h5", compile=False)

#%%Run inference on the test for the above loaded keras model

test_data = []
y_pred = []

labels = ['No_Collision', 'Collision', 'Collided'] 
size = (224,224)
#Give the path of the test data
test_path = "/home/Vehicular_Collision_Image_Classification/Test_data/"
for label in labels: 
    path = os.path.join(test_path,label)
    class_num = labels.index(label)
    for imgname in os.listdir(path):
        try:
            image_path = os.path.join(path,imgname)
            image = cv2.imread(image_path)
            image_resized = cv2.resize(image,size) 
            image_array = np.asarray(image_resized)
            normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            data[0] = normalized_image_array
            
            test_data.append([image_resized, class_num])

            prediction = model.predict(data)
            index = np.argmax(prediction)
            #class_name = class_names[index]
            label_name = labels[index]
            confidence_score = prediction[0][index]
            
            y_pred.append(index)
        
        except Exception as e:
            print(e)

y_true = []
for i in range(len(test_data)):
    y_true.append(test_data[i][1])
    
#%%Compute the performance metrics  - Confusion Matrix and Classification Report
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix = cm,
        display_labels=labels)
    
disp.plot()

print("Confusion Matrix")
print(disp.confusion_matrix)

plt.show()

print('\nClassification Report')
#report = classification_report(y_true, y_pred)
report = classification_report(y_true, y_pred,target_names = ['No Collision (Class 0)','Collision (Class 1)', 'Collided (Class 2)'])
print(report)

#%%Create a text file to log the results of inference on test data
logfile = open("/home/Vehicular_Collision_Image_Classification/log_inference.txt","a+")
logfile.write("\nConfusion Matrix\n")
logfile.write(str(cm))
logfile.write("\nClassification Report\n")
logfile.write(str(report))
#%%



