import numpy as np
import math
import random
#CNNs - convolution padding, striding, pooling
"""
A Convolutional Neural Network (CNN) is a type of deep learning algorithm that is particularly well-suited for 
image recognition and processing tasks. It is made up of multiple layers, including convolutional layers, 
pooling layers, and fully connected layers. The architecture of CNNs is inspired by the visual processing 
in the human brain, and they are well-suited for capturing hierarchical patterns and spatial dependencies 
within images.

    --- Convolution layer --> Done
    --- Pooling layer --> Done
    --- Activation layer --> Done
    --- Fully connected --> In process   
"""
def Layer_Convolution(matrix_img, filters= 1,kernel_size = 2, padding=0, strides = 1, channels = 1):
    padded_rows = matrix_img.shape[0] + 2*padding
    padded_cols = matrix_img.shape[1] + 2*padding
    padded_matrix = np.zeros((padded_rows,padded_cols), dtype=matrix_img.dtype)
    padded_matrix[padding:padding+matrix_img.shape[0], padding:padding+matrix_img.shape[1]] = matrix_img
    kernel_list = [np.zeros((kernel_size, kernel_size)) for _ in range(filters)]
    fan_in = channels * kernel_size**2
    fan_out = filters * kernel_size**2
    limit = math.sqrt(3 / (fan_in + fan_out))
    for elem in kernel_list:
        for i in range(0,len(elem[0]),strides):
            for j in range(0,len(elem[0]),strides):
                elem[i][j] = random.uniform(-limit, limit)
    #print(kernel_list)
    map_list = []
    for elem in kernel_list:
        map = np.zeros((int((len(padded_matrix)-kernel_size)/strides)+1,int((len(padded_matrix)-kernel_size)/strides)+1))
        for i in range(len(padded_matrix)-kernel_size+1):
            for j in range(len(padded_matrix)-kernel_size+1):    
                map[i][j] = np.sum(np.multiply(padded_matrix[i:i+kernel_size, j:j+kernel_size],elem))
     #           print(len(map))
        map_list.append(map)
    return map_list
matrix = np.arange(1, 37).reshape(6, 6)
#print(Layer_Convolution(matrix))
c1 = Layer_Convolution(matrix)
print("conv result:", c1)
def activation(matrix, mode = 'relu', leaky_alpha = 0.01):
    if mode == 'relu':
        for elem in matrix[0]:
            #elem = elem.tolist()
            for i in range(len(elem)):
                #print(elem[i])
                if elem[i] < 0.0:
                    elem[i] = 0
        return matrix
    elif mode == 'sigmoid':
        for elem in matrix[0]:
           # print(elem)
            for el in elem:
                el = 1/(1+np.exp(-el))
        return matrix    
    elif mode == 'tanh':
        for elem in matrix[0]:
            for el in elem:
                el = (np.exp(el) - np.exp(-el))/(np.exp(el)+np.exp(-el))
        return matrix
    elif mode == 'leaky relu':
        for elem in matrix[0]:
            for el in elem:
                if el<0:
                    el = leaky_alpha*el
        return matrix
    elif mode == 'softmax':
        pass 
#    return matrix
a1 = activation(matrix=c1)
print("act result: ", a1)
def Pooling(matrix, mean = False, max = True, size = 2, stride = 1):
    matrix = matrix[0]
    #print(matrix.shape)
    if mean == max:
        print('Select either mean or max pooling')
        return
    if mean:
        pooled_matrix = np.zeros((int((len(matrix)-size)/stride)+1,int((len(matrix)-size)/stride)+1))
        print(pooled_matrix)
        for i in range(0,len(matrix)-size+1,stride):
            for j in range(0,len(matrix)-size+1,stride):
                pooled_matrix[i][j] = np.mean(np.array([elem for elem in matrix[i:i+size, j:j+size]]))
        return pooled_matrix
    if max:
        pooled_matrix = np.zeros((int((len(matrix)-size)/stride)+1,int((len(matrix)-size)/stride)+1))
        #print(pooled_matrix.shape)
        for i in range(0,int((len(matrix)-size)/stride)+1,stride):
            for j in range(0,int((len(matrix)-size)/stride)+1,stride):
         #       print(i,j)
                pooled_matrix[i][j] = np.max(np.array([elem for elem in matrix[i:i+size, j:j+size]]))
        return pooled_matrix
p1 = Pooling(a1) 
print("Pooling:", p1)
import os
image_dir = './linear_classification/testing'
image_name = random.choice(os.listdir(image_dir))
img_path = os.path.join(image_dir,image_name)
import cv2

img = cv2.imread(img_path)
blue,green,red = cv2.split(img)
blue = blue.astype(np.float32)
green = green.astype(np.float32)
red = red.astype(np.float32)

blue/=255.0
green/=255.0
red/=255.0
#print(blue, green, red)

c1 = Layer_Convolution(blue)
print(c1)
a1 = activation(matrix=c1)
print(a1)
p1 = Pooling(a1) 

c2 = Layer_Convolution(green)
a2 = activation(matrix=c2)
p2 = Pooling(a2) 

c3 = Layer_Convolution(red)
a3 = activation(matrix=c3)
p3 = Pooling(a3)

"""processed_image = cv2.merge((p1.astype(np.uint8), p2.astype(np.uint8), p3.astype(np.uint8)))
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

#print(p1,p2,p3) 
def CNN_one_iter(img_path):
    img = cv2.imread(img_path)
    blue,green,red = cv2.split(img)
    blue = blue.astype(np.float32)
    green = green.astype(np.float32)
    red = red.astype(np.float32)

    blue/=255.0
    green/=255.0
    red/=255.0
    #print(blue, green, red)

    c1 = Layer_Convolution(blue)
    #print(c1)
    a1 = activation(matrix=c1)
    #print(a1)
    p1 = Pooling(a1)
    p1_viz = p1* 255.0 

    c2 = Layer_Convolution(green)
    a2 = activation(matrix=c2)
    p2 = Pooling(a2) 
    p2_viz = p2* 255.0
    c3 = Layer_Convolution(red)
    a3 = activation(matrix=c3)
    p3 = Pooling(a3)
    p3_viz = p3*255.0
    
    #Visualizing 
    processed_image = cv2.merge((p1_viz.astype(np.uint8), p2_viz.astype(np.uint8), p3_viz.astype(np.uint8)))
    cv2.imshow('Processed Image', processed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
CNN_one_iter(img_path)



