import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
# Load the image using OpenCV
PATH_CATS = os.path.join('data','cats')
PATH_DOGS = os.path.join('data','dogs')
image = cv2.imread(os.path.join(PATH_CATS,'cat1.jpg'))
HEIGHT = 250
WIDTH = 250


from skimage import feature
from skimage.color import rgb2gray
"""
Manual KNN
    -- for this task of binary classification of dog and cat images, I chose 3 features: texture, pixel intensity histogram, pixel values
    -- based on the distance between those features, weighted KNN was manually implemented
    -- Has decent but slow (expectedly) performance on augmented data
    -- Data augmentation: rotation, mirroring, change in color
    
"""

# Example function to compute GLCM texture features for a grayscale image
def compute_glcm_features(image):
    # Convert image to grayscale
    image = cv2.resize(image, (HEIGHT,WIDTH))
    gray_image = rgb2gray(image)

    # Convert the grayscale image to unsigned integer type
    gray_image_uint = gray_image.astype(np.uint8)

    # Compute GLCM with adjusted parameters
    glcm = feature.graycomatrix(gray_image_uint, distances=[1,1.5,2,2.5,3,4,5,6], angles=[0,np.pi/4,np.pi/2, 3*np.pi/4,np.pi], levels=32, symmetric=True, normed=False)

    # Compute GLCM properties (e.g., contrast, energy, homogeneity)
    contrast = feature.graycoprops(glcm, 'contrast')
    energy = feature.graycoprops(glcm, 'energy')
    homogeneity = feature.graycoprops(glcm, 'homogeneity')

    # Concatenate texture features into a single feature vector
    texture_features = np.concatenate([contrast.flatten(), energy.flatten(), homogeneity.flatten()])

    return texture_features

def dist_grayscale(img, point):
    score_new = compute_glcm_features(img)
    score_point = compute_glcm_features(point[2])
    return 100*np.linalg.norm(np.array(score_new)- np.array(score_point)), point




def hist_distance(image1, point2):
    hist1 = cv2.calcHist([image1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.calcHist([point2[2]], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return np.linalg.norm(hist1 - hist2), point2


def distance_pixel(point1,point2):
    distances = []
    for vector1, vector2 in zip(point1,point2[0]):
        distance = np.linalg.norm(np.array(vector1)- np.array(vector2))
        distances.append(distance)
    final_dist = np.mean(distances)
    return final_dist,point2[1]

def preprocess(img):
    resized_img_normal = cv2.resize(img, (HEIGHT,WIDTH))
    resized_img = resized_img_normal.astype(float)/255.0
    pixel_values = resized_img.reshape((-1, 3))
    return resized_img_normal, pixel_values, resized_img
img,img_pixes, img = preprocess(image)
#print(os.path.join(PATH_DOGS,os.listdir(PATH_DOGS)[0]))
pic1 = cv2.imread(os.path.join(PATH_CATS,random.choice(os.listdir(PATH_CATS))))
#print(compute_glcm_features(pic1))
#print(preprocess(pic1))
pic2 = cv2.imread(os.path.join(PATH_CATS,random.choice(os.listdir(PATH_CATS))))
print(dist_grayscale(pic1,[0,0,pic2])[0])

#print(preprocess(pic1))
#print(hist_distance(pic1,pic2))
import sys

#sys.exit()

#print(preprocess(image))
def feature_extraction(image):
# Extract pixel values
    image = preprocess(image)
    #print(image)
    pixel_values = image.reshape((-1, 3))  # Reshape to a 2D array of pixels
    print(pixel_values)
    # Plot pixel intensity distribution (histogram)
    plt.hist(pixel_values, bins=256, color=['blue', 'green', 'red'])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Pixel Intensity Distribution')
    plt.show()

    # Extract color histograms
    color_channels = ('b', 'g', 'r')
    histograms = []

    for i, channel in enumerate(color_channels):
        # Calculate histogram for each channel
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        histograms.append(histogram)

    # Plot color histograms
    plt.figure()
    plt.title('Color Histograms')
    for i, histogram in enumerate(histograms):
        plt.plot(histogram, color=color_channels[i])
        plt.xlim([0, 1])
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def get_all_img():
    
    data = []
    for img in os.listdir(PATH_CATS):
        image = cv2.imread(os.path.join(PATH_CATS, img))
        preprocess_tuple = preprocess(image)
        data.append([preprocess_tuple[1], 1, image])
    for img in os.listdir(PATH_DOGS):
        image = cv2.imread(os.path.join(PATH_DOGS, img))
        preprocess_tuple = preprocess(image)
        data.append([preprocess_tuple[1], 0, image])
    return data

data = get_all_img()
random.shuffle(data)
#why do I get different results if I compare to all the data wtf
from collections import Counter

def combine_predictions(k_labels):
    # Flatten the list of k labels
    all_labels = [label for sublist in k_labels for label in sublist]
    
    # Count the occurrences of each label
    label_counts = Counter(all_labels)
    
    # Find the label with the highest frequency
    final_label = max(label_counts, key=label_counts.get)
    
    return final_label

def change_dist(list, label):
    new_dist = []
    for elem in list:
        if elem[1] == label and elem[0]!=0:
            new_dist.append(1/elem[0])
        elif elem[0] == 0:
            new_dist.append(10)
    return new_dist
            
    
def knn(k):
    new_img = random.choice(os.listdir('./testing'))
    print(new_img)
    new_img = cv2.imread(os.path.join('testing', new_img))
    new_img_array = preprocess(new_img)[1]
    dist_pix = []
    dist_hist = []
    dist_gray = []
    print(len(data))
    #print(data)
    for point in data:
        #print(point)
        #print("label: ", point[0][1])
        dist_gray.append(dist_grayscale(new_img,point))
        dist_hist.append(hist_distance(new_img,point))
        dist_pix.append(distance_pixel(new_img_array, point))
    
    
    dist_hist = sorted(dist_hist, key=lambda x:x[0])
    dist_pix = sorted(dist_pix, key=lambda x: x[0])
    dist_gray = sorted(dist_gray, key=lambda x:x[0])
    
    first_k_hist = dist_hist[:k]
    first_k_hist= [(dist[0], dist[1][1]) for dist in first_k_hist]
    hist_0 = change_dist(first_k_hist,0)
        
    hist_1 = change_dist(first_k_hist,1)
    print(sum(hist_0))
    print(sum(hist_1))
    pred_hist = int(sum(hist_0)<sum(hist_1))
    
    first_k_gray = dist_gray[:k]
    first_k_gray= [(dist[0], dist[1][1]) for dist in first_k_gray]
    gray_0 = change_dist(first_k_gray,0)
    gray_1 = change_dist(first_k_gray,1)
    pred_gray = int(sum(gray_0)<sum(gray_1))
    
    print("hist:", first_k_hist)
    #plt.imshow(first_k_hist[0][1][2])
    print("gray:", first_k_gray)
    #plt.show()
    first_k = dist_pix[:k]
    pix_0 = change_dist(first_k,0)
    pix_1 = change_dist(first_k,1)
    pred_pix = int(sum(pix_0)<sum(pix_1))
    
    print("pix:", first_k)
    
    print("Pred: ", sum([pred_hist,pred_gray,pred_hist])>int(len([pred_hist,pred_gray,pred_hist])/2))
    label_list = []
    label_gray = [elem[1] for elem in first_k_gray]
    label_hist = [elem[1] for elem in first_k_hist]
    label_pix = [elem[1] for elem in first_k]
    label_list.append(label_gray)
    label_list.append(label_pix)
    label_list.append(label_hist)
    return combine_predictions(label_list)
    

print(knn(5))
    