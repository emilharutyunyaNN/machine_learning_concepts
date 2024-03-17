import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import os
import random
from skimage import feature
HEIGHT = 250
WIDTH = 250
PATH_CATS = os.path.join('data','cats')
PATH_DOGS = os.path.join('data','dogs')

"""
Manual SVM for the same dog and cat data (manually collected and augmented)

    -- The data augmentation for both KNN and SVM is done here
    -- The weight for the combined feature vector is determined with gradient descent method
    -- Although analytical solution is available

"""


def data_augmentation(dir):
    for image_pth in os.listdir(dir):
        image_pth = os.path.join(dir, image_pth)
        image = cv2.imread(image_pth)
        im_rotate = cv2.rotate(image, cv2.ROTATE_180)
        im_flip = cv2.flip(image,0)
        # Convert image to float32
        img_bright = np.float32(image)
        
        # Apply brightness adjustment
        img_bright += 50
        
        # Apply contrast adjustment
        img_bright = img_bright * (1 + 20 / 127)
        img_bright = np.clip(img_bright, 0, 255)
        
        # Convert back to uint8
        img_bright = np.uint8(img_bright)
        
        noise = np.random.normal(0, 20, image.shape).astype('uint8')
        noisy_image = cv2.add(image, noise)
        
        
        cv2.imwrite(image_pth+'_bright', img_bright)
        cv2.imwrite(image_pth+'_rotate', im_rotate)
        cv2.imwrite(image_pth+'_flip', im_flip)
        cv2.imwrite(image_pth+'_noisy', noisy_image)

#data_augmentation(PATH_CATS)
#data_augmentation(PATH_DOGS)

print(len(os.listdir(PATH_CATS)))
print(len(os.listdir(PATH_DOGS)))
#sys.exit()

def compute_histogram(image):
    
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compute_texture_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = feature.graycomatrix(gray_image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = feature.graycoprops(glcm, 'contrast')
    energy = feature.graycoprops(glcm, 'energy')
    homogeneity = feature.graycoprops(glcm, 'homogeneity')
    texture_features = np.concatenate([contrast.flatten(), energy.flatten(), homogeneity.flatten()])
    return texture_features

def combine_features(image):
    # Reshape pixel values
    image = cv2.resize(image, (HEIGHT,WIDTH))
    pixel_values = image.astype(float)/255.0
    pixel_values = pixel_values.flatten()
    pix_l = len(pixel_values)
    
    # Compute pixel intensity histograms
    hist = compute_histogram(image)
    hist_l = len(hist)
    # Compute grayscale texture features
    texture_features = compute_texture_features(image)
    text_l = len(texture_features)
    # Concatenate all feature vectors
    combined_features = np.concatenate([pixel_values, hist, texture_features])
   # pixel = pixel_values.tolist()+[0]*(len(combine_features)-pix_l)
   # hist = [0]*len(pix_l)+hist.tolist()+[0]*(len(combine_features)-pix_l-hist_l)
    #texture = [0]*(hist_l+pix_l)+texture_features.tolist()
    combined_features /= np.max(combined_features)
    log_combined = np.log2(combined_features+1)

    return combined_features
"""from sklearn.preprocessing import MinMaxScaler
def normalize_features(features):
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()
    
    # Fit the scaler to the features and transform them
    normalized_features = scaler.fit_transform(features.reshape(1, -1))
    
    return normalized_features.flatten()"""

PATH_CATS = os.path.join('data','cats')
PATH_DOGS = os.path.join('data','dogs')
image = cv2.imread(os.path.join(PATH_CATS,'cat1.jpg'))

#print(min(combine_features(image)), max(combine_features(image)))
print("Normal: ", combine_features(image)[1])
print("Log: ", combine_features(image)[1])
EPOCHS = 100
def dataset():
    
    data_cats = []
    data_dogs = []
    train_labels_c = []
    train_labels_d = []
    for img in os.listdir(PATH_CATS):
        image = cv2.imread(os.path.join(PATH_CATS,img))
        data_cats.append([combine_features(image),1])
        train_labels_c.append(1)
    random.shuffle(data_cats)
    train_data_cats = data_cats[:int(len(data_cats)*0.7)]
    train_labels_c = train_labels_c[:int(len(data_cats)*0.7)]
    test_data_cats = data_cats[int(len(data_cats)*0.7):]
    
    for img in os.listdir(PATH_DOGS):
        image = cv2.imread(os.path.join(PATH_DOGS,img))
        data_dogs.append([combine_features(image),-1])
        train_labels_d.append(0)
        
    random.shuffle(data_dogs)
    train_data_dogs = data_dogs[:int(len(data_dogs)*0.7)]
    train_labels_d = train_labels_d[:int(len(data_cats)*0.7)]
    test_data_dogs = data_dogs[int(len(data_dogs)*0.7):]
    
    train_data = train_data_cats+train_data_dogs
    test_data = test_data_cats+test_data_dogs  
    return train_data,test_data, train_labels_c+train_labels_d

train_set, test_set, train_labels = dataset()   
print(len(train_set))
#random.shuffle(train_set)
def sigmoid_numpy(X):
    return 1/(1+np.exp(-X))
#print("TRAIN SET:", train_set)

    
#Linear SVM
"""
def initialize_weights(length):
    # Generate random weights from a normal distribution with mean 0 and standard deviation 0.01
    w = np.random.normal(loc=0, scale=0.01, size=length)
    # Convert ndarray to list
    w = w.tolist()
    return w
def quadratic_kernel(x):
    return np.square(x)
def svm_manual(kernel = None):
    # Initialize weights with zeros
    w = np.zeros(len(train_set[0][0]))
    b = 0
    alpha = 0.2
    
    for epoch in range(EPOCHS):
        print("bias: ", b)
        random.shuffle(train_set)
        for point in train_set:
            x, y = point
            # Compute the dot product
            if kernel:
                x = kernel(x)
            f = np.dot(np.transpose(w), x) + b
            # Compute the loss
           # epsilon = max(0,1-y*f)
            loss = max(0, int(1 - y * f))
            # Compute gradients
            if loss > 0:
                
                grad_w = -y * x 
                grad_b = -y
            else:
#               
                grad_w = np.zeros(len(train_set[0][0]))
                grad_b = 0
            
            if epoch == 8:
                print("some info: ", alpha * grad_b) 
            # Update weights and bias
            w -= alpha * grad_w
            b -= alpha * grad_b
            if epoch == 99:
                print("label: ", y)
                print("f value: ", f)
                print("loss: ", loss)
                print("grad w: ", grad_w)
                print("grad b: ", grad_b,'\n')
        print(f"Epoch {epoch}: ", '\n', "w: ", w, "b: ", b)
    return w, b
w, b = svm_manual()
margin_pos = w, b+ 1/np.linalg.norm(w)
margin_neg = w, b-1/np.linalg.norm(w)
print("Negtive margin: ", margin_neg)
print("Positive margin: ", margin_pos)
print("margin: ", 1/np.linalg.norm(w))
"""
"""
def soft_svm(C=1.0, lambda_reg = 0.001):
    w = np.zeros(len(train_set[0][0]))
    b = 0
    alpha = 0.2
    for _ in range(EPOCHS):
        for point in train_set:
            x,y = point
            f = np.dot(np.transpose(w),x)+b
            
            loss = max(0,1-y*f)
            #total_loss += C*loss + np.dot(np.transpose(w),w)
            
            if loss>0:
                grad_w = -C*y*x
                grad_b = -C*y
                
            else:
                grad_w = np.zeros(len(train_set[0][0]))
                grad_b = 0
            if _ == 99:
                print("label: ", y)
                print("f value: ", f)
                print("loss: ", loss)
                print("grad w: ", grad_w)
                print("grad b: ", grad_b,'\n')
            w -= alpha*grad_w
            b -= alpha*grad_b     
    return w,b 
w,b = soft_svm(C=0.01)
print(w,b)
print(1/np.linalg.norm(w))"""

"""
w,b = svm_manual()
print(w,b)
#testing 
false_pred = 0
true_pred= 0
for point in test_set:
    x,y = point
    f = np.dot(np.transpose(w),x)+b
    print("f value : ", f, '\n', "true label: ", y)
    y_hat = int(f<=0)
    

    if y==y_hat:
        true_pred+=1
    else:
        false_pred+=1
        
w,b = soft_svm()
print(w,b)
#testing 
false_pred = 0
true_pred= 0
for point in test_set:
    x,y = point
    f = np.dot(np.transpose(w),x)+b
    print("f value : ", f, '\n', "true label: ", y)
    y_hat = int(f<=0)
    

    if y==y_hat:
        true_pred+=1
    else:
        false_pred+=1"""
#print("True rate: ", true_pred/(true_pred+false_pred))
#print("False rate: ", false_pred/(true_pred+false_pred))
"""def poly_kernel(x,n):
    return np.power(x,n)
#Kernel SVM
def gaussian_kernel(x,xi,stdev = 1.0):
    return np.exp(-(np.linalg.norm(x-xi)**2)/2*(stdev**2))
print("Gaussian kernel: ")
x1,y1 = train_set[0]
x2,y2 = train_set[1]

print(gaussian_kernel(x1,x2))
EPOCHS = 100
def svm_kernel(kernel = None, degree = 1):
    w = np.zeros(len(train_set[0][0]))
    b = 0
    alpha = 0.2
    for epoch in range(EPOCHS):
        random.shuffle(train_set)
        for point in train_set:
            x,y = point
            print("Before kernel: ", x)
            if kernel is not None:
                x = kernel(x,degree)
                
            print("After kernel: ", x)
            f = np.dot(np.transpose(w),x)+b
            loss = max(0, int(1-y*f))
           
            if loss>0:
                grad_w = -y*x
                grad_b = -y
            else:
                grad_w = np.zeros(len(train_set[0][0]))
                grad_b = 0
             
            w -= alpha*grad_w
            b -= alpha*grad_b
            
            if epoch == 49:
                print("label: ", y)
                print("f value: ", f)
                print("loss: ", loss)
                print("grad w: ", grad_w)
                print("grad b: ", grad_b,'\n')
        print(f"Epoch {epoch}: ", '\n', "w: ", w, "b: ", b)
        
    return w,b

w,b = svm_kernel(poly_kernel, 3)
print(w,b)
"""
#Simple Quadratic Kernel Example

"""
    Here we have a case of lineraly non separable data that becomes linearly separable after applying quadratic kernel to it,
    data is plotted by two features, we can also make it 3D if we want to and find the plane that separated the data after transformation
"""
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
num_samples = 100
# Class 0 (circles)
class_01 = 0.25 * np.random.randn(num_samples, 2) + np.array([2, 2])
class_02 = 0.25 * np.random.randn(num_samples, 2) + np.array([-2, -2])
class_0 = np.concatenate((class_01,class_02), axis = 0)
print(type(class_0))
# Class 1 (crosses)
class_11 = 0.25 * np.random.randn(num_samples, 2) + np.array([1, 1])
class_12 = 0.25 * np.random.randn(num_samples, 2) + np.array([-1, -1])
class_1 = np.concatenate((class_11,class_12), axis = 0)

# Plot the original data
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(class_0[:, 0], class_0[:, 1], marker='o', label='Class 0')
plt.scatter(class_1[:, 0], class_1[:, 1], marker='x', label='Class 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Original Data')
plt.legend()
plt.grid(True)

# Square the second feature of class 1
class_1_squared = np.copy(class_1)
class_1_squared[:, 1] **= 2
class_0_squared = np.copy(class_0)
class_0_squared[:, 1] **= 2

# Plot the modified data
plt.subplot(1, 2, 2)
plt.scatter(class_0_squared[:, 0], class_0_squared[:, 1], marker='o', label='Class 0 Squared')
plt.scatter(class_1_squared[:, 0], class_1_squared[:, 1], marker='x', label='Class 1 Squared')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2^2')
plt.title('Modified Data (Feature Squared)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ircular kernel example
"""
    We see that when we plot the x and y in a ring shape the data is not linearly separable but when we transform them into their
    polar coordinate versions we can now perform gradient descent to get the line with the maximum margin. 
    Another example of kernel functions, we first apply the kernel fucntion which is a matrix multiplication essentially then in the new 
    linearly separable dataset we find the best line.    
    
"""
def generate_points_in_ring(num_points, inner_radius, outer_radius):
    """
    Generate random points uniformly distributed within a ring-shaped region.

    Parameters:
        num_points (int): Number of points to generate.
        inner_radius (float): Inner radius of the ring.
        outer_radius (float): Outer radius of the ring.

    Returns:
        numpy.ndarray: Array of shape (num_points, 2) containing the generated points.
    """
    # Generate random angles between 0 and 2*pi
    angles = np.random.uniform(0, 2*np.pi, num_points)
    
    # Generate random radii between inner_radius and outer_radius
    radii = np.sqrt(np.random.uniform(inner_radius**2, outer_radius**2, num_points))
    
    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    
    # Combine x and y coordinates into a single array
    points = np.column_stack((x, y))
    points_polar = np.column_stack((radii, angles))
    
    return points, points_polar

# Example usage:
num_points = 1000
inner_radius_0 = 2.0
outer_radius_0 = 3.0
inner_radius_1 = 0.0
outer_radius_1 = 1.0
points_0, points_0_polar = generate_points_in_ring(num_points, inner_radius_0, outer_radius_0)

points_1, points_1_polar = generate_points_in_ring(num_points, inner_radius_1, outer_radius_1)
plt.figure(figsize=(6, 6))
plt.scatter(points_0[:, 0], points_0[:, 1], s=10, c='blue', alpha=0.5)
plt.scatter(points_1[:, 0], points_1[:, 1], s=10, c='red', alpha=0.5)
plt.title('Points in Ring-Shaped Region')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

plt.figure(figsize=(6, 6))
plt.scatter(points_0_polar[:, 0], points_0_polar[:, 1], s=10, c='blue', alpha=0.5)
plt.scatter(points_1_polar[:, 0], points_1_polar[:, 1], s=10, c='red', alpha=0.5)
plt.title('Points in Ring-Shaped Region')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
   


