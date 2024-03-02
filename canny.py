import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import cv2

# Converting imaged to a grey scaled image (since only intensity of colour is important)
def to_grey(img: np.ndarray):
  [r_coef, g_coef, b_coef] = [0.3, 0.59, 0.11]

  r, g, b = img[..., 0], img[..., 1], img[..., 2]
  grey_img =  r_coef * r + g_coef * g + b_coef * b
  
  return grey_img

def generate_gaussian_filter(sigma: float, filterSize: int):
 gaussianFilter = np.zeros((filterSize,filterSize))
 center = filterSize//2 #used chatGPT to fix issue where gaussian wasn't symmetric
 #Create the actual filter
 for x in range(filterSize):
  for y in range(filterSize):
   gaussianFilter[x,y] = (1/(2.0 * np.pi * sigma**2.0))*np.exp(-((x-center)**2.0+(y-center)**2.0)/(2.0*sigma**2.0))
 return gaussianFilter
    
def convolution(img: np.ndarray, kernel: np.ndarray):
 convoluted = np.zeros(img.shape)
 kernelSize = kernel.shape[0]
 halfKernel = kernelSize//2
 [width, height] = img.shape

 # First need to pad image with duplicates of the edge pixels so we can deal with the edge case
 img = np.pad(img, ((halfKernel, halfKernel),(halfKernel, halfKernel)), mode="edge")

 # Loop through the image, taking subportions of the image of size kernel
 for x in range(width):
     for y in range(height):
         img_portion = img[x:x+kernelSize, y:y+kernelSize]
         convoluted[x,y] = np.sum(img_portion*kernel)
 return convoluted
    
def sobel_edge_detection(blurredImg: np.ndarray):
 matrix_x = np.array(
     [[-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1]], np.float32
 )
 matrix_y = np.array(
         [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]], np.float32
     )
 gx = convolution(blurredImg, matrix_x)
 gy = convolution(blurredImg, matrix_y)
 g = np.sqrt(gx**2.0 + gy**2.0)
 theta = np.arctan2(gy, gx)
 return g, theta, gx, gy
    
def non_max_suppression(g: np.ndarray, theta: np.ndarray):
 m, n = g.shape
 angle = np.degrees(theta)
 
 for x in range(1, m - 1):
  for y in range(1, n - 1):
   # Ensure angle is non-negative
   if angle[x, y] < 0:
    angle[x, y] += 180
   
   # Determine neighboring pixels based on gradient direction
   if 0 <= angle[x, y] < 22.5:
    neigbourPixel1, neigbourPixel2 = g[x, y-1], g[x, y+1]
   elif 22.5 <= angle[x, y] < 67.5:
    neigbourPixel1, neigbourPixel2 = g[x-1, y+1], g[x+1, y-1]
   elif 67.5 <= angle[x, y] < 112.5:
    neigbourPixel1, neigbourPixel2 = g[x-1, y], g[x+1, y]
   elif 67.5 <= angle[x, y] < 112.5:
    neigbourPixel1, neigbourPixel2 = g[x+1, y+1], g[x-1, y-1]
   else:
    neigbourPixel1, neigbourPixel2 = g[x, y-1], g[x, y+1]

   # Pixels that are less than any of their neigbours are set to 0
   if g[x, y] < neigbourPixel1 or g[x, y] < neigbourPixel2:
    g[x, y] = 0 
 return g

# #identify strong and weak points    
def double_threshold(img: np.ndarray, highThresh: float, lowThresh: float):
 [n,m] = img.shape

 marking = np.zeros(img.shape)
 for x in range(n):
  for y in range(m):
   if img[x,y] > highThresh:
    # mark as highThresh
    marking[x,y] = 255
   elif img[x,y] < lowThresh:
    #mark as low
    marking[x,y] = 0
   else:
    #mark as weak
    marking[x,y] = 25
 return marking

def hysteresis(img: np.ndarray):
 [n,m] = img.shape

 #check if neibourging pixels for the weak edges are strong or not
 for x in range(n-1):
  for y in range(m-1):
   if img[x,y] == 25:
    #check if neibouring pixels are strong
    if ((img[x+1,y] == 255) or (img[x-1,y] == 255) or (img[x+1,y+1] == 255) or (img[x+1,y-1] == 255) or
    (img[x-1,y+1] == 255) or (img[x-1,y-1] == 255) or (img[x,y+1] == 255) or (img[x,y-1] == 255)):
     img[x,y] = 255
    else:
     img[x,y] = 0
 return img

# Loading Image
script_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(script_dir, './images/flower.jpg')
img = np.array(Image.open(img_path))

sigma = 1.5
ksize = 3
tl = 15
th = 30

#Step 1 Noise Reduction
grey = to_grey(img)
kernel = generate_gaussian_filter(sigma, ksize)
blurred = convolution(grey, kernel) #Generate a blurred image

#Step 2 Gradient Calc
[g, theta, gx, gy] = sobel_edge_detection(blurred)

#Step 3 Non Max Calc
suppressedImg = non_max_suppression(g, theta)

# #Step 4 and 5
marks = double_threshold(suppressedImg, th, tl)
final = hysteresis(marks)

plt.subplot(2,2,1)
plt.imshow(img)
plt.title("Original Image (A)")
plt.subplot(2,2,2)
plt.imshow(blurred, cmap='gray')
plt.title("Blur Image (B)")
plt.subplot(2,2,3)
plt.imshow(g, cmap='gray')
plt.title("Magnitude of Gradient Image (C)")
plt.subplot(2,2,4)
plt.imshow(suppressedImg, cmap='gray')
plt.title("Non Max Supressed Image (D)")
# plt.subplot(3,2,5)
# plt.imshow(final, cmap='gray')
# plt.title("Canny Final Image (E)")
plt.show()

# Comparing with cv2
img = cv2.imread(img_path)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (ksize, ksize),sigma)
# # Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=3) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3) # Combined X and Y Sobel Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=tl, threshold2=th) # Canny Edge Detection

# # img = np.array(Image.open(img_path))
# # plt.subplot(1,3,1)
# # plt.imshow(img)
# # plt.title("Original Image")

# plt.subplot(1,2,1)
# plt.imshow(g, cmap='gray')
# plt.title("Implemented Edge Detection - Blurred Image(A)")

# plt.subplot(1,2,2)
# s = np.sqrt(sobelx**2.0 + sobely**2.0)
# plt.imshow(s, cmap='gray')
# plt.title("Edge Detection with OpenCV - Blurred Image (B)")
# plt.show()


plt.subplot(1,2,1)
plt.imshow(final, cmap='gray')
plt.title("Implemented Edge Detection (A)")

plt.subplot(1,2,2)
plt.imshow(edges, cmap='gray')
plt.title("Edge Detection with OpenCV (B)")
plt.show()

# script_dir = os.path.dirname(os.path.abspath(__file__))
# img_path = os.path.join(script_dir, './images/licensePlate.png')
# l = np.array(Image.open(img_path))
# img_path = os.path.join(script_dir, './images/city.jpg')
# city = np.array(Image.open(img_path))
# img_path = os.path.join(script_dir, './images/tumor.jpg')
# brain = np.array(Image.open(img_path))
# img_path = os.path.join(script_dir, './images/panda.png')
# panda = np.array(Image.open(img_path))

# plt.subplot(2,2,1)
# plt.imshow(l, cmap='gray')
# plt.title("License Plate Image (A)")

# plt.subplot(2,2,3)
# plt.imshow(city, cmap='gray')
# plt.title("New York City Image (C)")

# plt.subplot(2,2,2)
# plt.imshow(panda, cmap='gray')
# plt.title("Grainy Panda Image (B)")
# plt.show()

