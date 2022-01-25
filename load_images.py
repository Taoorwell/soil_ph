from glob import glob
from skimage import io
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from tqdm import tqdm

figures_path = r'../Figures/**/**/**/'
images = glob(figures_path + '*.png', recursive=True)
print('Overall amounts:', len(images))
print(images[0])
image = io.imread(images[-1])
print('image size:', image.shape)
random = np.random.randint(len(images), size=4)
print('random selected images:', random)

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(9, 3))
print(axs)
for r, ax in zip(random, axs.flat):
    image = io.imread(images[r])[150:900, 200:900]
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    t = cv.threshold(gray, 0, 255, cv.THRESH_OTSU)
    x, y, w, h = cv.boundingRect(t[1])
    image = image[y:y+h, x:x+w]
    image = np.flipud(image)
    image = image[50:-5, 20:-50]
    # ax.imshow(np.rot90(image, 2))
    ax.imshow(image)

for im in tqdm(images):
    print(im)

# plt.imshow(image)
# plt.show()
