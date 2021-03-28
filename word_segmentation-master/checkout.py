"""
For Testing purposes
    Take image from user, crop the background and transform perspective
    from the perspective detect the word and return the array of word's
    bounding boxes
"""

import cv2
from PIL import Image

import page
import words

# User input page image
image = cv2.cvtColor(cv2.imread("testrocheta.jpeg"), cv2.COLOR_BGR2RGB)

# Crop image and get bounding boxes
crop = page.detection(image)
boxes = words.detection(crop)
lines = words.sort_words(boxes)

# Saving the bounded words from the page image in sorted way
i = 0
print("esraa before loop")
for line in lines:
    print("Esraa fe el loop el kbeera")
    text = crop.copy()
    for (x1, y1, x2, y2) in line:
        print("Esraa fe el loop el so3'ayara")
        # roi = text[y1:y2, x1:x2]
        save = Image.fromarray(text[y1:y2, x1:x2])
        # print(i)
        save.save("segmented/segment" + str(i) + ".jpeg")
        i += 1
