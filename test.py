PATH = "/Users/supradparashar/Documents/Suprad/Code/Machine Learning/Datasets/Watermark Images/train"

from PIL import Image
import os
from tqdm import tqdm

# Delete file.
os.remove(PATH + "/self-watermark/.DS_Store")
print("Done")
os.remove(PATH + "/no-watermark/.DS_Store")
print("Done")

# for image in tqdm(os.listdir(PATH + "/self-watermark")):
#     try:
#         watermark = Image.open(PATH + "/self-watermark/" + image)
#         original = Image.open(PATH + "/no-watermark/" + image)
#         if watermark.size != original.size or watermark.mode != "RGB" or original.mode != "RGB":
#             print(image)
#             print(watermark.size, original.size)
#             print(watermark.mode, original.mode)
#     except:
#         print(image)