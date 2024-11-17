DATASET_PATH = "/Users/supradparashar/Documents/Suprad/Code/Machine Learning/Datasets/Watermark Images/valid/no-watermark"
OUTPUT_PATH = "/Users/supradparashar/Documents/Suprad/Code/Machine Learning/Datasets/Watermark Images/valid/self-watermark"

from PIL import Image, ImageDraw, ImageFont
import emoji
from wonderwords import RandomWord
import random
import os
from p_tqdm import p_map

emojis = list(map(lambda x: x.encode("unicode_escape"), emoji.unicode_codes.EMOJI_DATA.keys()))
get = RandomWord()
random_text = lambda: get.word(word_max_length=6) + " " + get.word(word_max_length=6)
random_pos = lambda image: (random.randint(50, image.size[0] - 50), random.randint(50, image.size[1] - 50))

def watermark_text(file_name):
    photo = Image.open(DATASET_PATH + "/" + file_name)
    drawing = ImageDraw.Draw(photo)
    font = ImageFont.truetype("Arial Unicode.ttf", 36)
    for _ in range(2):
        drawing.text(random_pos(photo), random_text(), fill=(0, 0, 0), font=font)
    photo.save(OUTPUT_PATH + "/" + file_name)

if __name__ == "__main__":
    p_map(watermark_text, os.listdir(DATASET_PATH), num_cpus=4)