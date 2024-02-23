import os
import glob

from PIL import Image

def get2dImageProps(image_url):
    try:
        image = Image.open(image_url)
    except FileNotFoundError as e:
        print(f"Image not found, Dir : {image_url}\nException : {e}")

    if image:
        height, width = image.size
        resolution = image.info.get('resolution')
        return (width, height, resolution)
    else:
        return None


def calcAvrImgSize(dataset_url):
    if not (os.path.exists(dataset_url) & os.path.isdir(dataset_url)):
        raise NotADirectoryError(f"{dataset_url} is not a valid directory.")

    image_data_dirs = glob.glob(dataset_url + '/**/*.png', recursive=True)
    sum_of_heights = 0.0
    sum_of_widths = 0.0

    for image_dir in image_data_dirs:
        w, h, r = get2dImageProps(image_dir)
        sum_of_widths += w
        sum_of_heights += h

    avr_width = round(sum_of_widths / len(image_data_dirs))
    avr_height = round(sum_of_heights / len(image_data_dirs))

    return avr_width, avr_height

