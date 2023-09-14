import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy
import numpy as np
from PIL import Image, ImageFont, ImageDraw


font_path = plt.matplotlib.get_data_path()+'/fonts/ttf/'

def string_to_points(s, font_size=20, normalize_size=False)->np.ndarray:
    font = ImageFont.truetype(font_path+'DejaVuSans.ttf', font_size)
    (left, top, right, bottom) = font.getbbox(s)
    w, h = right-left, bottom-top
    im = Image.new('L', (w, h))
    draw  = ImageDraw.Draw(im)
    draw.text((-left, -top), s, fill=255, font=font)
    im = np.uint8(im) # type: ignore
    y, x = np.float32(im.nonzero()) # type: ignore
    pos = np.column_stack([x, y])
    if len(pos) > 0:
        pos -= (w/2, h/2)
        pos[:,1] *= -1
    if normalize_size:
        pos /= font_size
    return pos


if __name__ == '__main__':
	s = "A"
	pts = string_to_points(s)