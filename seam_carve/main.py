import argparse
import numpy as np
from PIL import Image

from seam_carve import seam_carve


def convert_img_to_mask(img_src):
    return ((img_src[:, :, 0] != 0) * -1 + (img_src[:, :, 1] != 0)).astype('int8')


# creating command line parameters parser
parser = argparse.ArgumentParser(description='Context sensitive image scaling')
parser.add_argument('command', help='shrink/expand')
parser.add_argument('direction', help='horizontal/vertical')
parser.add_argument('n_iter', help='number of algorithm\'s iterations')
parser.add_argument('path_in', help='input image path')
parser.add_argument('path_mask', help='mask path')
parser.add_argument('path_out', help='output image path')

args = parser.parse_args()

img = np.array(Image.open(args.path_in).convert("RGB"))

mask = np.array(Image.open(args.path_mask).convert("RGB"))
mask = convert_img_to_mask(mask)

if args.command not in ["shrink", "expand"]:
    raise ValueError(f"Wrong command: {args.command}")

if args.direction not in ["horizontal", "vertical"]:
    raise ValueError(f"Wrong direction: {args.direction}")

seam_args = [args.command, args.direction]

for i in range(int(args.n_iter)):
    img, mask, _ = seam_carve(img, seam_args, mask)

Image.fromarray(img).save(args.path_out)
