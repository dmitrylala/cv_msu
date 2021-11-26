import argparse
import numpy as np
from PIL import Image
from skimage.color import rgb2gray

from bayer import bilinear_interpolation, improved_interpolation, compute_psnr


# creating command line parameters parser
parser = argparse.ArgumentParser(description='Bilinear and improved linear interpolation')
parser.add_argument('command', help='bilinear, improved or psnr command')
parser.add_argument('path_in', help='gt image path')
parser.add_argument('path_out', help='interpolated image path')

args = parser.parse_args()

img_gt = np.array(Image.open(args.path_in).convert("RGB"))

# executing commands
if args.command == "bilinear":
    result = bilinear_interpolation(img_gt)
    Image.fromarray(result).save(args.path_out)
elif args.command == "improved":
    result = improved_interpolation(rgb2gray(img_gt))
    Image.fromarray(result).save(args.path_out)
elif args.command == "psnr":
    img_inter = np.array(Image.open(args.path_out).convert("RGB"))
    print(f"PSNR: {compute_psnr(img_inter, img_gt)}")
else:
    raise ValueError("No such command")
