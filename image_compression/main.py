import argparse
from PIL import Image

from image_compression import compression_pipeline


# creating command line parameters parser
parser = argparse.ArgumentParser(description='Bilinear and improved linear interpolation')
parser.add_argument('command', help='bilinear, improved or psnr command')
parser.add_argument('path_in', help='gt image path')
parser.add_argument('path_out', help='interpolated image path')

args = parser.parse_args()
