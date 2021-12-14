import argparse
import numpy as np
from PIL import Image

from align import align


# creating command line parameters parser
parser = argparse.ArgumentParser(description='Image channels alignment')
parser.add_argument('command', help='align')
parser.add_argument('path_in', help='input image with 3 vertically stacked channels')
parser.add_argument('path_out', help='aligned image')

args = parser.parse_args()

img_in = np.array(Image.open(args.path_in).convert("L"))

# executing command
if args.command == "align":
    result, _, _ = align(img_in, (0, 0))
    Image.fromarray(result).save(args.path_out)
else:
    raise ValueError("No such command")
