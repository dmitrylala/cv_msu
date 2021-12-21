import os
import argparse
import numpy as np
from PIL import Image

from detection import detect, vis_keypoints, EXTENSIONS

# creating command line parameters parser
parser = argparse.ArgumentParser(description='Finding face keypoints')
parser.add_argument('model_ckpt', help='facepoints model (ckpt)')
parser.add_argument('dir_src', help='dir with face images; should be .jpg, .jpeg, .bmp or .png')
parser.add_argument('dir_out', help='dir for output')

args = parser.parse_args()

predicts = detect(args.model_ckpt, args.dir_src)
for path in sorted(filter(lambda name: any(name.endswith(ext) for ext in EXTENSIONS),
                          list(os.walk(args.dir_src))[0][2])):
    image = np.array(Image.open(os.path.join(args.dir_src, path)).convert('RGB'))
    keypoints = predicts[path].reshape(14, 2)
    image_with_kp = vis_keypoints(image, keypoints, diameter=3)
    Image.fromarray(image_with_kp).save(os.path.join(args.dir_out, path))
