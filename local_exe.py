import face_compare
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('img1', type=str, help="Input image 1.")
parser.add_argument('img2', type=str, help="Input image 2.")
args = parser.parse_args()

dist = face_compare.perform(args.img1 ,args.img2)
print "Distance computed:"
print dist