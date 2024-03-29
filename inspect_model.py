from tensorflow.keras.applications import VGG16
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include_top", type=int, default=1, help="Whether or not to include the top layers of model")
args = vars(ap.parse_args())

if args['include_top'] > 0:
    _include_top = True
else:
    _include_top = False

print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=_include_top)

print("[INFO] showing layers...")
for (i, layer) in enumerate(model.layers):
    print("[INFO] {}\t{}".format(i, layer.__class__.__name__))



