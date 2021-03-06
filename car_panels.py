"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 balloon.py train --dataset=/path/to/balloon/dataset --weights=imagenet

    # Apply color splash to an image
    python3 balloon.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 balloon.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import pandas
import cv2
# Root directory of the project
from skimage.measure import find_contours

from mrcnn.visualize import random_colors, apply_mask

ROOT_DIR = os.path.abspath("")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class CarConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "car"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 30  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8


############################################################
#  Dataset
############################################################

class CarsDataset(utils.Dataset):

    def load_cars_data(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        with open(ROOT_DIR+"/labels.json", "r") as f:
            class_names = json.load(f)
        f.close()
        for dir, panel in class_names.items():
            if dir != "damage":
                for key, value in panel.items():
                    self.add_class("cars", value, dir+key)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        labeled_data_path = os.path.join(ROOT_DIR, "labels", subset)
        file_names = os.listdir(labeled_data_path)

        for file in file_names:
            if file == "30.json" or file == "2.json" or file == "1.json":
                annotations = json.load(open(os.path.join(labeled_data_path, file), 'r'))
                annotations = list(annotations["_via_img_metadata"].values())
                annotations = [a for a in annotations if len(a["regions"]) > 0]
                print (file)
                for a in annotations:
                    polygons = []
                    objects = []
                    regions = a["regions"]
                    
                    for r in regions:
                        if not r["region_attributes"]["damage"].strip():
                            if r["region_attributes"]["front"].strip() or r["region_attributes"]["rear"].strip() or r["region_attributes"]["left"].strip() or r["region_attributes"]["right"].strip():
                                polygons.append(r['shape_attributes'])
                                objects.append(r['region_attributes'])

                    class_ids = []
                    print(file, "    ", a["filename"])

                    for c in objects:
                        if c["front"].strip():
                            class_ids.append(class_names["front"][c["front"].strip()])
                        elif c["rear"].strip():
                            class_ids.append(class_names["rear"][c["rear"].strip()])
                        elif c["left"].strip():
                            class_ids.append(class_names["left"][c["left"].strip()])
                        elif c["right"].strip():
                            class_ids.append(class_names["right"][c["right"].strip()])
                        else:
                            pass
                    image_name = a['filename']
                    image_path = os.path.join(ROOT_DIR, "../data2/"+file.split(".")[0], image_name)
                    image = skimage.io.imread(image_path)
                    height, width = image.shape[:2]

                    self.add_image(
                        "cars",
                        image_id=image_name,  # use file name as a unique image id
                        path=image_path,
                        width=width, height=height,
                        polygons=polygons,
                        class_ids=class_ids)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "cars":
            return super(self.__class__, self).load_mask(image_id)
        class_ids = image_info['class_ids']
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"] ,len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon( p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # class_ids=np.array([self.class_names.index(shapes[0])])
        # print("info['class_ids']=", info['class_ids'])
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids  # [mask.shape[-1]] #np.ones([mask.shape[-1]], dtype=np.int32)#class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "food":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CarsDataset()
    dataset_train.load_cars_data(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CarsDataset()
    dataset_val.load_cars_data(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")

    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=40,
                layers='heads')

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=120,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=160,
                layers='all')

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    blank_image = np.zeros(image.shape,np.uint8)
    blank_image[:,:] = (158, 224, 255)
    # padded_mask = np.zeros(
    #     (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
    # padded_mask[1:-1, 1:-1] = mask
    # contours = visualize.find_contours(padded_mask, 0.5)
    # Copy color pixels from the original color image where mask is set

    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask,blank_image, image).astype(np.uint8)

        splash = cv2.addWeighted(splash, 0.6, image, 0.4, 0)
        # cv2.imshow("blank", splash)
        # cv2.waitKey()
        # splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash

def save_masked_image(image, boxes, masks, class_ids, class_names,
                      scores=None, captions=None) :
    N = boxes.shape[0]

    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    color = (165, 160, 65)
    image_copy = image.copy()
    for i in range(N):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]

        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.7, (237, 95, 223), lineType=cv2.LINE_AA)

        # Mask
        mask = masks[:, :, i]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            verts = verts.reshape((-1, 1, 2))
            cv2.polylines(image, np.int32(verts), True, color, 2)
            cv2.fillPoly(image_copy, np.int32([verts]), color)
            cv2.addWeighted(image_copy,0.1, image, 0.9, 0, image)

    return image

def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path
    class_names = {}
    with open(ROOT_DIR + "/labels.json", "r") as f:
        class_names = json.load(f)

    labels = {}
    for dir, panel in class_names.items():
        for key, value in panel.items():
            labels[value] = dir+key
    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        import cv2
        image = cv2.imread(args.image)
        r = model.detect([image], verbose=1)[0]
        # splash = color_splash(image, r['masks'])
        # Save output
        splash = save_masked_image(image, r['rois'], r['masks'], r['class_ids'], labels, r['scores'])
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        cv2.imwrite(file_name, splash)

    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/balloon/dataset/",
                        help='Directory of the Balloon dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = CarConfig()
    else:
        class InferenceConfig(CarConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        print (COCO_WEIGHTS_PATH)
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        # img_list = os.listdir(args.image)
        # for img in img_list[20:120]:
        #     img_
        #     print (img)
        detect_and_color_splash(model, image_path=args.image,
                            video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
