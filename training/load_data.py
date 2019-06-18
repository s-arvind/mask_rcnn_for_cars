import os
import json
import numpy as np
import copy

# path = "../videos/"
label_path = "/Users/arvind/Documents/deeplearningvideo/labels/"

import skimage


# list_dir = os.listdir(path)

# print (list_dir)


class Dataset(object):

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        self.class_map = {}
        # Background is always the first class
        with open("labels.json", "r") as f:
            classes = json.load(f)
        f.close()
        self.class_map = copy.deepcopy(classes)
        self.class_info = list(self.class_map.keys())
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, regions, name, image_id):
        image_info = {
            "regions": regions,
            "name": name,
            "image_id": image_id
        }

        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.

        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.

        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        # Mapping from source class and image IDs to internal IDs
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.

        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's available online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image_path = os.path.join("~/Documents/data/",image_id["name"])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        print (image_path)
        return (image, height, width)

    def load_mask(self, image_id, height, width):
        """Generate instance masks for an image.
          Returns:
          masks: A bool array of shape [height, width, instance count] with
              one mask per instance.
          class_ids: a 1D array of class IDs of the instance masks.
          """
        # If not a balloon dataset image, delegate to parent class

        image_masks = image_id["regions"]
        # image_region = image_id["region"]
        # if image_info["source"] != "damage":
        #     return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        # info = self.image_info[image_id]
        len_mask = 0
        for i in image_masks:
            key = value = ""
            for k, v in i["region_attributes"].items():
                if len(v.strip()) > 0 and k != "damage":
                    key = k.strip()
                    value = v.split('\\')[0]
                    break
            if i["shape_attributes"]["name"] == "polyline" and key and key != "damage":
                len_mask += 1

        mask = np.zeros([height, width, len_mask],
                        dtype=np.uint8)
        class_ids = []
        for i, p in enumerate(image_masks):
            # Get indexes of pixels inside the polygon and set them to 1
            key = value = ""
            for k, v in p["region_attributes"].items():
                if len(v.strip()) > 0 and k != "damage":
                    key = k.strip()
                    value = v if "\n" not in v else v[:-1]
                    break
            if key and key != "damage" and p["shape_attributes"]["name"] == "polyline":
                rr, cc = skimage.draw.polygon(p["shape_attributes"]['all_points_y'],
                                              p["shape_attributes"]['all_points_x'])

                mask[rr, cc, i] = self.class_map[key][value]
                class_ids.append(self.class_map[key][value])

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s

        return mask.astype(np.bool), np.array(class_ids, dtype=np.int32)


def load_dataset(dataset):
    print("function called")
    train = os.listdir(os.path.join(label_path,"train"))
    val = os.listdir(os.path.join(label_path,"val"))
    image_train_dataset = Dataset()
    image_val_dataset = Dataset()
    for label in train:
        file = os.path.join(label_path, "train", label)
        print (file)
        with open(file, "r") as f:
            label_json = json.load(f)
        f.close()
        list_values = list(label_json["_via_img_metadata"].values())
        annotations = [file for file in list_values if len(file["regions"]) > 0]
        path = label.split(".")[0]
        counter = 0
        for file in annotations:  #
            image_name = path + "_" + file["filename"]
            image_regions = file["regions"]
            image_id = counter
            counter += 1
            image_train_dataset.add_image(image_regions, image_name, image_id)

    for label in val:
        file = os.path.join(label_path, "val", label)
        print (file)
        with open(file, "r") as f:
            label_json = json.load(f)
        f.close()
        list_values = list(label_json["_via_img_metadata"].values())
        annotations = [file for file in list_values if len(file["regions"]) > 0]
        path = label.split(".")[0]
        counter = 0
        for file in annotations:  #
            image_name = path + "_" + file["filename"]
            image_regions = file["regions"]
            image_id = counter
            counter += 1
            image_val_dataset.add_image(image_regions, image_name, image_id)


    return (image_train_dataset, image_val_dataset)


