from styx_msgs.msg import TrafficLight
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Frozen inference graph files. NOTE: change the path to where you saved the models.
#SSD_GRAPH_FILE = 'frozen_sim_inception/frozen_inference_graph.pb'
SSD_GRAPH_FILE = 'frozen_real_inception/frozen_inference_graph.pb'


#
# Utility funcs
#

def filter_boxes(min_score, boxes, scores, classes):
    """Return boxes with a confidence >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)

    filtered_boxes = boxes[idxs, ...]
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_boxes, filtered_scores, filtered_classes


def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords


def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    # Colors (one for each class)
    cmap = ImageColor.colormap
    print("Number of colors =", len(cmap))
    COLOR_LIST = sorted([c for c in cmap.keys()])
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)


class TLClassifier(object):
    def __init__(self):


        """Loads a frozen inference graph"""
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(SSD_GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        distance = 1e6
        state = TrafficLight.UNKNOWN

        with tf.Session(graph=self.graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.2
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            width, height = image.size
            box_coords = to_image_coords(boxes, height, width)

            # Each class with be represented by a differently colored box
            draw_boxes(image, box_coords, classes)

            #plt.figure(figsize=(12, 8))
            #plt.imshow(image)

            # pick the closes traffic light
            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > confidence_cutoff:
                    class_name = classes[i]
                    print('{}'.format(class_name), scores[i])

                    #fx = 0.97428
                    #fy = 1.73205

                    fx = 1345.200806
                    fy = 1353.838257
                    
                    perceived_width_x = (boxes[i][3] - boxes[i][1]) * 800
                    perceived_width_y = (boxes[i][2] - boxes[i][0]) * 600

                    # ymin, xmin, ymax, xmax = box
                    # depth_prime = (width_real * focal) / perceived_width
                    perceived_depth_x = ((.1 * fx) / perceived_width_x)
                    perceived_depth_y = ((.3 * fy) / perceived_width_y)

                    estimated_distance = (perceived_depth_x + perceived_depth_y) / 2
                    if estimated_distance < distance:
                        distance = estimated_distance;
                        state = TrafficLight(class_name)
                        print("state ", state )
                        print("Distance (metres)", estimated_distance)


        return distance, state
