
import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib
import time
import cv2

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf

_ROOT = '/home/mingtao.huang/'

MODEL_TYPE = 'owlii'

root_path = '/home/mingtao.huang/'
model_dir = '/home/mingtao.huang/log_repo/deploy/'
res_dir = model_dir + 'res/'
if not os.path.exists(res_dir):
  os.makedirs(res_dir)

FROZEN_GRAPH_NAME = model_dir + 'mobilenet_v3_035_deploy.pb'
#FROZEN_GRAPH_NAME = '/home/mingtao.huang/deploy_graph.pb'
INPUT_SIZE = 513

class SgmtModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'image:0'
  OUTPUT_TENSOR_NAME = 'heatmap:0'

  def __init__(self):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    graph_def = None
    with tf.gfile.GFile(FROZEN_GRAPH_NAME, "rb") as f:
      #print(FROZEN_GRAPH_NAME)
      graph_def = tf.GraphDef().FromString(f.read())

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')
    
    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    the_input = [np.asarray(resized_image)]

    t1 = time.time()
    heatmap = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: the_input})
    t2 = time.time()
    total = (t2 - t1) * 1000

    heatmap = heatmap[0, :, :, :]

    return heatmap, total

MODEL = SgmtModel()
def infer_one(image, use_heatmap=True):
  # image preprocessing
#  img = Image.open(image_path)
  width, height = image.size
  large_one = max(width, height)
  
  scale = float(INPUT_SIZE) / float(large_one)
  
  new_width = 0
  new_height = 0
  if width >= height:
    new_width = INPUT_SIZE
    new_height = int(height * scale)
  else:
    new_height = INPUT_SIZE
    new_width = int(width * scale)
  
  image = image.resize((new_width, new_height), Image.ANTIALIAS)
  # padding
  delta_w = INPUT_SIZE - new_width
  delta_h = INPUT_SIZE - new_height
  top, bottom = int(delta_h / 2), int(delta_h) - int(delta_h / 2)
  left, right = int(delta_w / 2), int(delta_w) - int(delta_w / 2)
  color = [127, 127, 127]
  img_array = np.array(image)
  img_array = cv2.copyMakeBorder(img_array, top, bottom, left, right, cv2.BORDER_CONSTANT,
      value=color)
  
  image = Image.fromarray(np.uint8(img_array))
  # run model
  
  heatmap, running_time = MODEL.run(image)
  heatmap = np.float32(heatmap) / 255.0
  heatmap = cv2.resize(heatmap,(INPUT_SIZE,INPUT_SIZE)).reshape((INPUT_SIZE,INPUT_SIZE,1))
  if not use_heatmap:
    heatmap = np.where(heatmap > 0.5, 1, 0)
    
  heatmap_array = np.squeeze(heatmap)
  if not use_heatmap:
    heatmap_array = heatmap_array>0.5
  else:
    heatmap_array = heatmap_array[int(delta_h / 2) : int(new_height + delta_h / 2), int(delta_w / 2) : int(new_width + delta_w / 2)]*255
    
  heatmap_crop = Image.fromarray(np.uint8(heatmap_array))
#  heatmap_crop.save('data/heatmap_tf.png')

  return heatmap_crop

#image = Image.open(root_path + 'test2.jpg')
#embed_crop, heatmap_crop, running_time,heat_map = infer_one(image, True)


'''
# now start inferring
with open(listpath) as f:
    lines = f.readlines()
lines = [x.strip('\n') for x in lines] 
#print(lines)

for filename in lines:
  filename_root = os.path.splitext(filename)[0]

  image = Image.open(data_dir + filename)
  embed_crop, heatmap_crop, running_time = infer_one(image, False)
  
  image.save(res_dir + filename)
  embed_crop.save(res_dir + filename_root + '.embed_tf.png')
  heatmap_crop.save(res_dir + filename_root + '.heatmap_tf.png')
  print('Time consumed on ', filename, ': ', running_time, ' ms.')
'''

root_path = '/home/mingtao.huang/'
data_path = root_path + 'dataset/coco/'
image_path = data_path + 'train2017/'
mask_path = data_path + 'mask_train2017/'

cnt = 0

for name in os.listdir(image_path):
    cnt += 1
    if cnt % 100 == 0:
        print(cnt)
    path = image_path + name
    image = Image.open(path)
    heatmap_crop = infer_one(image, False)
    #cv2.imwrite(mask_path + name.split('.')[0]+'.png',embed_crop)
    heatmap_crop.save(mask_path + name.split('.')[0]+'.png')