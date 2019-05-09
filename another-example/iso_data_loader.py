import numpy as np
import io, os, glob, json
from PIL import Image
import base64

def bytes_from_base64_string(base64_string):
  return base64.b64decode(base64_string)

import torch.utils.data


CUTOUT_PATH = '/home/ubuntu/images_with_cutout'
class StockDataset(torch.utils.data.Dataset):
  def __init__(self, chunk_files, preprocess_fn):
    # self.extensions = ['jpg', 'jpeg', 'png']
    self.preprocess = preprocess_fn
    self.transform = None
    self.items = []
    for chunk_file in chunk_files:
      chunk_items = self._read_chunk_file(chunk_file)
      self.items.extend(chunk_items)

  def __len__(self):
    return len(self.items)

  def _read_chunk_file(self, chunk_file):
    # items = []
    with open(chunk_file, 'r') as f:
      items = list(f) # NEW (replaces below)
      '''
      for line in f:
        data_dict = json.loxads(line)
        items.append(data_dict)
      '''
    return items

  # def _prepare_item(self, item):
  #   item = json.loads(item) # NEW
  #   cid = item['cid']
  #   image_data = item['image']
  #   image_data = bytes_from_base64_string(image_data)
  #   new_item = (cid, image_data)
  #   return new_item

  def __getitem__(self, index):
    try:
        item = self.items[index]
        url = item.strip()
        cid = url.split('_')[2]
        filename = os.path.join(CUTOUT_PATH, url.split('/')[-1])
        pil_image = Image.open(filename)
        sample = self.loader(pil_image)
        return url, cid, torch.squeeze(sample)
    except:
        return None



     
  def loader(self, image_data):
    try:
      image = self.preprocess(image_data)
      return image
    except Exception as e:
      print('ERROR in StockDataset.loader:', e)
      return None

  # def __getitem__(self, index):
  #   item = self.items[index]
  #   prepared = self._prepare_item_for_cutout(item)
  #   cid, image_data = prepared
  #   sample = self.loader(image_data)
  #   if sample is None:
  #     # Exception in loader caught. Filter this item out via custom collate_fn below.
  #     return None
  #   #print(sample.shape)
  #   #sample = np.ascontiguousarray(sample)
  #   #print(sample.shape)
  #   # sample = np.ascontiguousarray(sample, dtype=np.float32) # what should dtype be?
  #   return cid, sample

def filtered_collate_fn(batch):
  # Skip errors. __get_item__ returns None if catches an exception.
  return torch.utils.data.dataloader.default_collate([x for x in batch if x is not None])
