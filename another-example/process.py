import torch
from data_loader import StockDataset, filtered_collate_fn
from isolation import *
BATCH_SIZE = 32
NUM_PREPROCESS_WORKERS = 1
import isolation_util
checkpoint = '/home/ubuntu/repo-isolation/isolation_20171204.pth'
model = Isolation(checkpoint=checkpoint, gpu_id=0)
items = ['urls_for_pramod_cutout.csv']
dataset = StockDataset(items, isolation_util.preprocess_image)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_PREPROCESS_WORKERS, drop_last=False, collate_fn=filtered_collate_fn)
url_outputs = []
cid_outputs = []
proba_outputs = []
try:
  for i, (urls, cids, batch) in enumerate(data_loader):
    try:
      output_probs = model.model(batch)
      print('processing batch {}'.format(i))
      for (url, cid, output_prob) in zip(urls, cids, output_probs):
            cid_outputs.append(cid)
            url_outputs.append(url)
            proba_outputs.append(output_prob)
    except Exception as e:
      print('Inner loop error:', e)
except Exception as e:
  print('Outer loop error:', e)