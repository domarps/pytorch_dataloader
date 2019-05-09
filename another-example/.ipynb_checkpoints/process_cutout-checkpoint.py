import torch
from data_loader import StockDataset, filtered_collate_fn
from isolation import *
BATCH_SIZE = 32
NUM_PREPROCESS_WORKERS = 1

checkpoint = '/home/ubuntu/repo-isolation/isolation_20171204.pth'
model = Isolation(checkpoint=checkpoint, gpu_id=0)
items = ['urls_for_pramod_cutout.csv']
dataset = StockDataset(items, model.preprocess)

output_csv = './64k_cutout.csv'

data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_PREPROCESS_WORKERS, drop_last=False, collate_fn=filtered_collate_fn)
try:
  for (cids, batch) in data_loader:
    try:
      output_probs = model.model(batch)
      cids, output_probs)
      print(output_probs[0])
    except Exception as e:
      print('Inner loop error:', e)
except Exception as e:
  print('Outer loop error:', e)
