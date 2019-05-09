import os
import numpy as np
from PIL import Image
import torch
from net import Net
import json, base64

import isolation_util

BATCH_SIZE = 32

class Isolation(object):

    def __init__(self, checkpoint, gpu_id=0):
        assert os.path.exists(checkpoint)
        self.net = Net(num_classes=5)
        self.net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))
        
        if torch.cuda.is_available():
            self.net.cuda(device=gpu_id)
            self.dtype_float = torch.cuda.FloatTensor        
        else:
            self.dtype_float = torch.FloatTensor
        self.net.eval()
        self.softmax_func = torch.nn.Softmax()
    
    def eval(self, image):
        image_var = self._image_preprocess(image)
        scores = self.net(image_var)
        scores = self.softmax_func(scores)
        scores = scores.data.numpy()[0].tolist()
        rsp = {
            "Isolated on a white background": scores[0],
            "Isolated on a non-white background": scores[1],
            "Isolated on a contextual background": scores[2],
            "Partial isolation": scores[3],
            "Non-isolation": scores[4]
        }
        return rsp
    
    def model(self, image_batch):
        self.image_var = torch.autograd.Variable(torch.Tensor(len(image_batch), 3, 224, 224)).type(self.dtype_float).cuda()
        self.image_var.data.copy_(image_batch)
        scores = self.net(self.image_var)
        scores = self.softmax_func(scores)
        scores = scores.cpu().data.numpy()
        return scores
    
    def preprocess(self, pil_image):
        #pil_image = isolation_util.bytes_to_pil_image(raw_image_bytes)
        image = isolation_util.preprocess_image(pil_image)
        return image
    

if __name__ == '__main__':
    image_path = 'test.jpg'
    checkpoint = 'isolation_20171204.pth'
    isolation = Isolation(checkpoint=checkpoint)
    with open('test-chunk.json', 'r') as f:
        items = list(f)
        
        
    image_batch = []
    for item in items:
        item = json.loads(item)
        raw_image_bytes = base64.b64decode(item['image'])
        image_batch.append(isolation.preprocess(raw_image_bytes))
        
    images = torch.squeeze(torch.stack(image_batch))
    for i in range(0, images.shape[0], BATCH_SIZE):
        print('processing batch {}'.format(i // 32))
        batch = images[i:i+BATCH_SIZE]
        scores = isolation.model(batch)
        print(scores.shape, scores[0])