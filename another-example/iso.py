import numpy as np
from PIL import Image
import io
import torch

def bytes_to_pil_image(raw_bytes):
    '''
    image bytes to pil_image
    '''
    pil_image = Image.open(io.BytesIO(raw_bytes))
    return pil_image

def preprocess_image(image):
    image = image.resize((224, 224), Image.LANCZOS)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if hasattr(image, '_getexif'):
        try:
            exif = image._getexif()
            if exif:
                rotation = {3: 180, 6: 270, 8: 90}.get(exif.get(274))
                if rotation:
                    image = image.rotate(rotation, expand=True)
        except:
            pass

    image = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    image = 2 * (image / 255.0 - 0.5)
    image = image[np.newaxis, :]
    image_data = torch.from_numpy(image)
    return image_data
    #image_var = torch.autograd.Variable(torch.from_numpy(image)).type(torch.FloatTensor)
    #return image_var
