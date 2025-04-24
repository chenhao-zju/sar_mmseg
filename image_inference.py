import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from uuid import uuid1
from PIL import Image
import numpy as np
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor

import mmcv
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], [0, 11, 123], \
                         [118, 20, 12], [122, 81, 25], [241, 134, 51], [120, 240, 90], [244, 20, 57]]
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def image_inference(image_path, model_path):
    palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], \
               [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51], [120, 240, 90], [244, 20, 57]]
    
    uuid = str(uuid1())
    output_image = f"./results/{uuid}.png"

    model = init_segmentor("./config.py", model_path, device='cuda:0')
    seg_map = inference_segmentor(model, image_path)[0].astype('uint8')
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    seg_img.save(output_image)

    print(f"The segmented image has been saved at {output_image}")

if __name__ == "__main__":

    parser = ArgumentParser()  
    parser.add_argument("--input_image", default='./examples/demo.png', help="input image path")
    parser.add_argument("--model", default='./checkpoints/iter_100000.pth', help="model path")

    args = parser.parse_args()

    image_inference(args.input_image, args.model)