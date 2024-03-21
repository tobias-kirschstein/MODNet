from unittest import TestCase

import torch
from torch import nn

from modnet.models.modnet import MODNet
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F

class ModNetTest(TestCase):
    def test_modnet(self):
        ckpt_path = "C:/Users/kirschstein/Downloads/modnet_photographic_portrait_matting.ckpt"
        im_names = ["C:/Users/kirschstein/Downloads/img00000000.png"]

        # define hyper-parameters
        ref_size = 512



        # define image to tensor transform
        im_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        # create MODNet and load the pre-trained ckpt
        modnet = MODNet(backbone_pretrained=False)
        modnet = nn.DataParallel(modnet)

        if torch.cuda.is_available():
            modnet = modnet.cuda()
            weights = torch.load(ckpt_path)
        else:
            weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
        modnet.load_state_dict(weights)
        modnet.eval()

        # inference images
        for im_name in im_names:
            print('Process image: {0}'.format(im_name))

            # read image
            im = Image.open(im_name)

            # unify image channels to 3
            im = np.asarray(im)
            if len(im.shape) == 2:
                im = im[:, :, None]
            if im.shape[2] == 1:
                im = np.repeat(im, 3, axis=2)
            elif im.shape[2] == 4:
                im = im[:, :, 0:3]

            # convert image to PyTorch tensor
            im = Image.fromarray(im)
            im = im_transform(im)

            # add mini-batch dim
            im = im[None, :, :, :]

            # resize image for input
            im_b, im_c, im_h, im_w = im.shape
            if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
                if im_w >= im_h:
                    im_rh = ref_size
                    im_rw = int(im_w / im_h * ref_size)
                elif im_w < im_h:
                    im_rw = ref_size
                    im_rh = int(im_h / im_w * ref_size)
            else:
                im_rh = im_h
                im_rw = im_w

            im_rw = im_rw - im_rw % 32
            im_rh = im_rh - im_rh % 32
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

            # inference
            _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

            # resize and save matte
            matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
            matte = matte[0][0].data.cpu().numpy()
            matte_name = im_name.split('.')[0] + '.png'
            #Image.fromarray(((matte * 255).astype('uint8')), mode='L').save(os.path.join(args.output_path, matte_name))