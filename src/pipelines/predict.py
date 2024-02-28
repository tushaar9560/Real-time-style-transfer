import os
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from src.components.network import image_encoder, decoder
from src.components.stylized import single, multi
from src.logger import logging
from src.exception import CustomException
import sys
from src.components.args_parser import arg_parser

class StyleTransfer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.encoder = image_encoder.Encoder().to(self.device)
        self.decoder = decoder.Decoder().to(self.device)
        self.smatrix = single.MulLayer(z_dim = self.args.latent)
        self.mmatrix = multi.MulLayer_4x(z_dim= self.args.latent)
        self.encoder.load_state_dict(torch.load(self.args.encoder_dir, map_location=self.device))
        self.decoder.load_state_dict(torch.load(self.args.decoder_dir))
        self.smatrix.load_state_dict(torch.load(self.args.matrix_dir))
        self.mmatrix.load_state_dict(torch.load(self.args.matrix_dir))
        self.encoder.eval()
        self.decoder.eval()
        self.test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def stylize_image(self, content_image_path, style_image_paths, output_dir):
        try:
            content_image = Image.open(content_image_path).convert('RGB')
            style_images = [Image.open(path).convert('RGB') for path in style_image_paths]

            content_tensor = self.test_transform(content_image).unsqueeze(0).to(self.device)
            style_tensors = [self.test_transform(style_images).unsqueez(0).to(self.device)]

            with torch.no_grad():
                content_features = self.encoder(content_tensor)
                style_features = [self.encoder(style_tensor) for style_tensor in style_tensors]

            if self.args.multistyle:
                stylized_image = self.multi_style_transfer(content_features, style_features)
                output_img_path = self.save_image(stylized_image, output_dir)
                logging.info(f"Style Image saved at {output_img_path}")
            else:
                for idx, feature in enumerate(style_features):
                    stylized_image = self.single_style_transfer(content_features, feature)
                    output_img_path = self.save_image(stylized_image, output_dir, i=idx)
                    logging.info(f"Style Image saved at {output_img_path}")
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
    def single_style_transfer(self, content_features, style_feature):
        try:
            logging.info("Single Style transfering...")
            with torch.no_grad():
                cF = content_features[self.args.layer]
                sF = style_feature[self.args.layer]
                feature, _, _ = self.smatrix(cF, sF)
                transfer = self.decoder(feature)
            return transfer
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)


    def multi_style_transfer(self, cF, sF):
        try:
            logging.info("Multi Style transferring...")
            transfer = torch.zeros_like(cF)
            for a in range(0,5,1):
                for b in range(0, 5, 1):
                    aa = a/5.0
                    bb = b/5.0
                    k = (1 - aa) * (1 - bb)
                    l = bb * (1 - aa)
                    m = aa * (1 - bb)
                    n = aa * bb
                    with torch.no_grad():
                        weighted_feature, _, _ = self.mmatrix(k,l,m,n, cF[self.args.layer], sF[0][self.args.layer],sF[1][self.args.layer],sF[2][self.args.layer],sF[3][self.args.layer])
                        weighted_transfer = self.decoder(weighted_feature)
                        transfer += weighted_transfer
            transfer /= len(sF)
            return transfer        
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
    def save_image(stylized_img, output_dir, i=""):
        stylized_image = stylized_image.squeeze(0).cpu()
        stylized_image = transforms.ToPILImage()(stylized_image.clamp(0,1))
        output_img_path = os.path.join(output_dir, f'stylized_image{i}.jpg')
        stylized_image.save(output_img_path)
        return output_img_path