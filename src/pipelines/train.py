import os
import torch
from src.components.args_parser import arg_parser
import torch.nn as nn
from torch.optim import Adam
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from src.components.stylized.single import MulLayer
from src.components.criterion import LossCriterion
from src.components.network import image_encoder
from src.components.network import decoder
from src.components.network import loss_network
from src.utils import print_options
from src.pipelines.data_ingestion import Getdata
cudnn.benchmark = True

import sys
from src.logger import logging
from src.exception import CustomException


class StyleTransferTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.encoder = image_encoder()
        self.decoder = decoder()
        self.loss_net = loss_network()
        self.matrix = MulLayer(z_dim = args.latent)
        self.optimizer = Adam(self.matrix.parameters(), args.lr)
        self.criterion = LossCriterion(args.style_layers,
                                       args.content_layers,
                                       args.style_weight,
                                       args.content_weight)
        self.getdata = Getdata()
    
    def load_dataset(self):
        try:
            logging.info("Loading train set")
            train_set = self.getdata.get_training_set(self.args.data_dir)
            loaded_train_data = DataLoader(dataset=train_set, num_workers=self.args.threads, batch_size=self.args.batchSize, shuffle=True)
            logging.info("Loading Success")
            return loaded_train_data
        
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)
    
    def train_epoch(self, epoch):
        try:
            epoch_loss = 0
            train_data = self.load_dataset()
            logging.info("Training...")
            for iteration, batch in enumerate(train_data, 1):
                content, target, style = Variable(batch[0], Variable(batch[1], Variable(batch[2])))
                content = content.to(self.device)
                target = target.to(self.device)
                style = style.to(self.device)

                self.optimizer.zero_grad()

                # forward
                sF = self.encoder(style)
                cF = self.encoder(content)

                feature, transmatrix, KL = self.matrix(cF[self.args.layer], sF[self.args.layer])

                transfer = self.decoder(feature)

                sF_loss = self.loss_net(style)
                cF_loss = self.loss_net(content)
                transfer_loss = self.loss_net(transfer)

                loss, styleLoss, contentLoss, KL = self.criterion(transfer_loss, sF_loss, cF_loss, KL)

                loss.backward()
                self.optimizer.step()
                logging.info("===> Epoch[{}]({}/{}): loss: {:.4f} || content: {:.4f} || style: {:.4f} KL: {:.4f}."
                             .format(epoch, iteration,len(train_data), loss, contentLoss, styleLoss, KL,))
                epoch_loss += loss
            logging.info("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(train_data)))

            return content, style, transfer
        
        except Exception as e:
            logging.info(e)
            raise CustomException(e)
    
    def save_results(self, epoch, content, style, transfer):
        try:
            content = content.clamp(0,1).cpu().data
            style = style.clamp(0,1).cpu().data
            transfer = transfer.clamp(0,1).cpu().data
            concat = torch.cat((content, style, transfer), dim=0)
            vutils.save_image(concat, '%s/%d.png' % (self.args.outf, epoch), normalize = True, scale_each=True, nrow=self.args.batchSize)

            torch.save(self.matrix.state_dict(), "%s/%s_epoch_%d.pth" % (self.args.outf, self.args.layer, epoch))
        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

    def train(self):
        try:
            for epoch in range(self.args.start_iter, self.args.nEpochs + 1):
                content, style, transfer = self.train_epoch(epoch)

                if (epoch + 1) % 100 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 10.0
                    logging.info('Learning Rate decay: lr = {}'.format(self.optimizer.param_groups[0]['lr']))

                if epoch % (self.args.snapshots) == 0:
                    self.save_results(epoch, content, style, transfer)
            logging.info("Training Finished")

        except Exception as e:
            logging.info(e)
            raise CustomException(e,sys)

