import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='parser for style-transfer')
    subparsers = parser.add_subparsers(title='subcommands', dest='subcommand')

    train_parser = subparsers.add_parser("train", help="parser for training arguments")
    
    train_parser.add_argument("--vgg_dir", default='artifacts/models/vgg_r41.pth', help='pre-trained encoder path')
    train_parser.add_argument("--loss_network_dir", default='artifacts/models/vgg_r51.pth', help='used for loss network')
    train_parser.add_argument("--decoder_dir", default='artifacts/models/dec_r41.pth', help='pre-trained decoder path')
    train_parser.add_argument('--pretrained', type=int, default=1)
    train_parser.add_argument("--matrix_dir", default='artifacts/models/matrix_r41_new.pth', help='pre-trained matrix path')
    train_parser.add_argument("--data_dir", default="artifacts/train_data", help='path to training dataset')
    train_parser.add_argument('--data_augmentation', type=int, default=1)
    train_parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
    train_parser.add_argument("--outf", default="artifacts/output/", help='folder to output images and model checkpoints')
    train_parser.add_argument("--content_layers", default="r41", help='layers for content')
    train_parser.add_argument("--cuda", type=int, required=True, help='set it to 1 for running on GPU, 0 for CPU')
    train_parser.add_argument("--style_layers", default="r11,r21,r31,r41", help='layers for style')
    train_parser.add_argument("--batchSize", type=int,default=8, help='batch size')
    train_parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
    train_parser.add_argument("--content_weight", type=float, default=1.0, help='content loss weight')
    train_parser.add_argument("--style_weight", type=float, default=0.02, help='style loss weight, 0.02 for origin')
    train_parser.add_argument("--log_interval", type=int, default=500, help='log interval')
    train_parser.add_argument("--save_interval", type=int, default=5000, help='checkpoint save interval')
    train_parser.add_argument("--layer", default="r41", help='which features to transfer, either r31 or r41')
    train_parser.add_argument("--latent", type=int, default=1024, help='length of latent vector')
    train_parser.add_argument('--nEpochs', type=int, default=5000, help='number of epochs to train for')
    train_parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
    train_parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')


    pred_parser = subparsers.add_parser("predict", help="parser for prediction")
    
    pred_parser.add_argument("--cuda", type=int, required=True, help='set it to 1 for running on GPU, 0 for CPU')
    pred_parser.add_argument("--content-image", type=str, default='artifacts/predict/input', help="Path to content image(s)")
    pred_parser.add_argument("--style-image", type=str, default="artifacts/predict/style", help="Path to style image(s)")
    pred_parser.add_argument("--output_image", type=str, default="artifacts/output")
    pred_parser.add_argument("--encoder_dir", type=str, default="artifacts/models/vgg_r41.pth")
    pred_parser.add_argument("--decoder_dir", type=str, default="artifacts/models/dec_r41.pth")
    pred_parser.add_argument("--matrix_dir", type=str, default='artifacts.models/matrix_r41_new.pth')
    pred_parser.add_argument("--latent", type=int, default=256, help="length of latent vectors")
    pred_parser.add_argument("--multistyle", type=bool, default=0, help="1 for multi style transfer or 0 for single style transfer on the content image")
    # pred_parser.add_argument("--layer", type=str, default="r41", help="which features to transfer, either r31 or r41")

    args = parser.parse_args()
    return args