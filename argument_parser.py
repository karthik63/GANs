import argparse
import numpy as np

class ArgumentParser:

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--n_epochs', help='The number of epochs to train', type=int, default=25)
        parser.add_argument('--learning_rate', help='Learning rate for adam optimiser', type=float, default=0.0002)
        parser.add_argument('--beta1', help='momentum term of adam' , type=float, default=0.5)
        parser.add_argument('--train_size', help='The size of train images', type=float, default=np.inf)
        parser.add_argument('--batch_size', help='The batch size of the images', type=int, default=1)
        parser.add_argument('--input_height', help='The height of the input image', type=int, default=28)
        parser.add_argument('--input_width', help='The width of the input image', type=int, default=28)
        parser.add_argument('--output_height', help='The height of the output image', type=int, default=28)
        parser.add_argument('--output_width', help='The width of the output image', type=int, default=28)
        parser.add_argument('--dataset', help='The name of the dataset', type=str, default='mnist')
        parser.add_argument('--input_fname_pattern', help='Glob pattern of filename of input images [*]',
                            type=str, default='.jpg')
        parser.add_argument('--checkpoint_dir', help='Directory name to save checkpoints', type=str,
                            default='checkpoints')
        parser.add_argument('--sample_dir', help='directory name to save image samples', type=str, default='samples')
        parser.add_argument('--train', help='set this option for training', dest='train', action='store_true')
        parser.add_argument('--test', help='set this option for testing', dest='train', action='store_false')
        parser.add_argument('--crop', help='set this option for training', dest='crop', action='store_true')
        parser.add_argument('--no_crop', help='set this option for testing', dest='crop', action='store_false')
        parser.add_argument('--visualise', help='select this option to visualise', dest='visualise',
                            action='store_true')
        parser.add_argument('--do_not_visualise', help='select this option to not visualise', dest='visualise',
                            action='store_false')
        parser.add_argument('--resize', help='True to resize image, false to add black border', dest='resize',
                            action='store_true')
        parser.set_defaults(visualise=False, train=True, crop=True, resize=True)
        self.args = parser.parse_args()
        print(self.args)

    def return_arguments(self):
        return self.args

k = ArgumentParser()

