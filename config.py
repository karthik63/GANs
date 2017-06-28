
class Config:
    def __init__(self, args):
        self.crop = args.crop

        self.batch_size = args.batch_size

        self.input_height = args.input_height
        self.input_width = args.input_width
        self.output_height = args.output_height
        self.output_width = args.output_width

        self.n_epochs = args.n_epochs
        self.learning_rate = args.learning_rate
        self.beta1 = args.beta1
        self.train_size = args.train_size

        self.dataset = args.dataset
        self.input_fname_pattern = args.input_fname_pattern
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir

        self.train = args.train
        self.visualise = args.visualise
        self.resize = args.resize
