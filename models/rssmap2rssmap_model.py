import torch
from .base_model import BaseModel
from . import networks


class RssMap2RssMapModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.add_argument('--blocked_size', type=int, default=0, help='size of blocked area')
        parser.add_argument('--mask_cx', type=int, default=32, help='center x of mask')
        parser.add_argument('--mask_cy', type=int, default=32, help='center y of mask')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_T', type=float, default=100.0, help='weight for T loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['T_A']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.text_names = []

        if self.isTrain:
            self.model_names = ['T']
        else:  # during test time, only load G
            self.model_names = ['T']

        blocked_size = opt.blocked_size
        self.mask = torch.ones((1, 1, 64, 64)).float().cuda()
        self.mask[:, :, opt.mask_cx-blocked_size:opt.mask_cx+blocked_size, opt.mask_cy-blocked_size:opt.mask_cy+blocked_size] = 0

        self.netT = networks.define_T(init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, mask=self.mask)

        if self.isTrain:
            # define loss functions
            self.criterionT = torch.nn.SmoothL1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_T = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
            self.optimizers.append(self.optimizer_T)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.tx_loc_pwr = input['tx_loc_pwr'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if not self.opt.isTrain: # this is for testing only; during training, we will get
            self.task_A = self.netT(self.real_A) # T(A)

    def backward_T(self):
        """Calculate TaskNetwork loss"""
        self.optimizer_T.zero_grad()        # set T's gradients to zero
        self.task_A = task_A = self.netT(self.real_A) # T(A)
        self.loss_T_A = self.criterionT(task_A, self.tx_loc_pwr)
        self.loss_T_A.backward(retain_graph=True)
        self.optimizer_T.step()             # udpate T's weights

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        # update T
        self.set_requires_grad(self.netT, True)  # D requires no gradients when optimizing T
        self.backward_T()                   # calculate graidents for T