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
        """
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'T_A', 'T_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.text_names = ['tx_loc_pwr', 'task_A', 'task_B']
        """
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_task_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.text_names = ['tx_loc_pwr', 'latent_coords']


        if self.isTrain:
            #self.model_names = ['G', 'D', 'T']
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            #self.model_names = ['G', 'T']
            self.model_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        blocked_size = opt.blocked_size
        self.mask = torch.ones((1, 1, 64, 64)).float().cuda()
        self.mask[:, :, opt.mask_cx-blocked_size:opt.mask_cx+blocked_size, opt.mask_cy-blocked_size:opt.mask_cy+blocked_size] = 0

        #self.netT = networks.define_T(init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids, mask=self.mask)

        if self.isTrain:  # define a discriminator; only single channel input here 
            assert(opt.input_nc == opt.output_nc)
            self.netD = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain,self.gpu_ids, mask=self.mask)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionT = torch.nn.SmoothL1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            #self.optimizer_T = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)) 
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            #self.optimizers.append(self.optimizer_T)

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
        self.fake_B = self.netG(self.real_A)  # G(A)
        """
        if not self.opt.isTrain: # this is for testing only; during training, we will get
            self.task_A = self.netT(self.real_A) # T(A)
            self.task_B = self.netT(self.fake_B) # T(G(A))
        """
        
        # extract latent value -- careful! the first dimension here is the BATCH index!
        # we also might have to `copy_` in order to avoid messing up the differentiable history
        # of our generator?
        self.latent_coords = networks.latent_val[:,0:2].squeeze()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_B = self.real_B
        pred_real = self.netD(real_B)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_B = self.fake_B
        pred_fake = self.netD(fake_B)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Should we keep this L1 loss?
        # ---------------------
        # Second, G(A) = B
        #self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # ---------------------

        # Task constraint on encoder
        # ---------------------
        self.loss_G_task_L1 = self.criterionT(self.latent_coords, self.tx_loc_pwr[:,0:2]) * self.opt.lambda_T
        # ---------------------

        # Combine loss and calculate gradients
        # ---------------------
        #self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_task_L1
        self.loss_G = self.loss_G_GAN + self.loss_G_task_L1
        self.loss_G.backward()

    """
    def backward_T(self):
        Calculate TaskNetwork loss
        self.optimizer_T.zero_grad()        # set T's gradients to zero
        self.task_A = task_A = self.netT(self.real_A) # T(A)
        self.loss_T_A = self.criterionT(task_A, self.tx_loc_pwr)
        self.loss_T_A.backward(retain_graph=True)
        self.optimizer_T.step()             # udpate T's weights

        self.optimizer_T.zero_grad()        # set T's gradients to zero
        self.task_B = task_B = self.netT(self.fake_B) # T(G(A))
        self.loss_T_B = self.criterionT(task_B, self.tx_loc_pwr)
        self.loss_T_B *= self.opt.lambda_T
        self.loss_T_B.backward(retain_graph=True)
        self.optimizer_T.step()
    """

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights

        """
        # update T
        self.set_requires_grad(self.netT, True)  # D requires no gradients when optimizing T
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_T()                   # calculate graidents for T
        """

        # update G
        self.optimizer_G.zero_grad()
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights

