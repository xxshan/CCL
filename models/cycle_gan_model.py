import torch
import os
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
import torch.nn.functional as F
class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if opt.use_gpu else torch.device('cpu')
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
       
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        '''if(opt.use_pre_gan):
            self.netG_A.load_state_dict({k.replace('module.',''):v for k,v in torch.load(os.path.join(opt.pretrained_path, 'latest_net_G_A.pth')).items()})
            self.netG_B.load_state_dict({k.replace('module.',''):v for k,v in torch.load(os.path.join(opt.pretrained_path, 'latest_net_G_B.pth')).items()})
            #self.netG_A.load_state_dict(torch.load(os.path.join(opt.pretrained_path, 'latest_net_G_A.pth')))
            #self.netG_B.load_state_dict(torch.load(os.path.join(opt.pretrained_path, 'latest_net_G_B.pth')))
            print('loading pretrained WCyclegan on G successfully!')'''
        #### Types of Cycle Gan
        self.is_sn_gan = opt.sn_gan
        if self.isTrain:
            self.is_wgan = opt.wgan
            self.with_gp = opt.with_gp
            self.lambda_gp = opt.lambda_gp
            if self.is_sn_gan:
                print('Training WITH Spectral Normalization')
            else:
                print('Training WITHOUT Spectral Normalization')
            if self.is_wgan:
                print('Training WITH Wassertein Loss')
            else:
                print('Training WITHOUT Wassertein Loss')
            if self.with_gp:
                print('Training WITH Gradient Penalty')
            else:
                print('Training WITHOUT Gradient Penalty')
        
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.is_sn_gan:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, "spectral", use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, "spectral", use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                self.gpu_ids) #default='instance' -- opt.norm
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain,
                                                self.gpu_ids)
        '''if(opt.use_pre_gan):
            self.netD_A.load_state_dict({k.replace('module.',''):v for k,v in torch.load(os.path.join(opt.pretrained_path, 'latest_net_D_A.pth')).items()})
            self.netD_B.load_state_dict({k.replace('module.',''):v for k,v in torch.load(os.path.join(opt.pretrained_path, 'latest_net_D_B.pth')).items()})
            #self.netD_A.load_state_dict(torch.load(os.path.join(opt.pretrained_path, 'latest_net_D_A.pth')))
            #self.netD_B.load_state_dict(torch.load(os.path.join(opt.pretrained_path, 'latest_net_D_B.pth')))
            print('loading pretrained WCyclegan on D successfully!')'''
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.gradient_penalty = networks.GradPenalty(use_cuda=True if opt.gpu_ids!=-1 else False)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.RMSprop(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr)
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_labels = input['A_labels' if AtoB else 'B_labels']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))
        
        return self.real_A, self.real_B, self.fake_A, self.fake_B, self.rec_A, self.rec_B

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)

        if self.is_wgan:############ MODIF
            loss_D_real = -pred_real.mean()
        #    loss_D_real.backward(-1,retain_graph=True)
        else:
            loss_D_real = self.criterionGAN(pred_real, True)
        #    loss_D_real.backward(retain_graph=True)
        # Fake
        pred_fake = netD(fake.detach())
        if self.is_wgan:########### MODIF
            loss_D_fake = pred_fake.mean()
        else:
            loss_D_fake = self.criterionGAN(pred_fake, False)
        #loss_D_fake.backward(retain_graph=True)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5

        if self.with_gp: ########### MODIF
            eps = torch.autograd.Variable(torch.rand(1), requires_grad=True)
            eps = eps.expand(real.size())
            eps = eps.cuda()
            x_tilde = eps * real + (1 - eps) * fake.detach()
            x_tilde = x_tilde.cuda()
            pred_tilde = netD(x_tilde)
            gradients = torch.autograd.grad(outputs=pred_tilde, inputs=x_tilde,
                             grad_outputs=torch.ones(pred_tilde.size()).cuda(),
                             create_graph=True, retain_graph=True, only_inputs=True)[0]
            loss_D = loss_D + 10*((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        return fake_B

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
        return fake_A

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.netG_A(self.real_B)
            if(self.idt_A.size(3) != self.real_B.size(3)):
              self.idt_A = F.interpolate(self.idt_A, size = [self.real_B.size(2), self.real_B.size(3)], mode='bilinear')
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.netG_B(self.real_A)
            if(self.idt_B.size(3) != self.real_A.size(3)):
              self.idt_B = F.interpolate(self.idt_B, size = [self.real_A.size(2), self.real_A.size(3)], mode='bilinear')
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        if self.is_wgan:
            self.loss_G_A = -self.netD_A(self.fake_B).mean()
            # GAN loss D_B(G_B(B))
            self.loss_G_B = -self.netD_B(self.fake_A).mean()
        else:
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            # GAN loss D_B(G_B(B))
            self.loss_G_B = 2 * self.criterionGAN(self.netD_B(self.fake_A), True)
        if(self.rec_A.size(3) != self.real_A.size(3)):
              self.rec_A = F.interpolate(self.rec_A, size = [self.real_A.size(2), self.real_A.size(3)], mode='bilinear')
        if(self.rec_B.size(3) != self.real_B.size(3)):
              self.rec_B = F.interpolate(self.rec_B, size = [self.real_B.size(2), self.real_B.size(3)], mode='bilinear')
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        self.loss_G.backward()
        
        return self.idt_A, self.idt_B

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # compute fake images and reconstruction images.
        self.real_A, self.real_B, self.fake_A, self.fake_B, self.rec_A, self.rec_B = self.forward()      
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.idt_A, self.idt_B = self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        fake_B = self.backward_D_A()      # calculate gradients for D_A
        fake_A = self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        return self.real_A, self.real_B, self.fake_A, self.rec_A, self.idt_B