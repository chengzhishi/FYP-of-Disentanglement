import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
import math
import copy


EPS = 1e-12


class Trainer():
    def __init__(self, model, optimizer, cont_capacity=None,
                 disc_capacity=None, print_loss_every=50, record_loss_every=5,
                 use_cuda=True,lambda_d=None, lambda_od=None, lambda_dis=None):
        """
        Class to handle training of model.

        Parameters
        ----------
        model : jointvae.models.VAE instance

        optimizer : torch.optim.Optimizer instance

        cont_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_z).
            Parameters to control the capacity of the continuous latent
            channels. Cannot be None if model.is_continuous is True.

        disc_capacity : tuple (float, float, int, float) or None
            Tuple containing (min_capacity, max_capacity, num_iters, gamma_c).
            Parameters to control the capacity of the discrete latent channels.
            Cannot be None if model.is_discrete is True.

        print_loss_every : int
            Frequency with which loss is printed during training.

        record_loss_every : int
            Frequency with which loss is recorded during training.

        use_cuda : bool
            If True moves model and training to GPU.
        """
        self.model = model
        self.optimizer = optimizer
        self.cont_capacity = cont_capacity
        self.disc_capacity = disc_capacity
        self.print_loss_every = print_loss_every
        self.record_loss_every = record_loss_every
        self.use_cuda = use_cuda
        self.lambda_d = lambda_d
        self.lambda_od = lambda_od
        self.lambda_dis = lambda_dis
        self.best_val_loss = 1e10
        self.best_model = model


        if self.model.is_continuous and self.cont_capacity is None:
            raise RuntimeError("Model is continuous but cont_capacity not provided.")

        if self.model.is_discrete and self.disc_capacity is None:
            raise RuntimeError("Model is discrete but disc_capacity not provided.")

        if self.use_cuda:
            self.model.cuda()

        # Initialize attributes
        self.num_steps = 0
        self.batch_size = None
        self.losses = {'loss': [],
                       'recon_loss': [],
                       'kl_loss': [],
                       'DIP_loss': []}

        # Keep track of divergence values for each latent variable
        if self.model.is_continuous:
            self.losses['kl_loss_cont'] = []
            # For every dimension of continuous latent variables
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)] = []

        if self.model.is_discrete:
            self.losses['kl_loss_disc'] = []
            # For every discrete latent variable
            for i in range(len(self.model.latent_spec['disc'])):
                self.losses['kl_loss_disc_' + str(i)] = []

    def train(self, train_loader, valid_loader, epochs=10, save_training_gif=None):
        """
        Trains the model.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader

        epochs : int
            Number of epochs to train the model for.

        save_training_gif : None or tuple (string, Visualizer instance)
            If not None, will use visualizer object to create image of samples
            after every epoch and will save gif of these at location specified
            by string. Note that string should end with '.gif'.
        """
        if save_training_gif is not None:
            training_progress_images = []

        self.batch_size = train_loader.batch_size

        self.model.train()
        for epoch in range(epochs):
            valid_loss, mean_epoch_loss,recon_error = self._train_epoch(train_loader, valid_loader)
            print('Epoch: {} Average loss: {:.2f} Valid loss: {}\tRecon Error:{:.3f}'.format(epoch + 1,
                                                          self.batch_size * self.model.num_pixels * mean_epoch_loss, valid_loss,recon_error))#self.losses['recon_loss'][-1]


        #     if save_training_gif is not None:
        #         # Generate batch of images and convert to grid
        #         viz = save_training_gif[1]
        #         viz.save_images = False
        #         img_grid = viz.all_latent_traversals(size=10)
        #         # Convert to numpy and transpose axes to fit imageio convention
        #         # i.e. (width, height, channels)
        #         img_grid = np.transpose(img_grid.numpy(), (1, 2, 0))
        #         # Add image grid to training progress
        #         training_progress_images.append(img_grid)
        #
        # if save_training_gif is not None:
        #     imageio.mimsave(save_training_gif[0], training_progress_images,
        #                     fps=24)
        self.model=self.best_model
    def _train_epoch(self, train_loader, valid_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        epoch_loss = 0.
        val_loss = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
                               # self.print_loss_every

        for batch_idx, (data, label) in enumerate(list(train_loader)):
        #items = iter(train_loader)
        #for batch_idx in range(len(train_loader)):
            #(data, label) = next(items)
            iter_loss = self._train_iteration(data)
            epoch_loss += iter_loss
            print_every_loss += iter_loss
            # Print loss info every self.print_loss_every iteration
            if batch_idx % self.print_loss_every == 0:
                if batch_idx == 0:
                    mean_loss = print_every_loss
                else:
                    mean_loss = print_every_loss / self.print_loss_every
                print('{}/{}\tLoss: {:.3f}'.format(batch_idx * len(data),
                                                  len(train_loader.dataset),
                                                  self.model.num_pixels * mean_loss))
                print_every_loss = 0.
        val_loss,recon_error = self.evaluate(valid_loader)
        print(val_loss)
        if val_loss < self.best_val_loss:
            self.best_model = copy.deepcopy(self.model)
            self.best_val_loss = val_loss
        # Return mean epoch loss
        return val_loss, epoch_loss / len(train_loader.dataset),recon_error

    def evaluate(self,valid_loader):
        """
        Evaluate the model on valid set
        :param data:
        :return: valid loss
        """
        batch_num = 0.
        epoch_loss = 0.
        for batch_idx, (data, label) in enumerate(list(valid_loader)):
            data = torch.unsqueeze(data,1).cuda().to(dtype=torch.float32)
            #data=data.cuda()
            recon_batch, latent_dist = self.model(data)
            loss,recon = self._loss_function(data, recon_batch, latent_dist,eval=True)# + l0_reg
            iter_loss = loss.item()
            epoch_loss += iter_loss
            batch_num += 1
        mean_loss = epoch_loss/batch_num
        valid_loss = self.model.num_pixels * mean_loss
        recon_error = recon/self.model.num_pixels
        print('Valid Loss: {:.3f}, Recon Error: {:.3f}'.format(valid_loss,recon_error ))
        
        return valid_loss,recon_error


    def _train_iteration(self, data):
        """
        Trains the model for one iteration on a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            A batch of data. Shape (N, C, H, W)
        """
        self.num_steps += 1
        if self.use_cuda:
            data = torch.unsqueeze(data,1).cuda().to(dtype=torch.float32)
            #data = data.cuda()

        self.optimizer.zero_grad()
        recon_batch, latent_dist  = self.model(data)
        loss = self._loss_function(data, recon_batch, latent_dist)


        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss

    def _loss_function(self, data, recon_data, latent_dist, eval=False):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy

        # recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels),
        #                                     data.view(-1, self.model.num_pixels))
        rec_loss =torch.nn.MSELoss(reduction='mean')
        recon_loss = rec_loss(recon_data.view(-1, self.model.num_pixels),
                                            data.view(-1, self.model.num_pixels))
        # recon_loss = F.smooth_l1_loss(recon_data.view(-1, self.model.num_pixels),
        #                                     data.view(-1, self.model.num_pixels))
        # F.binary_cross_entropy takes mean over pixels, so unnormalise this
        recon_loss *= self.model.num_pixels

        # Calculate KL divergences
        kl_cont_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        kl_disc_loss = 0  # Used to compute capacity loss (but not a loss in itself)
        cont_capacity_loss = 0
        disc_capacity_loss = 0
        DIP_cont = 0
        DIP_disc = 0

        if self.model.is_continuous:
            # Calculate KL divergence
            mean, logvar = latent_dist['cont']
            kl_cont_loss = self._kl_normal_loss(mean, logvar)
            # Linearly increase capacity of continuous channels
            cont_min, cont_max, cont_num_iters, cont_gamma = \
                self.cont_capacity
            # Increase continuous capacity without exceeding cont_max
            cont_cap_current = (cont_max - cont_min) * self.num_steps / float(cont_num_iters) + cont_min
            cont_cap_current = min(cont_cap_current, cont_max)
            # Calculate continuous capacity loss
            # cont_capacity_loss = cont_gamma * torch.abs(cont_cap_current - kl_cont_loss)
            cont_capacity_loss = cont_gamma * kl_cont_loss



            ###DIP-VAE i
            # expectation of mu (mean of distributions)
            exp_mu = torch.mean(mean, dim=0)  #####mean through batch

            # expectation of mu mu.tranpose
            mu_expand1 = mean.unsqueeze(1)  #####(batch_size, 1, number of mean of latent variables)
            mu_expand2 = mean.unsqueeze(2)  #####(batch_size, number of mean of latent variables, 1) ignore batch_size, only transpose the means
            exp_mu_mu_t = torch.mean(mu_expand1 * mu_expand2, dim=0)

            # covariance of model mean
            cov = exp_mu_mu_t - exp_mu.unsqueeze(0) * exp_mu.unsqueeze(1)  ##1, mean* mean, 1

            diag_part = torch.diagonal(cov, offset = 0, dim1 = -2, dim2 = -1)
            off_diag_part = cov - torch.diag(diag_part)

            regulariser_od = self.lambda_od * torch.sum(off_diag_part ** 2)
            regulariser_d = self.lambda_d * torch.sum((diag_part - 1) ** 2)

            DIP_cont = regulariser_d + regulariser_od


        if self.model.is_discrete:
            # Calculate KL divergence
            kl_disc_loss = self._kl_multiple_discrete_loss(latent_dist['disc'])
            # Linearly increase capacity of discrete channels
            disc_min, disc_max, disc_num_iters, disc_gamma = \
                self.disc_capacity
            # Increase discrete capacity without exceeding disc_max or theoretical
            # maximum (i.e. sum of log of dimension of each discrete variable)
            disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
            disc_cap_current = min(disc_cap_current, disc_max)
            # Require float conversion here to not end up with numpy float
            disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_spec['disc']])
            disc_cap_current = min(disc_cap_current, disc_theoretical_max)
            # Calculate discrete capacity loss
            disc_capacity_loss = disc_gamma * disc_cap_current

            ###DIP-VAE i
            DIP_disc = self.lambda_dis*self.jsd(latent_dist['disc'])#### the more the better, diversity

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss + kl_disc_loss
        DIP_loss = DIP_cont - DIP_disc
        # Calculate total loss
        total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss + DIP_loss
        #print("rec",recon_loss,"cont_capa",cont_capacity_loss,"disc_capa",disc_capacity_loss,"DIP_cont",DIP_cont,"DIP_disc",DIP_disc)
        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['DIP_loss'].append(DIP_loss.item())
            self.losses['loss'].append(total_loss.item())

        # To avoid large losses normalise by number of pixels
        if not eval:
            return total_loss / self.model.num_pixels
        else:
        
            return total_loss / self.model.num_pixels,recon_loss
      

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution with
        diagonal covariance and a unit normal distribution.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        # Calculate KL divergence
        #print("logvar:",torch.log(var)[0,:],"var",var[0,:])
        kl_values = -0.5 * (1 + logvar - mean.pow(2) - torch.exp(logvar))
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        kl_means = kl_means 
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['kl_loss_cont'].append(kl_loss.item())
            for i in range(self.model.latent_spec['cont']):
                self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

        return kl_loss
    # def _disc_chi2_loss(self, alphas):
    #     i = 0
    #
    #     latent_mean['disc'] = []
    #     for alpha in alphas:
    #
    #         latent_mean['disc'].append(torch.mean(alpha, dim=0))### mean through batch
    #
    #     return chi_loss
    def _kl_multiple_discrete_loss(self, alphas):
        """
        Calculates the KL divergence between a set of categorical distributions
        and a set of uniform categorical distributions.

        Parameters
        ----------
        alphas : list
            List of the alpha parameters of a categorical (or gumbel-softmax)
            distribution. For example, if the categorical atent distribution of
            the model has dimensions [2, 5, 10] then alphas will contain 3
            torch.Tensor instances with the parameters for each of
            the distributions. Each of these will have shape (N, D).
        """
        # Calculate kl losses for each discrete latent
        kl_losses = [self._kl_discrete_loss(alpha) for alpha in alphas]
        if not alphas:  ##check empty
            kl_loss = torch.zeros(1).cuda()

        else:# Total loss is sum of kl loss for each discrete latent
            kl_loss = torch.sum(torch.cat(kl_losses))

            # Record losses
            if self.model.training and self.num_steps % self.record_loss_every == 1:
                self.losses['kl_loss_disc'].append(kl_loss.item())
                for i in range(len(alphas)):
                    self.losses['kl_loss_disc_' + str(i)].append(kl_losses[i].item())

        return kl_loss

    def _kl_discrete_loss(self, alpha):
        """
        Calculates the KL divergence between a categorical distribution and a
        uniform categorical distribution.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the categorical or gumbel-softmax distribution.
            Shape (N, D)
        """
        disc_dim = int(alpha.size()[-1])
        log_dim = torch.Tensor([np.log(disc_dim)])
        if self.use_cuda:
            log_dim = log_dim.cuda()
        # Calculate negative entropy of each row
        neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
        # Take mean of negative entropy across batch
        mean_neg_entropy = torch.mean(neg_entropy, dim=0)
        # KL loss of alpha with uniform categorical variable
        kl_loss = log_dim + mean_neg_entropy
        return kl_loss

    def entropy(self, prob_dist):
        temp_entropy = 0
        for p in prob_dist:
            if p != 0:
                temp_entropy += torch.sum(p*torch.log(p))
        return -temp_entropy
    
    
    def jsd(self, prob_dists):
        if len(prob_dists) >= 2: 
            weight = 1 / len(prob_dists)  # all same weight
            js_left = [0]*len(prob_dists[0])
            js_right = 0
            for pd in prob_dists:
                #print("pd",pd)
                pd = torch.mean(pd, dim=0)
                for i in range(len(pd)):
                    js_left[i] += pd[i] * weight
                js_right += weight * self.entropy(pd)
            return self.entropy(js_left) - js_right
        else:
            return 0

    def best_model(self):

        return self.best_model()

    def get_losses(self):

        return self.losses
    """
    
    def jsd(self, prob_dists):
        
        if len(prob_dists) >= 2: 
            weight = 1 / len(prob_dists)  # all same weight

            js_left = [0]*len(prob_dists)
            js_right = 0
            for pd in prob_dists:
                #print("pd",pd)
                pd = torch.mean(pd, dim=0)
                print("pd",pd)
                for i in range(len(prob_dists)):
                    js_left[i] += pd[i] * weight
                js_right += weight * self.entropy(pd)
                print("left",self.entropy(js_left),"minus_right",-js_right)
            return self.entropy(js_left) - js_right
        else:
            return 0"""




