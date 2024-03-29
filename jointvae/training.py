import imageio
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
import math
import copy
import matplotlib.pyplot as plt
from scipy import misc
import os
import numpy as np
import correlations
EPS = 1e-12


class Trainer():
    def __init__(self, model, optimizer, label_size, cont_capacity=None,
                 disc_capacity=None, print_loss_every=50, record_loss_every=5,
                 use_cuda=True, lambda_d=None, lambda_od=None, L_Lambda=None, lambda_dis=None):
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
        self.L_Lambda = L_Lambda
        self.best_val_loss = 1e10
        self.best_model = model
        if self.model.img_size[1:] == (64, 64):
            self.recon_coeff = 0.5
        else:
            self.recon_coeff = 8
        self.label_size = label_size
        self.epoch = 0

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
            for i in range(len(self.model.ladder_dim)):
                self.losses['kl_loss_cont_' + str(i)] = []

        # if self.model.is_discrete:
        #     self.losses['kl_loss_disc'] = []
        #     # For every discrete latent variable
        #     for i in range(len(self.model.latent_spec['disc'])):
        #         self.losses['kl_loss_disc_' + str(i)] = []

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
            valid_loss, mean_epoch_loss, recon_error = self._train_epoch(train_loader, valid_loader)
            print('Epoch: {} Average loss: {:.2f} Valid loss: {}\tRecon Error:{:.3f}'.format(epoch + 1,
                                                                                             mean_epoch_loss,
                                                                                             valid_loss,
                                                                                             recon_error))  # self.losses['recon_loss'][-1]
            self.epoch += 1

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
        self.model = self.best_model

    def _train_epoch(self, train_loader, valid_loader):
        """
        Trains the model for one epoch.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
        """
        self.train_loader = train_loader
        self.model.train()
        epoch_loss = 0.
        val_loss = 0.
        total_l0 = 0.
        print_every_loss = 0.  # Keeps track of loss to print every
        batch_count = 0
        if len(train_loader.dataset) <= 100000:
            val_loss, recon_error = self.evaluate(valid_loader, self.epoch)
            print(val_loss)
            for batch_idx, (data, label) in enumerate(list(train_loader)):
                data = data.cuda().to(dtype=torch.float32)
                batch_count += 1
                iter_loss, l0_loss = self._train_iteration(data)
                epoch_loss += iter_loss
                if batch_idx % self.print_loss_every == 0:
                    print('{}/{}\tLoss: {:.3f}\tL0 Loss: {:.3f}'.format(batch_idx * len(data),
                                                                        len(train_loader.dataset),
                                                                        iter_loss,
                                                                        l0_loss))
        else:
            for batch_idx, (data, label) in enumerate(list(train_loader)):
                data = data.cuda().to(dtype=torch.float32)
                batch_count += 1
                iter_loss, l0_loss = self._train_iteration(data)
                epoch_loss += iter_loss
                if batch_idx % self.print_loss_every == 0:
                    print('{}/{}\tLoss: {:.3f}\tL0 Loss: {:.3f}'.format(batch_idx * len(data),
                                                                        len(train_loader.dataset),
                                                                        iter_loss,
                                                                        l0_loss))
                if batch_idx % 1000 == 0:
                    val_loss, recon_error = self.evaluate(valid_loader, self.epoch)
                    print(val_loss)

            mean_epoch_loss = epoch_loss/batch_count

        if val_loss < self.best_val_loss:
            self.best_model = copy.deepcopy(self.model)
            self.best_val_loss = val_loss
        # Return mean epoch loss
        return val_loss, mean_epoch_loss, recon_error

    def evaluate(self, valid_loader, epoch):
        """
        Evaluate the model on valid set
        :param data:
        :return: valid loss
        """
        #self.model.eval()

        batch_num = 0.
        epoch_loss = 0.
        recon_error = 0.
        reg_loss = 0.
        gap, max_score = 0., 0.
        matrix = torch.zeros([sum(self.model.ladder_dim), 5]).float().cuda()
        #digit_matrix = torch.zeros([sum(self.model.ladder_dim), 10]).float().cuda()

        for batch_idx, (data, label) in enumerate(list(valid_loader)):
            data = data.cuda().to(dtype=torch.float32)
            # recon_batch, latent_dist, mask, l0_reg = self.model(torch.unsqueeze(data,1).to(dtype=torch.float32))
            with torch.no_grad():
                recon_batch, latent_list, mask, l0_reg = self.model(data)
            loss, recon, kl_loss = self._loss_function(data, recon_batch, latent_list, mask, eval=True)  # + l0_reg
            # gap_, max_, matrix_, d_matrix = self.compute_score(latent_list, label)
            gap_, max_, matrix_ = self.compute_score(latent_list, label)
            gap += gap_
            max_score += max_
            matrix += matrix_.transpose(0, 1)
            #digit_matrix += d_matrix.transpose(0,1)
            iter_loss = loss.item()
            epoch_loss += iter_loss
            recon_error += recon
            reg_loss += kl_loss
            batch_num += 1.
        reg_loss = reg_loss/ batch_num
        mean_loss = epoch_loss / batch_num
        # valid_loss = self.model.num_pixels * mean_loss
        recon_error = recon_error / batch_num
        max_score = max_score/batch_num
        print(torch.mean(gap/batch_num))
        print(max_score)
        if len(self.train_loader.dataset) <= 100000:
            self.plot_partial_correlation((matrix / batch_num).transpose(0, 1), hrule=None,
                                      mig=(gap / batch_num).reshape(-1, 1), max_score=max_score, ax=None,
                                      epoch=epoch)
        else:
            self.plot_partial_correlation((matrix / batch_num).transpose(0, 1), hrule=None,
                                          mig=(gap / batch_num).reshape(-1, 1), max_score=max_score, ax=None,
                                          epoch=self.num_steps)
        #self.plot_partial_correlation(((digit_matrix/batch_num).transpose(0, 1)), epoch=epoch)
        print('Regularization Loss: {:.3f},Recon Loss: {:.3f}\t Recon Error: {:.3f}'.format(reg_loss, recon_error*self.model.num_pixels, recon_error))
        return mean_loss, recon_error

    def compute_score(self, latent_list, label):
        #concat latent, which is a list of multiple tensor matrixs
        latent_mean = torch.cat([latent_list[key][0] for key in latent_list.keys()], dim=1)
        label = label.float()
        # corr_matrix = correlations.partial_correlation_matrix(latent_mean, label[:, :7])
        # digit_matrix = correlations.partial_correlation_matrix(latent_mean, label[:, 7:])
        # digit_matrix = torch.from_numpy(digit_matrix).float().cuda()
        corr_matrix = correlations.partial_correlation_matrix(latent_mean, label)
        corr_matrix = torch.from_numpy(corr_matrix).float().cuda()
        values, indices = torch.topk(abs(corr_matrix), 2, dim=0)
        gap_score = (values[0, :] - values[1, :])
        max_score = torch.mean(torch.max(abs(corr_matrix), dim=0)[0]).float()
        return gap_score, max_score, corr_matrix  #, digit_matrix

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
            data = data.cuda()

        self.optimizer.zero_grad()
        recon_batch, latent_dist, mask, l0_reg = self.model(data)
        loss = self._loss_function(data, recon_batch, latent_dist, mask) + l0_reg

        loss.backward()
        self.optimizer.step()

        train_loss = loss.item()
        return train_loss, l0_reg

    def _loss_function(self, data, recon_data, latent_list, mask, eval=False):
        """
        Calculates loss for a batch of data.

        Parameters
        ----------
        data : torch.Tensor
            Input data (e.g. batch of images). Should have shape (N, C, H, W)

        recon_data : torch.Tensor
            Reconstructed data. Should have shape (N, C, H, W)

        latent_list : dict
            Dict with keys 'cont' or 'disc' or both containing the parameters
            of the latent distributions as values.
        """
        # Reconstruction loss is pixel wise cross-entropy

        # recon_loss = F.binary_cross_entropy(recon_data.view(-1, self.model.num_pixels),
        #                                     data.view(-1, self.model.num_pixels))
        rec_loss = torch.nn.MSELoss(reduction='mean')
        recon_error = rec_loss(recon_data.view(-1, self.model.num_pixels),
                              data.view(-1, self.model.num_pixels))
        # print("recon ERROR:",recon_error)
        # recon_loss = torch.nn.L1Loss(reduction='mean')
        # recon_loss = recon_loss(recon_data.view(-1, self.model.num_pixels),
        #                        data.view(-1, self.model.num_pixels))
        recon_loss = torch.mean(torch.abs(recon_data.view(-1, self.model.num_pixels)-
                              data.view(-1, self.model.num_pixels)))
        #self.recon_mean = tf.reduce_mean(tf.abs(self.toutput - self.target_placeholder))
        #print("L2:",recon_error,"L1:", recon_loss)
        #recon_loss = F.smooth_l1_loss(recon_data.view(-1, self.model.num_pixels),
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
        for key in latent_list.keys():
            mean, std = latent_list[key]
            kl_cont_loss += self._kl_normal_loss(mean, std ** 2, mask)

        if self.model.is_continuous:
            # Calculate KL divergence
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
            mean = torch.cat([latent_list[key][0] for key in latent_list.keys()], dim=1)
            general_DIP = self.DIP_loss(mean)
            for key in latent_list.keys():
                general_DIP += self.DIP_loss(latent_list[key][0])

        # if self.model.is_discrete:
        #     # Calculate KL divergence
        #     kl_disc_loss = self._kl_multiple_discrete_loss(latent_list['disc'])
        #     # Linearly increase capacity of discrete channels
        #     disc_min, disc_max, disc_num_iters, disc_gamma = \
        #         self.disc_capacity
        #     # Increase discrete capacity without exceeding disc_max or theoretical
        #     # maximum (i.e. sum of log of dimension of each discrete variable)
        #     disc_cap_current = (disc_max - disc_min) * self.num_steps / float(disc_num_iters) + disc_min
        #     disc_cap_current = min(disc_cap_current, disc_max)
        #     # Require float conversion here to not end up with numpy float
        #     disc_theoretical_max = sum([float(np.log(disc_dim)) for disc_dim in self.model.latent_spec['disc']])
        #     disc_cap_current = min(disc_cap_current, disc_theoretical_max)
        #     # Calculate discrete capacity loss
        #     disc_capacity_loss = disc_gamma * disc_cap_current

            ###DIP-VAE i
            #DIP_disc = self.lambda_dis * self.jsd(latent_list['disc'])  #### the more the better, diversity

        # Calculate total kl value to record it
        kl_loss = kl_cont_loss #+ kl_disc_loss
        DIP_loss = general_DIP#DIP_cont #- DIP_disc
        # Calculate total loss
        # total_loss = recon_loss + cont_capacity_loss + disc_capacity_loss + DIP_loss +kl_loss
        total_loss = recon_loss + DIP_loss + 3*kl_loss
        #print("recon_loss:", recon_loss ,"DIP_loss:", DIP_loss , "kl_loss:", kl_loss)
        # print("rec",recon_loss,"cont_capa",cont_capacity_loss,"disc_capa",disc_capacity_loss,"DIP_cont",DIP_cont,"DIP_disc",DIP_disc)
        # print("rec",recon_loss,"DIP_cont",DIP_cont,"kl_cont",kl_cont_loss,"kl_disc",kl_disc_loss)
        # Record losses
        if self.model.training and self.num_steps % self.record_loss_every == 1:
            self.losses['recon_loss'].append(recon_loss.item())
            self.losses['kl_loss'].append(kl_loss.item())
            self.losses['DIP_loss'].append(DIP_loss.item())
            self.losses['loss'].append(total_loss.item())
            # print(self.losses,"mask",mask)
            # self.losses['DIP_cont'].append(DIP_cont.item())
            # self.losses['DIP_disc'].append(DIP_disc.item())

        # To avoid large losses normalise by number of pixels
        if not eval:
            return total_loss
        else:
            return total_loss, recon_error, kl_loss

    def DIP_loss(self, mean):
        exp_mu = torch.mean(mean, dim=0)  #####mean through batch
        # expectation of mu mu.tranpose
        mu_expand1 = mean.unsqueeze(1)  #####(batch_size, 1, number of mean of latent variables)
        mu_expand2 = mean.unsqueeze(2)  #####(batch_size, number of mean of latent variables, 1) ignore batch_size, only transpose the means
        exp_mu_mu_t = torch.mean(mu_expand1 * mu_expand2, dim=0)
        # covariance of model mean
        cov = exp_mu_mu_t - exp_mu.unsqueeze(0) * exp_mu.unsqueeze(1)  ##1, mean* mean, 1
        diag_part = torch.diagonal(cov, offset=0, dim1=-2, dim2=-1)
        off_diag_part = cov - torch.diag(diag_part)

        regulariser_od = self.lambda_od * torch.sum(off_diag_part ** 2)
        regulariser_d = self.lambda_d * torch.sum((diag_part - 1) ** 2)

        DIP = regulariser_d + regulariser_od

        return DIP

    def _kl_normal_loss(self, mean, var, mask):
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
        # print("logvar:",torch.log(var)[0,:],"var",var[0,:])
        kl_values = -0.5 * (1 + torch.log(var) - mean.pow(2) - var)
        # Mean KL divergence across batch for each latent variable
        kl_means = torch.mean(kl_values, dim=0)
        kl_means = kl_means #* mask  # .squeeze()
        # KL loss is sum of mean KL of each latent variable
        kl_loss = torch.sum(kl_means)

        # Record losses
        # if self.model.training and self.num_steps % self.record_loss_every == 1:
        #     self.losses['kl_loss_cont'].append(kl_loss.item())
        #     for i in range(self.model.latent_spec['cont']):
        #         self.losses['kl_loss_cont_' + str(i)].append(kl_means[i].item())

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

        else:  # Total loss is sum of kl loss for each discrete latent
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
                temp_entropy += torch.sum(p * torch.log(p))
        return -temp_entropy

    def jsd(self, prob_dists):
        if len(prob_dists) >= 2:
            weight = 1 / len(prob_dists)  # all same weight
            js_left = [0] * len(prob_dists[0])
            js_right = 0
            for pd in prob_dists:
                # print("pd",pd)
                pd = torch.mean(pd, dim=0)
                for i in range(len(pd)):
                    js_left[i] += pd[i] * weight
                js_right += weight * self.entropy(pd)
            return self.entropy(js_left) - js_right
        else:
            return 0

    def get_best_model(self):

        return self.best_model

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
    def plot_reconstruction(self, epoch, test_image, noisy_image, reconstruction, num_plot=2):
        if test_image.shape[-1] == 1:   # Black background for mnist, white for color images
            canvas = np.zeros((num_plot*self.data_dims[0], 3*self.data_dims[1] + 20, self.data_dims[2]))
        else:
            canvas = np.ones((num_plot*self.data_dims[0], 3*self.data_dims[1] + 20, self.data_dims[2]))
        for img_index in range(num_plot):
            # canvas[img_index*self.data_dims[0]:(img_index+1)*self.data_dims[0], 0:self.data_dims[1]] = \
            #     self.dataset.display(test_image[img_index, :, :])
            canvas[img_index*self.data_dims[0]:(img_index+1)*self.data_dims[0], self.data_dims[1]+10:self.data_dims[1]*2+10] = \
                self.dataset.display(noisy_image[img_index, :, :])
            canvas[img_index*self.data_dims[0]:(img_index+1)*self.data_dims[0], self.data_dims[1]*2+20:] = \
                self.dataset.display(reconstruction[img_index, :, :])

        img_folder = r"C:\Users\Cooper\FYP\FYP-of-Disentanglement-master\FYP-of-Disentanglement-master\Visualization\Reconstruction" + self.model.__class__.__name__
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)

        if canvas.shape[-1] == 1:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas[:, :, 0])
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % epoch), canvas[:, :, 0])
        else:
            misc.imsave(os.path.join(img_folder, 'current.png'), canvas)
            misc.imsave(os.path.join(img_folder, 'epoch%d.png' % epoch), canvas)

        if self.args.use_gui:
            if self.fig is None:
                self.fig, self.ax = plt.subplots()
                self.fig.suptitle(r"Reconstruction of " + self.model.__class__.__name__)
            self.ax.cla()
            if canvas.shape[-1] == 1:
                self.ax.imshow(canvas[:, :, 0], cmap=plt.get_cmap('Greys'))
            else:
                self.ax.imshow(canvas)
            plt.draw()
            plt.pause(1)

    def plot_hinton(self, corr, vmax=1., colors=('red', 'blue'), bgcolor='white', cmap='RdBu', ax=None, epoch=0, MIG=None):
        """Draw Hinton diagram for visualizing a weight matrix."""
        ##normalize corr to [-1,1]
        #corr = corr/(tf.max(corr)-tf.min(corr))*2-1
        ax = None
        corr = (corr-corr.min())/(corr.max()-corr.min())*2-1
        ax = ax if ax is not None else plt.gca()
        if isinstance(cmap, str):
            cmap = plt.cm.get_cmap(cmap)
        if cmap is not None:
            bgcolor = cmap(.5)
        ax.set_facecolor(bgcolor)
        ax.set_aspect('equal', 'box')
        for (y, x), w in np.ndenumerate(corr):
            if cmap is None:
                color = colors[int(w <= 0)]
            else:
                color = cmap(.5 * (w / vmax + 1.))
            size = np.sqrt(np.abs(w) / vmax)
            # patch = plt.Rectangle([x - size / 2, y - size / 2], size, size,
            #                       facecolor=color, edgecolor=color, lw=0)
            patch = plt.Circle([x, y], size / 2, facecolor=color, edgecolor=color, lw=0)
            ax.add_patch(patch)
        ax.set_xlim(-.5, corr.shape[1] - .5)
        ax.set_ylim(corr.shape[0] - .5, -.5)
        # fig = ax.get_figure()
        # if not MIG:
        #     fig.savefig("{}%s%s%s/%s.png".format('C:/Users/Cooper/FYP/Variational-Ladder-Autoencoder/visualization/corr/') %
        #                 (self.network.ladder0_dim,self.network.ladder1_dim,self.network.ladder2_dim,epoch))
        # else:
        #     fig.savefig("{}%s%s%s/%s.png".format('C:/Users/Cooper/FYP/Variational-Ladder-Autoencoder/visualization/MIG/') %
        #                 (self.network.ladder0_dim,self.network.ladder1_dim,self.network.ladder2_dim,epoch))
        # plt.clf()
        # plt.cla()
        # plt.close()

    def plot_partial_correlation(self, pcorr, hrule=None, mig=None, max_score=None, ax=None, epoch=0):
        ax = plt.gca() if ax is None else ax
        self.plot_hinton(pcorr.cpu().numpy(), cmap='RdBu', ax=ax)
        if len(self.model.ladder_dim) == 4:
            ladder0_dim, ladder1_dim, ladder2_dim, ladder3_dim = self.model.ladder_dim[0], self.model.ladder_dim[1],\
                                                                 self.model.ladder_dim[2], self.model.ladder_dim[3]
        else:
            ladder0_dim, ladder1_dim, ladder2_dim= self.model.ladder_dim[0], self.model.ladder_dim[1], self.model.ladder_dim[2]
        div_kwargs = dict(c='.8', lw=1, zorder=-1)
        if ladder0_dim > 0 and (ladder1_dim > 0 or ladder2_dim > 0):  # Cat vs. cont/bin separator
            ax.axvline(ladder0_dim - .5, **div_kwargs)
        if ladder1_dim > 0 and ladder2_dim > 0:  # Cont vs. bin separator
            ax.axvline(ladder0_dim + ladder1_dim - .5, **div_kwargs)
        if hrule is not None:  # Horizontal separator
            ax.axhline(hrule - .5, **div_kwargs)

        # If no categorical, start numbering continuous/binary from 1
        idx_offset = int(ladder0_dim > 0)

        ax.set_yticks(np.arange(pcorr.shape[0]))
        if mig is not None:
            # ax.set_yticklabels(['Area', 'Length', 'Thickness', 'Slant', 'Width', 'Height', 'Digits'])
            ax.set_yticklabels(['Shape', 'Scale', 'Orientation', 'PositionX', 'PositionY'])
        else:
            ax.set_yticklabels(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        ax.set_xticks(np.arange(pcorr.shape[1]))
        if len(self.model.ladder_dim) == 4:
            ax.set_xticklabels([f"$c_{{{i + idx_offset}}}^{{}}$" for i in range(ladder0_dim + ladder1_dim + \
                                                                            ladder2_dim + ladder3_dim)])
            # [f"$c_{{1}}^{{({i + 1})}}$" for i in range(ladder0_dim)] +
            ax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)

            def add_xlabel(x, s):
                ax.text(x, pcorr.shape[0] - .5 + .2, s, ha='center', va='top', size='small')

            cat_pos = (ladder0_dim - 1.) / 2.
            cont_pos = ladder0_dim + (ladder1_dim - 1.) / 2.
            bin_pos = ladder0_dim + ladder1_dim + (ladder2_dim - 1.) / 2.
            last_pos = ladder0_dim + ladder1_dim + ladder2_dim + (ladder3_dim - 1.) / 2.
        else:
            ax.set_xticklabels([f"$c_{{{i + idx_offset}}}^{{}}$" for i in range(ladder0_dim + ladder1_dim + \
                                                                                ladder2_dim)])
            # [f"$c_{{1}}^{{({i + 1})}}$" for i in range(ladder0_dim)] +
            ax.tick_params(bottom=False, labelbottom=False, top=True, labeltop=True)

            def add_xlabel(x, s):
                ax.text(x, pcorr.shape[0] - .5 + .2, s, ha='center', va='top', size='small')

            cat_pos = (ladder0_dim - 1.) / 2.
            cont_pos = ladder0_dim + (ladder1_dim - 1.) / 2.
            bin_pos = ladder0_dim + ladder1_dim + (ladder2_dim - 1.) / 2.

        if ladder1_dim > 0 and ladder2_dim > 0:  # Adjust positions to avoid overlap
            cont_pos -= .1
            bin_pos += .1

        if ladder0_dim > 0:
            add_xlabel(cat_pos, 'Lowest-level')
        if ladder1_dim > 0:
            add_xlabel(cont_pos, '2nd-level')
        if ladder2_dim > 0:
            add_xlabel(bin_pos, '3rd-level')
        if len(self.model.ladder_dim) == 4:
            if ladder3_dim > 0:
                add_xlabel(last_pos, '4th-level')

        if mig is not None:
            ax.text(pcorr.shape[1] + .4, -1, "MIG", ha='center', va='center', size='medium')
            for i in range(pcorr.shape[0]):
                ax.text(pcorr.shape[1] + .4, i, '{:.4f}'.format(mig[i][0]), ha='center', va='center', size='small')
            ax.text(pcorr.shape[1] + .4,7 , '{:.4f}'.format(mig.mean()), ha='center', va='center', size='medium')

        if max_score is not None:
            ax.text(pcorr.shape[1] + 1.8, -1, "MAX", ha='center', va='center', size='medium')
            ax.text(pcorr.shape[1] + 1.8, 3, '{:.4f}'.format(max_score), ha='center', va='center', size='small')

        fig = ax.get_figure()
        img_folder ="C:/Users/Cooper/FYP/FYP-of-Disentanglement-master/FYP-of-Disentanglement-master/visualization/"+"%s%s%s" % \
                    (self.model.ladder_dim[0], self.model.ladder_dim[1], self.model.ladder_dim[2])
        if not os.path.isdir(img_folder):
            os.makedirs(img_folder)
        if len(self.model.ladder_dim) == 4:
            if mig is not None:
                fig.savefig("{}/DIP%s%s%s%s/%s.png".format('C:/Users/Cooper/FYP/FYP-of-Disentanglement-master/FYP-of-Disentanglement-master/visualization/dsprites') %
                        (self.model.ladder_dim[0], self.model.ladder_dim[1], self.model.ladder_dim[2], self.model.ladder_dim[3], epoch), bbox_inches="tight")
        else:
            if mig is not None:
                fig.savefig("{}/DIP%s%s%s/%s.png".format(
                    'C:/Users/Cooper/FYP/FYP-of-Disentanglement-master/FYP-of-Disentanglement-master/visualization/dsprites') %
                            (self.model.ladder_dim[0], self.model.ladder_dim[1], self.model.ladder_dim[2], epoch),
                            bbox_inches="tight")
            else:
                fig.savefig("{}/DIP%s%s%s/Digits%s.png".format(
                    'C:/Users/Cooper/FYP/FYP-of-Disentanglement-master/FYP-of-Disentanglement-master/visualization/dsprites') %
                            (self.model.ladder_dim[0], self.model.ladder_dim[1], self.model.ladder_dim[2], epoch),
                            bbox_inches="tight")
        print("Plot Saved!!")
        plt.clf()
        plt.cla()
        plt.close()




