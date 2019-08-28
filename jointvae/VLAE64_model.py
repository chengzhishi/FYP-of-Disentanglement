import torch
from torch import nn, optim
from torch.nn import functional as F
from l0_layers import L0Dense, L0Pair
from torch.autograd import Variable
EPS = 1e-12


# define Flatten
class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

'''    def combine_latent(self, latent, ladder, method='gated_add'):
        if method is 'concat':
            return torch.cat([latent, ladder], dim=1)
        else:
            if method is 'add':
                return latent+ladder ####shape difference may cause error!!!
            elif method is 'gated_add':
                if self.gate == None:
                    self.gate = torch.nn.Parameter(data=torch.Tensor(latent.shape[1:]), requires_grad=True)
                    torch.nn.init.constant_(self.gate, 0.1)
                    return latent + self.gate*ladder
                else:
                    return latent + self.gate*ladder
                    '''
class Combine_latent(torch.nn.Module):
    def __init__(self):
        super(Combine_latent, self).__init__()
        self.gate = torch.nn.Parameter(torch.FloatTensor(1, 1024))
        nn.init.constant_(self.gate, 0.1)
    def forward(self,latent, ladder):
        return latent + self.gate*ladder


class Concat_latent(torch.nn.Module):
    def forward(self, latent, ladder):
        return torch.cat([latent, ladder], dim=1)


class Reshaper(torch.nn.Module):
    def __init__(self, img_size, channel):
        super(Reshaper, self).__init__()
        self.img_size = img_size
        self.channel = channel

    def forward(self, x):
        return x.reshape(-1, self.channel, self.img_size, self.img_size)


class MNISTReshaper(torch.nn.Module):
    def forward(self, x):
        return x.reshape(-1, 128, 7, 7)


class DSPRITESReshaper(torch.nn.Module):
    def forward(self, x):
        return x.reshape(-1, 128, 16, 16)


class VLAE(nn.Module):
    def __init__(self, img_size, ladder_dim, droprate_init=0.2, weight_decay=0.001, lambda_l0=0.1, temperature=.1, use_cuda=True):
        """
        Class which defines model and forward pass.

        Parameters
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).

        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.

        temperature : float
            Temperature for gumbel softmax distribution.

        use_cuda : bool
            If True moves model to GPU
        """
        super(VLAE, self).__init__()
        self.use_cuda = use_cuda

        # Parameters
        self.img_size = img_size
        # self.is_continuous = 'cont' in latent_spec
        # self.is_discrete = 'disc' in latent_spec
        self.is_continuous = True
        self.is_discrete = False
        self.ladder_dim = ladder_dim
        self.latent_spec = {'cont': sum(self.ladder_dim)}
        self.num_pixels = img_size[0] * img_size[1] * img_size[2]
        self.temperature = temperature
        self.hidden_dim = 256  # Hidden dimension of linear layer
        self.reshape = (64, 4, 4)  # Shape required to start transpose convs
        self.droprate_init = droprate_init
        self.weight_decay = weight_decay
        self.lambda_l0 = lambda_l0
        self.fs = [self.img_size[1], self.img_size[1] // 2, self.img_size[1] // 4, self.img_size[1] // 8,
                   self.img_size[1] // 16]
        self.ld = [3200, 65536, 16384, 8192]
        # Calculate dimensions of latent distribution
        self.latent_cont_dim = sum(self.ladder_dim)
        self.latent_disc_dim = 0
        self.num_disc_latents = 0
        if self.img_size[1:] == (64, 64):
            self.cs = [1, 64, 128, 256, 512, 1024]
        else:
            self.cs = [1, 64, 128, 1024]
        if self.is_continuous:
            self.latent_cont_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_disc_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        self.latent_dim = self.latent_cont_dim + self.latent_disc_dim

        inference0_layers = [
            nn.Conv2d(self.cs[0], self.cs[1], (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(self.cs[1]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.cs[1], self.cs[1], (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.cs[1]),
            nn.LeakyReLU(0.1)
        ]
        ladder0_layers = [
            nn.Conv2d(self.cs[0], self.cs[1], (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(self.cs[1]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.cs[1], self.cs[1], (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.cs[1]),
            nn.LeakyReLU(0.1),
            Flatten()####need 2 outputs, sigmoid
        ]
        inference1_layers = [
            nn.Conv2d(self.cs[1], self.cs[2], (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(self.cs[2]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.cs[2], self.cs[2], (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(self.cs[2]),
            nn.LeakyReLU(0.1),
            nn.Conv2d(self.cs[2], self.cs[3], (4, 4), stride=2, padding=1),
            nn.BatchNorm2d(self.cs[3]),
            nn.LeakyReLU(0.1)
        ]
        if self.img_size[1:] == (64, 64):
            ladder1_layers = [
                nn.Conv2d(self.cs[1], self.cs[2], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[2]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.cs[2], self.cs[2], (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(self.cs[2]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.cs[2], self.cs[3], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
                Flatten()
            ]
            inference2_layers = [
                nn.Conv2d(self.cs[3], self.cs[3], (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.cs[3], self.cs[4], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[4]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.cs[4], self.cs[4], (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(self.cs[4]),
                nn.LeakyReLU(0.1)
            ]
            ladder2_layers = [
                nn.Conv2d(self.cs[3], self.cs[3], (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.cs[3], self.cs[4], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[4]),
                nn.LeakyReLU(0.1),
                nn.Conv2d(self.cs[4], self.cs[4], (3, 3), stride=1, padding=1),
                nn.BatchNorm2d(self.cs[4]),
                nn.LeakyReLU(0.1),
                Flatten()
            ]
            inference3_layers = [
                Flatten(),
                nn.Linear(self.cs[4], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.LeakyReLU(0.1),
                nn.Linear(self.cs[5], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.LeakyReLU(0.1)
            ]
            ladder3_layers = [
                Flatten(),
                nn.Linear(self.ld[3], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.LeakyReLU(0.1),
                nn.Linear(self.cs[5], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.LeakyReLU(0.1),
            ]
            self.inference3 = nn.Sequential(*inference3_layers)
            self.inference2 = nn.Sequential(*inference2_layers)
            self.ladder3 = nn.Sequential(*ladder3_layers)
            ladder3_std_layers = [nn.Linear(self.cs[5], self.ladder_dim[3]), nn.Sigmoid()]
            self.ladder3_mean = nn.Linear(self.cs[5], self.ladder_dim[2])
            self.ladder3_std = nn.Sequential(*ladder3_std_layers)
        else:
            inference1_layers = [
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
                nn.Linear(self.cs[3], self.cs[3])
            ]
            ladder1_layers = [
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
            ]
            ladder2_layers = [
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm2d(self.cs[3]),
                nn.LeakyReLU(0.1),
            ]

        # Define encoder
        # self.img_to_features = nn.Sequential(*encoder_layers)
        self.inference0 = nn.Sequential(*inference0_layers)
        self.inference1 = nn.Sequential(*inference1_layers)
        self.ladder0 = nn.Sequential(*ladder0_layers)
        self.ladder1 = nn.Sequential(*ladder1_layers)
        self.ladder2 = nn.Sequential(*ladder2_layers)

        # Map encoded features into a hidden vector which will be used to
        # encode parameters of the latent distribution
        # self.features_to_hidden = nn.Sequential(
        #     nn.Linear(64 * 4 * 4, self.hidden_dim),
        #     nn.ReLU()
        # )

        # Encode parameters of latent distribution
        ##ladder distribution parameter outputs
        ladder0_std_layers = [nn.Linear(self.ld[1], self.ladder_dim[0]), nn.Sigmoid()]
        ladder1_std_layers = [nn.Linear(self.ld[2], self.ladder_dim[1]), nn.Sigmoid()]
        ladder2_std_layers = [nn.Linear(self.ld[3], self.ladder_dim[2]), nn.Sigmoid()]


        self.ladder0_mean = nn.Linear(self.ld[1], self.ladder_dim[0])
        self.ladder1_mean = nn.Linear(self.ld[2], self.ladder_dim[1])
        self.ladder2_mean = nn.Linear(self.ld[3], self.ladder_dim[2])

        self.ladder0_std = nn.Sequential(*ladder0_std_layers)
        self.ladder1_std = nn.Sequential(*ladder1_std_layers)
        self.ladder2_std = nn.Sequential(*ladder2_std_layers)


        if self.is_continuous:
#             self.fc_mean = nn.Linear(self.hidden_dim, self.latent_cont_dim)
#             self.fc_log_var = nn.Linear(self.hidden_dim, self.latent_cont_dim)
            
            '''self.fc_mean = nn.Sequential(L0Dense(self.hidden_dim, self.latent_cont_dim, droprate_init=self.droprate_init, weight_decay=self.weight_decay, lamba=self.lambda_l0))
        
            self.fc_log_var = nn.Sequential(L0Dense(self.hidden_dim, self.latent_cont_dim, droprate_init=self.droprate_init, weight_decay=self.weight_decay, lamba=self.lambda_l0))'''
        
            self.fc_latent = nn.Sequential(L0Pair(self.hidden_dim, self.latent_cont_dim, droprate_init=self.droprate_init, weight_decay=self.weight_decay, lamba=self.lambda_l0))
#         if self.is_discrete:
#             # Linear layer for each of the categorical distributions
#             fc_alphas = []
#             for disc_dim in self.latent_spec['disc']:
#                  fc_alphas.append(nn.Linear(self.hidden_dim, disc_dim))
# #                 fc_alphas.append(L0Dense(self.hidden_dim, disc_dim, droprate_init=self.droprate_init, weight_decay=self.weight_decay, lamba=self.lambda_l0))
#
#             self.fc_alphas = nn.ModuleList(fc_alphas)

#         self.pruning_latent = nn.Sequential(L0Dense(self.latent_dim, self.latent_dim, droprate_init=self.droprate_init, weight_decay=self.weight_decay, lamba=self.lambda_l0)
#         )
        # Define decoder

        self.combine_latent = Combine_latent()
        self.concat_latent = Concat_latent()

        if self.img_size[1:] == (64, 64):
            generative3_layers = [
                nn.Linear(self.ladder_dim[3], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.ReLU(),
                nn.Linear(self.cs[5], self.cs[5]),
                nn.BatchNorm1d(self.cs[5]),
                nn.ReLU(),
                nn.Linear(self.cs[5], int(self.fs[4]*self.fs[4]*self.cs[4])),
                nn.BatchNorm1d(int(self.fs[4]*self.fs[4]*self.cs[4])),
                nn.ReLU(),
                Reshaper(self.fs[4], self.cs[4])
            ]
            generative2_layers = [
                nn.ConvTranspose2d(2*self.cs[4], self.cs[4], (3, 3), stride=1, padding=1, output_padding=0),
                nn.BatchNorm2d(self.cs[4]),
                nn.ReLU(),
                nn.ConvTranspose2d(self.cs[4], self.cs[3], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[3]),
                nn.ReLU(),
                nn.ConvTranspose2d(self.cs[3], self.cs[3], (3, 3), stride=1, padding=1, output_padding=0),
                nn.BatchNorm2d(self.cs[3]),
                nn.ReLU()
            ]
            generative1_layers = [
                nn.ConvTranspose2d(2 * self.cs[3], self.cs[2], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[2]),
                nn.ReLU(),
                nn.ConvTranspose2d(self.cs[2], self.cs[2], (3, 3), stride=1, padding=1, output_padding=0),
                nn.BatchNorm2d(self.cs[2]),
                nn.ReLU(),
                nn.ConvTranspose2d(self.cs[2], self.cs[1], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[1]),
                nn.ReLU()
            ]
            generative0_layers = [
                nn.ConvTranspose2d(2*self.cs[1], self.cs[1], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[1]),
                nn.ReLU(),
                nn.ConvTranspose2d(self.cs[1], self.img_size[0], (3, 3), stride=1, padding=1, output_padding=0),
                nn.Sigmoid()
            ]
            if self.ladder_dim[0] != 0 and self.ladder_dim[1] != 0:
                ladder2_to_latent_layers = [
                    nn.Linear(self.ladder_dim[2], int(self.fs[4] * self.fs[4] * self.cs[4])),
                    nn.BatchNorm1d(int(self.fs[4] * self.fs[4] * self.cs[4])),
                    nn.ReLU(),
                    Reshaper(self.fs[4], self.cs[4])
                ]
                ladder1_to_latent_layers = [
                    nn.Linear(self.ladder_dim[1], int(self.fs[3] * self.fs[3] * self.cs[3])),
                    nn.BatchNorm1d(int(self.fs[3] * self.fs[3] * self.cs[3])),
                    nn.ReLU(),
                    Reshaper(self.fs[3], self.cs[3])
                ]
                ladder0_to_latent_layers = [
                    nn.Linear(self.ladder_dim[0], int(self.fs[1] * self.fs[1] * self.cs[1])),
                    nn.BatchNorm1d(int(self.fs[1] * self.fs[1] * self.cs[1])),
                    nn.ReLU(),
                    Reshaper(self.fs[1], self.cs[1])
                ]
                self.ladder0_to_latent = nn.Sequential(*ladder0_to_latent_layers)
                self.ladder1_to_latent = nn.Sequential(*ladder1_to_latent_layers)
                self.ladder2_to_latent = nn.Sequential(*ladder2_to_latent_layers)
                self.generative3 = nn.Sequential(*generative3_layers)
        else:
            generative2_layers = [
                nn.Linear(self.ladder_dim[2], self.cs[3]),
                nn.BatchNorm1d(self.cs[3]),
                nn.ReLU(),
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm1d(self.cs[3]),
                nn.ReLU(),
                nn.Linear(self.cs[3], self.cs[3])
            ]

            generative1_layers = [
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm1d(self.cs[3]),
                nn.ReLU(),
                nn.Linear(self.cs[3], self.cs[3]),
                nn.BatchNorm1d(self.cs[3]),
                nn.ReLU(),
                nn.Linear(self.cs[3], self.cs[3])
            ]

            generative0_layers = [
                nn.Linear(self.cs[3], int(self.fs[2]*self.fs[2]*self.cs[2])),
                nn.BatchNorm1d(int(self.fs[2]*self.fs[2]*self.cs[2])),
                nn.ReLU(),
                MNISTReshaper(),
                nn.ConvTranspose2d(self.cs[2], self.cs[1], (4, 4), stride=2, padding=1),
                nn.BatchNorm2d(self.cs[1]),
                nn.LeakyReLU(0.1),
                nn.ConvTranspose2d(self.cs[1], self.img_size[0], (4, 4), stride=2, padding=1),
                nn.Sigmoid()
            ]
            if self.ladder_dim[0] != 0 and self.ladder_dim[1] != 0:
                ladder1_to_latent_layers = [
                    nn.Linear(self.ladder_dim[1], self.cs[3]),
                    nn.BatchNorm1d(self.cs[3]),
                    nn.ReLU()
                ]
                ladder0_to_latent_layers = [
                    nn.Linear(self.ladder_dim[0], self.cs[3]),
                    nn.BatchNorm1d(self.cs[3]),
                    nn.LeakyReLU(0.1)
                ]
                self.ladder0_to_latent = nn.Sequential(*ladder0_to_latent_layers)
                self.ladder1_to_latent = nn.Sequential(*ladder1_to_latent_layers)


        self.generative2 = nn.Sequential(*generative2_layers)
        self.generative1 = nn.Sequential(*generative1_layers)
        self.generative0 = nn.Sequential(*generative0_layers)
        # Define decoder
        #self.features_to_img = nn.Sequential(*decoder_layers)



    def encode(self, x):
        """
        Encodes an image into parameters of a latent distribution defined in
        self.latent_spec.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, H, W)
        :return
        latent disc [cont disc],
        mask  (1, self.out_features)
        L0 regularization
        """
        regularization = 0.
        batch_size = x.size()[0]
        # Encode image to hidden features
#         features = self.img_to_features(x)
#         hidden = self.features_to_hidden(features.view(batch_size, -1))
#
#         # Output parameters of latent distribution from hidden representation
#         latent_dist = {}
#
#         # if self.is_continuous:
#         latent_dist['cont'] = [self.fc_latent(hidden)]
# #             for layer in latent_dist['cont']:
#         regularization += self.lambda_l0*self.fc_latent[0].regularization().cuda()/162079.
#         mask = self.fc_latent[0].sample_mask()

        # if self.is_discrete:
        #     latent_dist['disc'] = []
        #     count = 0
        #     for fc_alpha in self.fc_alphas:
        #
        #         latent_dist['disc'].append(F.softmax(fc_alpha(hidden), dim=1))
        #         #regularization += self.lambda_l0*fc_alpha.regularization().cuda()/162079.
        #         count+=1

        latent_list = {}
        inference0 = self.inference0(x)
        if self.ladder_dim[0] == 0 and self.ladder_dim[1] == 0:
            inference1 = self.inference1(inference0)
            inference2 = self.inference2(inference1)
            ladder3 = self.ladder3(inference2)
            latent_list["ladder3"] = (self.ladder3_mean(ladder3), self.ladder3_std(ladder3) + 0.0001)
        else:
            ladder0 = self.ladder0(x)
            latent_list["ladder0"] = (self.ladder0_mean(ladder0), self.ladder0_std(ladder0) + 0.0001)
            ladder1 = self.ladder1(inference0)
            latent_list["ladder1"] = (self.ladder1_mean(ladder1), self.ladder1_std(ladder1) + 0.0001)
            inference1 = self.inference1(inference0)
            ladder2 = self.ladder2(inference1)

            latent_list["ladder2"] = (self.ladder2_mean(ladder2), self.ladder2_std(ladder2) + 0.0001)
            if self.img_size[1:] == (64, 64):
                inference2 = self.inference2(inference1)
                # inference3= self.inference3(inference2)
                ladder3 = self.ladder3(inference2)
                latent_list["ladder3"] = (self.ladder3_mean(ladder3), self.ladder3_std(ladder3) + 0.0001)
        mask = 1  ##!!

        return latent_list, mask, regularization

    def reparameterize(self, latent_list):
        """
        Samples from latent distribution using the reparameterization trick.

        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
        """
        latent_sample = []
        #device = torch.device('cuda')
        '''
        if self.is_continuous:
            mean, var = latent_dist['cont'][0]
            mean = mean.cuda()
            var = var.cuda()
            #print("var",var[0,:],var.shape)
            cont_sample = self.sample_normal(mean, var)
            latent_sample.append(cont_sample)

        if self.is_discrete:
            for alpha in latent_dist['disc']:
                alpha = alpha.cuda()
                disc_sample = self.sample_gumbel_softmax(alpha.cuda())
                latent_sample.append(disc_sample)
        return torch.cat(latent_sample, dim=1)'''
        for key in latent_list.keys():
            latent_sample.append(self.sample_normal(latent_list[key][0].cuda(), latent_list[key][1].cuda()))

        return latent_sample
        # Concatenate continuous and discrete samples into one large sample


    def sample_normal(self, mean, std):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (N, D) where D is dimension
            of distribution.

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (N, D)
        """
        if self.training:
            eps = torch.zeros(std.size()).normal_()
            if self.use_cuda:
                eps = eps.cuda()
                
            return mean.cuda()+eps*std
        else:
            # Reconstruction mode
            return mean

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if self.use_cuda:
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            if self.use_cuda:
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples

#     def pruning(self, latent_sample):
#         regularization = 0.
#         for layer in self.pruning_latent:
#             regularization += self.lambda_l0*layer.regularization().cuda()/162079.
#         return self.pruning_latent(latent_sample), regularization



    def decode(self, sample_list):
        """
        Decodes sample from latent distribution into an image.

        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """

        # features = self.latent_to_features(latent_sample)
        # return self.features_to_img(features.view(-1, *self.reshape))


        if self.img_size[1:] == (64, 64):
            if self.ladder_dim[0] == 0 and self.ladder_dim[1] == 0:
                latent3 = self.generative3(sample_list[0])
                latent2 = self.generative2(latent3)
                latent1 = self.generative1(latent2)
                img = self.generative0(latent1)

            else:
                latent3 = self.generative3(sample_list[3].cuda())
                ladder2 = self.ladder2_to_latent(sample_list[2].cuda())
                latent2 = self.generative2(self.concat_latent(latent3, ladder2))
                ladder1 = self.ladder1_to_latent(sample_list[1].cuda())
                latent1 = self.generative1(self.concat_latent(latent2, ladder1))
                ladder0 = self.ladder0_to_latent(sample_list[0].cuda())
                img = self.generative0(self.concat_latent(latent1, ladder0))


        else:
            #more layers
            if self.ladder_dim[0] == 0 and self.ladder_dim[1] == 0:
                latent2 = self.generative2(sample_list[0].cuda())
                latent1 = self.generative1(latent2)
                img = self.generative0(latent1)
            else:
                latent2 = self.generative2(sample_list[2].cuda())
                ladder1 = self.ladder1_to_latent(sample_list[1].cuda())
                latent1 = self.generative1(self.combine_latent(latent2, ladder1, method='concat'))
                ladder0 = self.ladder0_to_latent(sample_list[0].cuda())
                img = self.generative0(self.combine_latent(latent1, ladder0, method='concat'))

        return img

    def forward(self, x):
        """
        Forward pass of model.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (N, C, H, W)
        """
        latent_list, mask, regularization = self.encode(x)
        latent_sample = self.reparameterize(latent_list)
        #latent_sample,regularization = self.pruning(latent_sample)
        return self.decode(latent_sample), latent_list, mask, regularization
