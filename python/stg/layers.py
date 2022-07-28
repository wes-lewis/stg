import torch.nn as nn
import torch
import math
from .utils import get_batcnnorm, get_dropout, get_activation

__all__ = [
    'LinearLayer', 'MLPLayer', 'FeatureSelector',
]

#add concrete layer implementation from: https://discuss.pytorch.org/t/concrete-dropout-implementation/4396/4
class ConcreteDropout(nn.Module):
    
    def __init__(self, layer, input_shape, wr=1e-6, dr=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()
        
        # Post dropout layer
        self.layer = layer
        # Input dim
        self.input_dim = np.prod(input_shape)
        # Regularisation hyper-params
        self.w_reg_param = wr
        self.d_reg_param = dr
        
        # Initialise p_logit
        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)
        
        self.p_logit = nn.Parameter(torch.Tensor(1))
        nn.init.uniform(self.p_logit, a=init_min, b=init_max)
        
    def sum_of_square(self):
        """
        For paramater regularisation
        """
        sum_of_square = 0
        for param in self.layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))
        return sum_of_square
    
    def regularisation(self):
        """
        Returns regularisation term, should be added to the loss
        """
        weights_regularizer = self.w_reg_param * self.sum_of_square() / (1 - self.p)
        dropout_regularizer = self.p * torch.log(self.p)
        dropout_regularizer += (1. - self.p) * torch.log(1. - self.p)
        dropout_regularizer *= self.d_reg_param * self.input_dim
        regularizer = weights_regularizer + dropout_regularizer
        return regularizer
    
    def forward(self, x):
        """
        Forward pass for dropout layer
        """
        eps = 1e-7
        temp = 0.1
        
        self.p = nn.functional.sigmoid(self.p_logit)

        unif_noise = Variable(torch.FloatTensor(np.random.uniform(size=tuple(x.size()))))

        drop_prob = (torch.log(self.p + eps) 
                    - torch.log(1 - self.p + eps)
                    + np.log(unif_noise + eps)
                    - np.log(1 - unif_noise + eps))
        drop_prob = nn.functional.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - self.p

        x  = torch.mul(x, random_tensor)
        x /= retain_prob
        
        return self.layer(x)
    
class Linear_relu(nn.Module):
    
    def __init__(self, inp, out):
        super(Linear_relu, self).__init__()
        self.model = nn.Sequential(nn.Linear(inp, out), nn.ReLU())
        
    def forward(self, x):
        return self.model(x)


class Model(nn.Module):
    
    def __init__(self, wr, dr):
        super(Model, self).__init__()
        self.forward_main = nn.Sequential(
                  ConcreteDropout(Linear_relu(1, nb_features), input_shape=1, wr=wr, dr=dr),
                  ConcreteDropout(Linear_relu(nb_features, nb_features), input_shape=nb_features, wr=wr, dr=dr),
                  ConcreteDropout(Linear_relu(nb_features, nb_features), input_shape=nb_features, wr=wr, dr=dr))
        self.forward_mean = ConcreteDropout(Linear_relu(nb_features, D), input_shape=nb_features, wr=wr, dr=dr)
        self.forward_logvar = ConcreteDropout(Linear_relu(nb_features, D), input_shape=nb_features, wr=wr, dr=dr)
        
    def forward(self, x):
        x = self.forward_main(x)
        mean = self.forward_mean(x)
        log_var = self.forward_logvar(x)
        return mean, log_var

    def heteroscedastic_loss(self, true, mean, log_var):
        precision = torch.exp(-log_var)
        return torch.sum(precision * (true - mean)**2 + log_var)
    
    def regularisation_loss(self):
        reg_loss = self.forward_main[0].regularisation()+self.forward_main[1].regularisation()+self.forward_main[2].regularisation()
        reg_loss += self.forward_mean.regularisation()
        reg_loss += self.forward_logvar.regularisation()
        return reg_loss


class FeatureSelector(nn.Module):
    def __init__(self, input_dim, sigma, device):
        super(FeatureSelector, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.noise = torch.randn(self.mu.size()) 
        self.sigma = sigma
        self.device = device
    
    def forward(self, prev_x):
        z = self.mu + self.sigma*self.noise.normal_()*self.training 
        stochastic_gate = self.hard_sigmoid(z)
        new_x = prev_x * stochastic_gate
        return new_x
    
    def hard_sigmoid(self, x):
        return torch.clamp(x+0.5, 0.0, 1.0)

    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) 

    def _apply(self, fn):
        super(FeatureSelector, self)._apply(fn)
        self.noise = fn(self.noise)
        return self


class GatingLayer(nn.Module):
    '''To implement L1-based gating layer (so that we can compare L1 with L0(STG) in a fair way)
    '''
    def __init__(self, input_dim, device):
        super(GatingLayer, self).__init__()
        self.mu = torch.nn.Parameter(0.01*torch.randn(input_dim, ), requires_grad=True)
        self.device = device
    
    def forward(self, prev_x):
        new_x = prev_x * self.mu 
        return new_x
    
    def regularizer(self, x):
        ''' Gaussian CDF. '''
        return torch.sum(torch.abs(x))


class LinearLayer(nn.Sequential):
    def __init__(self, in_features, out_features, batch_norm=None, dropout=None, bias=None, activation=None):
        if bias is None:
            bias = (batch_norm is None)

        modules = [nn.Linear(in_features, out_features, bias=bias)]
        if batch_norm is not None and batch_norm is not False:
            modules.append(get_batcnnorm(batch_norm, out_features, 1))
        if dropout is not None and dropout is not False:
            modules.append(get_dropout(dropout, 1))
        if activation is not None and activation is not False:
            modules.append(get_activation(activation))
        super().__init__(*modules)

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, batch_norm=None, dropout=None, activation='relu', flatten=True):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = []
        elif type(hidden_dims) is int:
            hidden_dims = [hidden_dims]

        dims = [input_dim]
        dims.extend(hidden_dims)
        dims.append(output_dim)
        modules = []

        nr_hiddens = len(hidden_dims)
        for i in range(nr_hiddens):
            layer = LinearLayer(dims[i], dims[i+1], batch_norm=batch_norm, dropout=dropout, activation=activation)
            modules.append(layer)
        layer = nn.Linear(dims[-2], dims[-1], bias=True)
        modules.append(layer)
        self.mlp = nn.Sequential(*modules)
        self.flatten = flatten

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, input):
        if self.flatten:
            input = input.view(input.size(0), -1)
        return self.mlp(input)
