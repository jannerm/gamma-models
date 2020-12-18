import torch
import pdb


def weighted_logsumexp(log_prob1, log_prob2, discount):
    weighted_prob = (1 - discount) * log_prob1.exp() + discount * log_prob2.exp()
    log_prob = weighted_prob.log()
    return log_prob

def stable_weighted_logsumexp(log_prob1, log_prob2, discount):
    a = torch.max(log_prob1, log_prob2)
    log_prob = weighted_logsumexp(log_prob1 - a, log_prob2 - a, discount)
    log_prob = log_prob + a
    return log_prob


class BootstrapTarget:
    """
        bootstrapped target distribution for mixtures of the form
            (1 - discount) * gaussian + discount * flow
    """

    def __init__(self, dist2, discount):
        self.dist2 = dist2
        self.discount = discount

    def update_p(self, means, sigma):
        self.dist1 = SingleStepGaussian(means, sigma)

    def update_discount(self, discount):
        self.discount = discount
        
    def sample(self, batch_size, condition_dict, next_condition_dict, discount=None):
        discount = discount or self.discount
        assert type(discount) == float

        s1 = self.dist1.sample(condition_dict)
        s2 = self.dist2.sample(batch_size, next_condition_dict)

        batch_size = len(s1)
        num_next = int(batch_size * discount)
        indicator = torch.zeros(batch_size, 1, device=s1.device, dtype=torch.float)
        indicator[:num_next] = 1

        samples = s1 * (1-indicator) + s2 * indicator
        return samples.detach()

    def log_prob(self, x, condition_dict, next_condition_dict):
        log_prob1 = self.dist1.log_prob(x, condition_dict)
        log_prob2 = self.dist2.log_prob(x, next_condition_dict)
        log_prob = stable_weighted_logsumexp(log_prob1, log_prob2, self.discount)
        return log_prob.detach()

    def grad_logp(self, x, condition_dict, next_condition_dict):
        logp = self.log_prob(x, condition_dict, next_condition_dict)
        grad_logp = torch.autograd.grad(logp.sum(), x, retain_graph=True, create_graph=True)[0]
        return grad_logp

class SingleStepGaussian:
    """
        single-step distribution is approximated as a gaussian
            with mean `s'` and covariance `sigma * I`
        takes in a `condition_dict` for consistency with flow
            and bootstrap function calls, but this is ignored
    """

    def __init__(self, means, sigma=1e-4):
        self.sigma = sigma
        sigmas = torch.ones_like(means) * sigma
        self.dist = torch.distributions.Normal(means, sigmas)

    def sample(self, condition_dict=None):
        samples = self.dist.sample()
        return samples
    
    def log_prob(self, x, condition_dict=None):
        log_prob = self.dist.log_prob(x).sum(-1)
        return log_prob

    def grad_logp(self, x, condition_dict=None):
        logp = self.log_prob(x)
        grad_logp = torch.autograd.grad(logp.sum(), x, retain_graph=True, create_graph=True)[0]
        return grad_logp

