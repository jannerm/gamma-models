from gamma.utils.arrays import to_torch

def make_condition(states, actions):
    condition_dict = {
        's': to_torch(states),
        'a': to_torch(actions),
    }
    return condition_dict

def format_batch(batch, policy):
    next_actions = policy(batch['next_observations'])
    condition_dict = make_condition(batch['observations'], batch['actions'])
    next_condition_dict = make_condition(batch['next_observations'], next_actions)
    return condition_dict, next_condition_dict

def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )