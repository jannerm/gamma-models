import imageio
import numpy as np
import matplotlib.pyplot as plt

def get_states(n_steps, theta_range=[0,2*np.pi], thetadot_range=[-8,8]):
    """
        returns `n_steps`^2 states from pendulum environment,
        sampling angle and angular velocity uniformly within environment bounds
    """
    theta_range = np.linspace(*theta_range, n_steps)
    thetadot_range = np.linspace(*thetadot_range, n_steps)[::-1]

    states = np.array([
        (np.cos(theta), np.sin(theta), thetadot)
        for thetadot in thetadot_range
        for theta in theta_range
    ])

    return states, theta_range, thetadot_range

def get_condition_states():
    """
        returns a few representative conditioning states
    """
    thetas = [np.pi/2, np.pi, np.pi*3/2]
    thetadots = [-4, 0, 4]

    condition_states = np.array([
        (np.cos(theta), np.sin(theta), thetadot)
        for theta in thetas
        for thetadot in thetadots
    ])
    
    return condition_states, thetas, thetadots

def visualize(prob_fn, n_steps=41, save_path=None, itr=None):
    plt.rcParams['figure.figsize'] = [12, 8]

    condition_states, xs, ys = get_condition_states()
    queries, x_range, y_range = get_states(n_steps)

    fig, axes = plt.subplots(len(xs), len(ys))
    axes = axes.flatten()

    for ax, cond in zip(axes, condition_states):
        cond_rep = cond[None].repeat(len(queries), axis=0)

        ## get probabilities
        probs = prob_fn(cond_rep, queries)
        ## reshape probabilities to image
        probs = probs.reshape(n_steps, n_steps)

        plot_probs(probs, x_range, y_range,
            init=cond, title=probs.max(), ax=ax)

    if itr is not None:
        plt.suptitle(f'Iteration {itr}')
        plt.tight_layout(rect=[0, 0, 1, .95])

    if save_path is not None:
        plt.savefig(save_path + '.png')
        img = imageio.imread(save_path + '.png')
    else:
        plt.show()
        img = None

    plt.close()
    return img

def plot_probs(probs, x_range, y_range, init=None, title=None, ax=None, labels=False, ticks=False):
    ax = ax or plt.gca()

    handle = ax.imshow(probs,
        extent=[x_range.min(), x_range.max(),
                y_range.min(), y_range.max()],
        aspect='auto')

    if labels:
        ax.set_ylabel('angle')
        ax.set_xlabel('angular velocity')    

    if ticks:
        n_steps = len(probs)
        tick_inds = [0, int(n_steps/2), n_steps-1]
        ax.set_xticks([round(x_range[i], 2) for i in tick_inds])
        ax.set_yticks([round(y_range[i], 2) for i in tick_inds])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    ## scatter plot for init state
    if init is not None:
        init_cos, init_sin, init_thetadot = init
        init_theta = np.arctan2(init_sin, init_cos) % (2*np.pi)
        ax.scatter(init_theta, init_thetadot, c='white')

        ax.set_title('State: ({:.2f}, {:.2f}) | {:.2e}'.format(init_theta, init_thetadot, title), pad=10)

    elif title is not None:
        ax.set_title('Max: {:.4e}'.format(title), pad=10)

    return handle

def visualize_values(prob_fn, n_steps=41, save_path=None, verbose=False):
    plt.rcParams['figure.figsize'] = [4, 8]

    queries, *_ = get_states(n_steps)
    rewards = reward_fn(queries)

    V = []
    for i, cond in enumerate(queries):
        cond_rep = cond[None].repeat(len(queries), axis=0)

        probs = prob_fn(cond_rep, queries)
        v = (probs * rewards).sum(-1)
        V.append(v)


    V = np.array(V).reshape(n_steps, n_steps)

    plt.imshow(V, aspect='auto')
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path + '.png')
        img = imageio.imread(save_path + '.png')
    else:
        plt.show()
        img = None

    plt.close()
    return img

def reward_fn(states):
    assert states.shape[1] == 3

    cos_theta = states[:,0]
    sin_theta = states[:,1]
    thetadot = states[:,2]

    theta = np.arctan2(sin_theta, cos_theta)
    costs = angle_normalize(theta)**2 + .1*thetadot**2 #+ .001*(u**2)

    return -costs

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)
