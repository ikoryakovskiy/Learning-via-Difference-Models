def init():

    learning = dict(
        ou_sigma=0.15,
        ou_theta=0.2,
        ou_mu=0.0,
        actor_learning_rate=0.0001,
        critic_learning_rate=0.001,
        gamma = 0.99,
        tau = 0.001
        )

    params = dict(
        learning=learning,
        replay_buffer=dict(load=0, save=0, max_size=300000, min_size=20000, minibatch_size=64),
        transitions=dict(load=0, save=0, save_filename='db_trajectories', buffer_size=5000),
        difference_model=0
        )

    return params
