def init():

    global reward, learning_success, test_run_on_model, ou_sigma, ou_theta, actor_learning_rate, critic_learning_rate
    reward = 0
    learning_success = 0
    test_run_on_model = 0
    ou_sigma = 0.15
    ou_theta = 0.2
    actor_learning_rate = 0.0001
    critic_learning_rate = 0.001
