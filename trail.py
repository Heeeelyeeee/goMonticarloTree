import gym

go_env = gym.make('gym_go:go-v0', size=7, komi=0, reward_method='real')


while True:
    state, reward, done, info = go_env.step(go_env.uniform_random_action())
    go_env.render('terminal')
    if(done):
        break
