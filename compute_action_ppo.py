#-------#
# Compute action by feeding observation into trained policies
# Read observation from pkl, save action to pkl
# Run get_obs via ros to capture the observation, then run this script from the terminal (python3 compute_action_ppo.py) to compute the action. Run sim_to_real_bm from ros to execute action
#-------#

import os, sys, multiprocessing, gym, ray, shutil, argparse, importlib, glob, time
import numpy as np
# from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.rllib.agents import ppo
from ray.tune.logger import pretty_print
from numpngw import write_apng
import pickle

def setup_config(algo, coop=False, seed=0, extra_configs={}):
    num_processes = multiprocessing.cpu_count()
    if algo == 'ppo':
        config = ppo.DEFAULT_CONFIG.copy()
        config['train_batch_size'] = 32
        config['num_sgd_iter'] = 50
        config['sgd_minibatch_size'] = 2
        config['lambda'] = 0.95
        config['model']['fcnet_hiddens'] = [50, 50]

    config['num_workers'] = num_processes
    config['num_cpus_per_worker'] = 0
    config['seed'] = seed
    config['log_level'] = 'ERROR'
    return {**config, **extra_configs}

def load_policy(algo, env_name, policy_path=None, coop=False, seed=0, extra_configs={}):
    if algo == 'ppo':
        agent = ppo.PPOTrainer(setup_config(algo, coop, seed, extra_configs), 'assistive_gym:'+env_name)
    if policy_path != '':
        if 'checkpoint' in policy_path:
            agent.restore(policy_path)
        else:
            # Find the most recent policy in the directory
            directory = os.path.join(policy_path, algo, env_name)
            files = [f.split('_')[-1] for f in glob.glob(os.path.join(directory, 'checkpoint_*'))]
            files_ints = [int(f) for f in files]
            if files:
                checkpoint_max = max(files_ints)
                checkpoint_num = files_ints.index(checkpoint_max)
                checkpoint_path = os.path.join(directory, 'checkpoint_%s' % files[checkpoint_num], 'checkpoint-%d' % checkpoint_max)
                agent.restore(checkpoint_path)
                # return agent, checkpoint_path
            return agent, None
    return agent, None

def make_env(env_name, coop=False, seed=1001):
    if not coop:
        env = gym.make('assistive_gym:'+env_name)
    else:
        module = importlib.import_module('assistive_gym.envs')
        env_class = getattr(module, env_name.split('-')[0] + 'Env')
        env = env_class()
    env.seed(seed)
    return env

def get_action_from_policy(env_name, algo, policy_path, coop=False, seed=0, extra_configs={}):
    ray.init(num_cpus=multiprocessing.cpu_count(), ignore_reinit_error=True, log_to_driver=False)

    obs = pickle.load(open('/home/hello-robot/git/observation.pkl','rb'))
    print('observation: ', obs)
    test_agent, _ = load_policy(algo, env_name, policy_path, coop, seed, extra_configs)
    action = test_agent.compute_action(obs)
    print('original action', action)
    action = action * [0.44, 1.05, 0.44, 1.05]

    print('scaled action', action)

    # save action for immediate use by sim_to_real_bm node
    pickle.dump(action, open('/home/hello-robot/git/action.pkl','wb'), protocol=2)

    # archive action
    dir = os.path.join('/home/hello-robot/git','real_dc','tl_'+args.tl, 'pose_' + args.pose)
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = 't_' + args.trial + '_action' + timestamp +'.pkl'
    save_path = os.path.join(dir, filename)

    pickle.dump(action, open(save_path,'wb'))

    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL for Assistive Gym')
    parser.add_argument('--env', default='BeddingManipulationSphere-v1',
                        help='Environment to train on (default: BeddingManipulationSphere-v1)')
    parser.add_argument('--algo', default='ppo',
                        help='Reinforcement learning algorithm')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--load-policy-path', default='./trained_models/',
                        help='Path name to saved policy checkpoint (NOTE: Use this to continue training an existing policy, or to evaluate a trained policy)')
    parser.add_argument('--tl', default='tl_4', help='target limb')
    parser.add_argument('--pose', default='0', help='pose')
    parser.add_argument('--trial', default='0', help='trial for the given pose')
    args = parser.parse_args()

    checkpoint_path = None

    # if args.train:
    #     checkpoint_path = train(args.env, args.algo, timesteps_total=args.train_timesteps, save_dir=args.save_dir, load_policy_path=args.load_policy_path, coop=coop, seed=args.seed)
    # if args.render:
    #     render_policy(None, args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, coop=coop, colab=args.colab, seed=args.seed, n_episodes=args.render_episodes)
    # if args.evaluate:
    #     evaluate_policy(args.env, args.algo, checkpoint_path if checkpoint_path is not None else args.load_policy_path, n_episodes=args.eval_episodes, coop=coop, seed=args.seed, verbose=args.verbose)

    get_action_from_policy('BeddingManipulationSphere-v1', 'ppo', checkpoint_path if checkpoint_path is not None else args.load_policy_path, seed=args.seed)

