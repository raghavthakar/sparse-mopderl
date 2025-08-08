import os, sys
from pathlib import Path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, ".."))

from datetime import datetime
import numpy as np, os, time, random
import gymnasium as gym, torch
import mo_gymnasium as mo_gym

# -----------------------------------------------------------------------------
# A thin adapter so MOPDERL still sees `info['obj'] = np.ndarray([obj1, obj2])`
class MOPDERLWrapper(gym.Env):
    def __init__(self, base_env):
        super().__init__()
        self.env = base_env
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

    def seed(self, seed=None):
        # forward seeding to the wrapped env and its action_space
        seeds = []
        if hasattr(self.env, "seed"):
            seeds.append(self.env.seed(seed))
        if hasattr(self.action_space, "seed"):
            seeds.append(self.action_space.seed(seed))
        # Gymnasium-style reset seeding for obs RNG can go here if needed
        return seeds

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, vec_r, terminated, truncated, info = self.env.step(action)
        # MO-Gymnasium returns the vector reward as the second return:
        #   vec_r = np.array([velocity, energy])
        info['obj'] = vec_r
        # MOPDERL expects a scalar for training, so we sum them here:
        return obs, float(vec_r.sum()), terminated, truncated, info
# -----------------------------------------------------------------------------
import argparse
from parameters import Parameters
import logging
import mo_agent
import utils
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (MO-Swimmer-v2) (MO-HalfCheetah-v2) (MO-Hopper-v2) ' +
                                 '(MO-Walker2d-v2) (MO-Ant-v2)', required=True, type=str)
parser.add_argument('-seed', help='Random seed to be used', type=int, required=True)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true')
parser.add_argument('-mut_mag', help='The magnitude of the mutation', type=float, default=0.05)
parser.add_argument('-mut_noise', help='Use a random mutation magnitude', action='store_true')
parser.add_argument('-logdir', help='Folder where to save results', type=str, required=True)
parser.add_argument('-warm_up', help='Warm up frames', type=int)
parser.add_argument('-max_frames', help='Max frames', type=int)
parser.add_argument('-num_individuals', help='Number of individual per pderl population', type=int, default=10)
parser.add_argument('-num_generations', help='Max number of generation', type=int)
parser.add_argument('-priority_mag', help='Percent of priority for objective', type=float, default=1.0)
parser.add_argument('-rl_type', help='Type of rl-agents', type=str, default="ddpg")
parser.add_argument('-checkpoint', help='Load checkpoint', action='store_true')
parser.add_argument('-checkpoint_id', help='Select -run- to load checkpoint', type=int)
parser.add_argument('-run_id', help="Specify run id, if not given, get id as len(run)", type=int)
parser.add_argument('-save_ckpt', help="Save checkpoint every _ step, 0 for no save", type=int, default=1)
parser.add_argument('-disable_wandb', action="store_true", default=False)
parser.add_argument('-boundary_only', help="If false, will create a distinct RL agent for each objective and also one that weighs each objective equally", action='store_true', default=True)
parser.add_argument('-warmup_workers', type=int, default=0, help='Parallelize ONLY warm-up policy evaluations. 0=off')
if __name__ == "__main__":
    parameters = Parameters(parser)  # Inject the cla arguments in the parameters object

    if not os.path.exists(parameters.save_foldername):
        os.mkdir(parameters.save_foldername)
    env_folder = os.path.join(parameters.save_foldername, parameters.env_name)
    if not os.path.exists(env_folder):
        os.mkdir(env_folder)
    list_run = sorted(os.listdir(env_folder))
    if parameters.checkpoint:
        if parameters.checkpoint_id is not None:
            run_folder = os.path.join(env_folder, "run_"+str(parameters.checkpoint_id))
        else:
            run_folder = os.path.join(env_folder, list_run[-1])
    else:
        run_id = "run_"+str(len(list_run))
        if parameters.run_id is not None:
            run_id = "run_"+str(parameters.run_id)
        run_folder = os.path.join(env_folder, run_id)
    if not os.path.exists(run_folder):    
        os.mkdir(run_folder)    

    if parameters.wandb: wandb.init(project=parameters.env_name, entity="mopderl", id=str(Path(run_folder).name), resume=parameters.checkpoint) 
    # if parameters.wandb: wandb.init(project=parameters.env_name, entity="mopderl", id=str(Path(run_folder).resolve().parents[0].name), resume=parameters.checkpoint) 
    logging.basicConfig(filename=os.path.join(run_folder, "logger.log"),
                        format=('[%(asctime)s] - '
                                '[%(levelname)4s]:\t'
                                '%(message)s'
                                '\t(%(filename)s:'
                                '%(funcName)s():'
                                '%(lineno)d)\t'),
                        filemode='a',
                        level=logging.DEBUG)
    logger = logging.getLogger()
    logger.info("Start time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

    # Create Env
    # map from MOPDERL’s name → MO-Gymnasium’s name
    name_map = {
        "MO-Swimmer-v2": "mo-swimmer-v5",
        # add other envs here if you need them:
        # "MO-Hopper-v2":  "mo-hopper-v5",
        # "MO-Ant-v2":     "mo-ant-v5",
    }
    mo_name = name_map.get(parameters.env_name, parameters.env_name.lower())
    base_env = mo_gym.make(mo_name)
    setattr(parameters, "mo_env_id", mo_name)
    # wrap + normalize actions exactly like before
    env = utils.NormalizedActions(MOPDERLWrapper(base_env))
    # env = gym.make(parameters.env_name)
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Write the parameters to a the info file and print them
    parameters.write_params(path=run_folder)

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    # Create Agent
    reward_keys = utils.parse_json("reward_keys.json")[parameters.env_name]
    agent = mo_agent.MOAgent(parameters, env, reward_keys, run_folder)
    print('Running', parameters.env_name, ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim)
    logger.info('Running' + str(parameters.env_name) + ' State_dim:' + str(parameters.state_dim) + ' Action_dim:' + str(parameters.action_dim))
    logger.info("Priority: " + str(parameters.priority))

    time_start = time.time()

    warm_up_saved = False

    while np.sum(agent.num_frames < agent.max_frames).astype(int) > 0:
        logger.info("************************************************")
        logger.info("\t\tGeneration: " + str(agent.iterations))
        logger.info("************************************************")
        stats_wandb = agent.train_final(logger)
        # rl_agent_scores = stats['rl_agents_scores']
        # pareto_first_front_type = stats["pareto_1st_front_type"]

        if parameters.wandb and len(stats_wandb):
            current_pareto = stats_wandb.pop("pareto")
            current_pareto = [list(point) for point in current_pareto]
            table = wandb.Table(data=current_pareto, columns=reward_keys)
            wandb.log({ 
                **{"Current pareto front" : wandb.plot.scatter(table, reward_keys[0], reward_keys[1], title="Current pareto front")}, 
                **stats_wandb
            })

        print('#Generation:', agent.iterations, '#Frames:', agent.num_frames,
              ' ENV:  '+ parameters.env_name, 
            #   ' Rl_agent_scores:', rl_agent_scores,
              )
        print()
        logger.info("\n")
        logger.info("\n")
        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # logger.info("=>>>>>> Rl_agent_scores " + str(rl_agent_scores))
        # logger.info("=>>>>>> Pareto_1st_front_type" + str(pareto_first_front_type))
        logger.info("=>>>>>> Num frames: " + str(agent.num_frames))
        logger.info("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

        if (parameters.save_ckpt_period > 0 and agent.iterations % parameters.save_ckpt_period == 0) or \
            np.sum(agent.num_frames < agent.max_frames).astype(int) == 0:
            agent.save_info()
            logger.info("Save info successfully!")
            logger.info("\n")
            logger.info("\n")

        if not warm_up_saved and np.sum(agent.num_frames < parameters.warm_up_frames).astype(int) == 0:
            agent.save_info()
            logger.info("Save warmup infor successfully!!!")   
            logger.info("\n")
            logger.info("\n")
            warm_up_saved = bool(parameters.checkpoint)  # True when resuming from a checkpoint
            break

        if len(stats_wandb):
            agent.archive.save_info()
        

    logger.info("End time: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
