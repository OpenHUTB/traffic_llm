from absl import app
from absl import flags
from environment.env import SumoEnv
from light.sumorl.sumo_rl.agents.dqn_agent import DqnAgent
from replay import ReplayBuffer
import torch
import math

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 50, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/single-intersection-vhvh.rou.xml', '')
flags.DEFINE_bool('use_gui', True, 'use sumo-gui instead of sumo')
flags.DEFINE_integer('num_episodes', 301, '')
flags.DEFINE_string('network', 'dqn', '')
flags.DEFINE_string('mode', 'train', '')
flags.DEFINE_float('eps_start', 1.0, '')
flags.DEFINE_float('eps_end', 0.1, '')
flags.DEFINE_integer('eps_decay', 83000, '')
flags.DEFINE_integer('target_update', 3000, '')
flags.DEFINE_string('network_file', '', '')
flags.DEFINE_float('gamma', 0.95, '')
flags.DEFINE_integer('batch_size', 32, '')

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(argv):
    del argv
    env = SumoEnv(net_file=FLAGS.net_file,
                  route_file=FLAGS.route_file,
                  skip_range=FLAGS.skip_range,
                  simulation_time=FLAGS.simulation_time,
                  yellow_time=FLAGS.yellow_time,
                  delta_rs_update_time=FLAGS.delta_rs_update_time,
                  use_gui=FLAGS.use_gui
                  )
    replay_buffer = ReplayBuffer(capacity=20000)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    agent = DqnAgent(FLAGS.mode, replay_buffer, FLAGS.target_update, FLAGS.gamma, FLAGS.eps_start, FLAGS.eps_end,
                     FLAGS.eps_decay, input_dim, output_dim, FLAGS.batch_size, FLAGS.network_file)

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        while not done:
            state = env.compute_state
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            next_state, reward, done, info = env.step(action)
            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False

            replay_buffer.add(env.train_state, next_state, reward, info['do_action'])
            agent.learn()

        env.close()

        print('i_episode:', episode)
        print('eps_threshold = :', FLAGS.eps_end + (FLAGS.eps_start - FLAGS.eps_end) *
              math.exp(-1. * replay_buffer.steps_done / FLAGS.eps_decay))
        print('learn_steps:', agent.learn_steps)


if __name__ == '__main__':
    app.run(main)
