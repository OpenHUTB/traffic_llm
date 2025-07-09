from absl import app
from absl import flags
from environment.env import SumoEnv
from light.sumorl.sumo_rl.agents.dqn_agent import DqnAgent
from replay import ReplayBuffer
import torch
import math

import json
import os

import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_integer('skip_range', 50, 'time(seconds) range for skip randomly at the beginning')
flags.DEFINE_float('simulation_time', 10000, 'time for simulation')
flags.DEFINE_integer('yellow_time', 2, 'time for yellow phase')
flags.DEFINE_integer('delta_rs_update_time', 10, 'time for calculate reward')
flags.DEFINE_string('net_file', 'nets/2way-single-intersection/single-intersection.net.xml', '')
flags.DEFINE_string('route_file', 'nets/2way-single-intersection/single-intersection-vhvh.rou.xml', '')
flags.DEFINE_bool('use_gui', False, 'use sumo-gui instead of sumo')
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

    prev_avg_waiting = None
    prev_avg_waiting_time = None
    prev_avg_queue = None

    for episode in range(FLAGS.num_episodes):
        initial_state = env.reset()
        env.train_state = initial_state
        done = False
        invalid_action = False
        episode_rewards = []
        episode_waiting = []
        episode_ratios = []
        episode_max_waiting = []
        episode_min_waiting = []
        episode_waiting_time = []

        episode_records = []  # 保存当前 episode 的所有 step 信息
        
        while not done:
            state = env.compute_state
            action = agent.select_action(state, replay_buffer.steps_done, invalid_action)
            next_state, reward, done, info = env.step(action)

            if info['do_action'] is None:
                invalid_action = True
                continue
            invalid_action = False

            # 收集详细数据
            if next_state is not None and reward is not None:
                step_record = {
                    'episode': episode,
                    'timestamp': env.sumo.simulation.getTime(),
                    'intersection_id': env.ts_id,
                    'action': info['do_action'],
                    'prev_state': env.train_state.tolist(),
                    'next_state': next_state.tolist(),
                    'reward': reward.item() if isinstance(reward, torch.Tensor) else reward,

                    'queue_length': info.get('stats', {}).get('total_waiting', 0),
                    'waiting_time': info.get('stats', {}).get('avg_waiting_time', 0),
                    'waiting_ratio': info.get('stats', {}).get('waiting_ratio', 0),

                }
                # 合并统计信息
                stats = info.get('stats', {})
                step_record.update(stats)
                episode_records.append(step_record)

            #保存经验

            replay_buffer.add(env.train_state, next_state, reward, info['do_action'])
            agent.learn()

            # 收集统计信息
            stats = info.get('stats', {})
            if stats:
                episode_rewards.append(stats.get('reward', 0))
                episode_waiting.append(stats.get('total_waiting', 0))
                episode_ratios.append(stats.get('waiting_ratio', 0))
                episode_max_waiting.append(stats.get('max_waiting', 0))
                episode_min_waiting.append(stats.get('min_waiting', 0))
                episode_waiting_time.append(stats.get('avg_waiting_time', 0))

        env.close()

        avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
        avg_waiting = sum(episode_waiting) / len(episode_waiting) if episode_waiting else 0
        avg_ratio = sum(episode_ratios) / len(episode_ratios) if episode_ratios else 0
        avg_max_waiting = sum(episode_max_waiting) / len(episode_max_waiting) if episode_max_waiting else 0
        avg_min_waiting = sum(episode_min_waiting) / len(episode_min_waiting) if episode_min_waiting else 0
        avg_waiting_time = sum(episode_waiting_time) / len(episode_waiting_time) if episode_waiting_time else 0
        avg_queue = sum(episode_max_waiting) / len(episode_max_waiting) if episode_max_waiting else 0


        print(f'==== Episode {episode} Summary ====')
        print(f'Average Reward: {avg_reward:.3f}')
        print(f'Average Total Waiting Vehicles: {avg_waiting:.2f}')
        print(f'Average Waiting Ratio of Action: {avg_ratio:.3f}')
        print(f'Average Waiting Time per Vehicle: {avg_waiting_time:.2f} s')
        print(f'Maximum Queue Length: {avg_max_waiting:.0f}')
        print(f'Minimum Queue Length: {avg_min_waiting:.0f}')
        print('eps_threshold = :', FLAGS.eps_end + (FLAGS.eps_start - FLAGS.eps_end) *
              math.exp(-1. * replay_buffer.steps_done / FLAGS.eps_decay))
        if prev_avg_waiting is not None:
            print(f'Compared to Episode {episode - 1}:')
            print(f'  ↓ Queue Length Reduced By     : {prev_avg_queue - avg_queue:.2f} vehicles')
            print(f'  ↓ Total Waiting Reduced By    : {prev_avg_waiting - avg_waiting:.2f} vehicles')
            print(f'  ↓ Waiting Time Reduced By     : {prev_avg_waiting_time - avg_waiting_time:.2f} seconds')
        print('======learn_steps:', agent.learn_steps, "========")

        prev_avg_waiting = avg_waiting
        prev_avg_waiting_time = avg_waiting_time
        prev_avg_queue = avg_queue


        # 保存为JSON格式
        if episode_records:
            episode_filename = f'logs/episode_{episode}_trace.json'
            os.makedirs(os.path.dirname(episode_filename), exist_ok=True)
            with open(episode_filename, 'w') as f:
                json.dump(episode_records, f, indent=4)  # 格式化输出
            episode_records.clear()

if __name__ == '__main__':
    app.run(main)
