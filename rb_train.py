#!/usr/bin/env python3
"""RoboBallet training runner for the 64GB / RTX 4090 remote instance.

The stock entry point uses the paper's default replay size (~2e6 items, which
needs ~92 GB RAM) and reloads the full 50k start-states file (~4.6 GB) per
episode. This runner keeps the official training code untouched but overrides:

  1. start-states file -> a 5000-state subsample (~32 MB, fast per-episode load)
  2. replay buffer size -> 200k items (~9 GB, fits 64 GB comfortably)
  3. train_steps / checkpoint cadence so a run actually completes

Run from /root so data_locator relative paths ('roboballet/data/...') resolve:

  cd /root && PYTHONPATH=/root \
    roboballet/venv/bin/python roboballet/rb_train.py \
      --rb_replay_max_size 200000 --rb_train_steps 50000 --num_actors 8
"""
import os

from absl import app
from absl import flags
import launchpad as lp

from roboballet.launch import hparams
from roboballet.launch import launch as launch_mod
from roboballet.launch import launch_utils

FLAGS = flags.FLAGS

flags.DEFINE_integer('rb_replay_max_size', 200000,
                     'Reverb replay capacity (transitions). Dominant RAM use.')
flags.DEFINE_integer('rb_replay_min_size', 20000,
                     'Reverb sample threshold before training starts.')
flags.DEFINE_integer('rb_train_steps', 50000, 'Number of learner steps.')
flags.DEFINE_integer('rb_train_batch_size', 128, 'Training batch size (GPU memory).')
flags.DEFINE_integer('rb_eval_episodes', 5,
                     'Evaluator episodes per network step.')
flags.DEFINE_string('rb_start_states', 'start_states/4r_10t_30o_small.pb',
                    'Start-states .pb path relative to roboballet/data/.')
flags.DEFINE_string('rb_artifacts', '/root/roboballet/run',
                    'Dir for model_checkpoints/ and tensorboard_log/.')


def main(argv):
  del argv

  artifacts = os.path.expanduser(FLAGS.rb_artifacts)
  checkpoint_dir = os.path.join(artifacts, 'model_checkpoints')
  tensorboard_dir = os.path.join(artifacts, 'tensorboard_log')
  os.makedirs(checkpoint_dir, exist_ok=True)
  os.makedirs(tensorboard_dir, exist_ok=True)

  configs = hparams.make_optimised_configs(
      num_targets=launch_mod._NUM_TARGETS.value,
      num_obstacles=launch_mod._NUM_OBSTACLES.value,
      num_robots=launch_mod._NUM_ROBOTS.value,
      pregen=True,
  )

  configs = configs.replace(
      actor_config=configs.actor_config.replace(
          start_states_path=FLAGS.rb_start_states,
          num_actors=launch_mod._NUM_ACTORS.value,
      )
  )

  configs = configs.replace(
      replay_config=launch_utils.ReplayConfig(
          replay_min_size=FLAGS.rb_replay_min_size,
          replay_max_size=FLAGS.rb_replay_max_size,
      )
  )

  configs = configs.replace(
      train_config=configs.train_config.replace(
          train_steps=FLAGS.rb_train_steps,
          train_batch_size=FLAGS.rb_train_batch_size,
          checkpoint_interval=20000,
          stats_reporting_interval=500,
      )
  )

  configs = configs.replace(
      eval_config=configs.eval_config.replace(
          episodes_per_network_step=FLAGS.rb_eval_episodes,
      )
  )

  rc = configs.replay_config
  print('=== RoboBallet training run ===')
  print(f'  artifacts    : {artifacts}')
  print(f'  start states : {FLAGS.rb_start_states}')
  print(f'  num_actors   : {launch_mod._NUM_ACTORS.value}')
  print(f'  replay       : min={rc.replay_min_size} max={rc.replay_max_size}')
  print(f'  train_steps  : {configs.train_config.train_steps}')
  print(f'  est replay RAM: ~{rc.replay_max_size * 46.3e3 / 1e9:.1f} GB')

  program = launch_mod.make_program(
      configs=configs,
      checkpoint_dir=checkpoint_dir,
      starting_checkpoint_dir=None,
      tensorboard_log_dir=tensorboard_dir,
  )

  lp.launch(program, launch_type='local_mp', terminal='output_to_files')


if __name__ == '__main__':
  app.run(main)
