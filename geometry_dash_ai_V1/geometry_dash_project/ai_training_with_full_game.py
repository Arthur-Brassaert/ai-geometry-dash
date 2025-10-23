import os
import argparse
import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from geometry_dash_env import GeometryDashEnv, EnvConfig, RewardConfig
from logging_config import get_tb_log_root


@dataclass
class TrainingConfig:
    # environment
    n_envs: int = 8
    headless: bool = True

    # training loop
    total_timesteps: int = 5_000_000
    tb_run_name: str = 'ai_full_game_trained'
    train_seconds: int = 0

    # PPO hyperparams
    policy_net: tuple = (256, 256)
    verbose: int = 1
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 128
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Eval callback
    eval_freq: int = 5_000
    eval_deterministic: bool = True
    eval_episodes: int = 20

    # vecnormalize
    norm_obs: bool = True
    norm_reward: bool = False
    clip_obs: float = 10.0

    # artifact retention
    keep_archives: int = 5

    # environment / level generation knobs
    spawn_min: int = 15
    spawn_max: int = 30
    spike_chance: float = 0.25
    env_speed: float = 6.0 * 60
    group_sizes: str = '1,2,3'  # comma-separated list parsed at runtime
    # curriculum (difficulty scheduling)
    curriculum_steps: int = 200_000  # number of timesteps over which to apply curriculum
    curriculum_start_spike: float = 0.05
    curriculum_end_spike: float = 0.35
    curriculum_start_speed: float = 4.0 * 60
    curriculum_end_speed: float = 8.0 * 60

    # reward knobs (centralized and easy to adjust)
    reward_frame: float = 0.1
    reward_pass: float = 10.0
    reward_death: float = -100.0
    reward_ground: float = 0.05

    # randomness / seeding
    seed: Optional[int] = None


def make_env(headless: bool = True, env_cfg: Optional[EnvConfig] = None, reward_cfg: Optional[RewardConfig] = None, seed: Optional[int] = None):
    def _init():
        env = GeometryDashEnv(headless=headless, env_cfg=env_cfg, reward_cfg=reward_cfg, seed=seed)
        return Monitor(env)

    return _init



class NotifyingEvalCallback(EvalCallback):
    """Eval callback that logs improvements, writes a best_overall.json and archives the model.

    Behaviour preserved from the previous script: write a per-run notification into a file,
    update best_overall.json, archive the checkpoint (timestamped), and keep only the last N archives.
    """

    def __init__(self, eval_env, trained_models_dir: str, cfg: TrainingConfig, notify_file: Optional[str] = None, **kwargs):
        super().__init__(eval_env, **kwargs)
        self.notify_file = notify_file
        self.trained_models_dir = trained_models_dir
        self.cfg = cfg
        # Initialize local best from persistent global best if available.
        try:
            best_file = os.path.join(self.trained_models_dir, 'best_overall.json')
            if os.path.exists(best_file):
                try:
                    with open(best_file, 'r', encoding='utf-8') as bf:
                        data = json.load(bf)
                        if 'best_mean_reward' in data and data['best_mean_reward'] is not None:
                            # Use the global best as the starting best_mean_reward so
                            # logging reflects comparisons to the true persisted best
                            # instead of showing 'was -inf' after a restart.
                            self.best_mean_reward = float(data['best_mean_reward'])
                except Exception:
                    pass
            else:
                # If the persistent best file was removed, create a minimal template
                # so subsequent code can update it reliably. We do not set a numerical
                # best here — let training/eval determine the first measurable best.
                try:
                    template = {
                        'best_mean_reward': None,
                        'previous_best': None,
                        'run': None,
                        'timestamp': None,
                    }
                    with open(best_file, 'w', encoding='utf-8') as bf:
                        json.dump(template, bf, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

    def _notify(self, message: str):
        try:
            print(message)
        except Exception:
            pass
        try:
            print('\a')
        except Exception:
            pass
        if self.notify_file:
            try:
                with open(self.notify_file, 'a', encoding='utf-8') as f:
                    f.write(message + '\n')
            except Exception:
                pass

    def _update_best_json_and_archive(self, mean_reward: float):
        best_file = os.path.join(self.trained_models_dir, 'best_overall.json')
        global_best = None
        if os.path.exists(best_file):
            try:
                with open(best_file, 'r', encoding='utf-8') as bf:
                    data = json.load(bf)
                    if 'best_mean_reward' in data:
                        global_best = float(data['best_mean_reward'])
            except Exception:
                global_best = None

        if global_best is None or mean_reward > global_best:
            summary = {
                'best_mean_reward': float(mean_reward),
                'previous_best': None if global_best is None else float(global_best),
                'run': getattr(self, 'log_path', self.cfg.tb_run_name) or self.cfg.tb_run_name,
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            }
            try:
                # find the most recently saved .zip (SB3's EvalCallback writes this)
                zips = [os.path.join(self.trained_models_dir, f) for f in os.listdir(self.trained_models_dir) if f.endswith('.zip')]
                zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                latest = zips[0] if zips else None

                # update canonical best_model.zip only (no timestamped archives)
                if latest:
                    dest_canonical = os.path.join(self.trained_models_dir, 'best_model.zip')
                    tmp_dest = dest_canonical + '.tmp'
                    try:
                        # copy latest to a temp file first, then atomically replace
                        shutil.copy2(latest, tmp_dest)
                        os.replace(tmp_dest, dest_canonical)
                        summary['canonical'] = os.path.basename(dest_canonical)
                        summary['canonical_path'] = dest_canonical
                        # log replacement
                        if self.notify_file:
                            try:
                                with open(self.notify_file, 'a', encoding='utf-8') as nf:
                                    nf.write(f'Replaced canonical best with {os.path.basename(latest)} -> {dest_canonical}\n')
                            except Exception:
                                pass
                    except Exception:
                        # cleanup temp file if needed, but don't create extra files
                        try:
                            if os.path.exists(tmp_dest):
                                os.remove(tmp_dest)
                        except Exception:
                            pass

                # write JSON summary
                with open(best_file, 'w', encoding='utf-8') as bf:
                    json.dump(summary, bf, indent=2)

                # notify
                self._notify(f"🏆 New global best! mean reward {mean_reward:.3f} (previous {global_best}) saved to {best_file}")

                # prune older archives
                try:
                    archives = [os.path.join(self.trained_models_dir, f) for f in os.listdir(self.trained_models_dir) if f.startswith('best_') and f.endswith('.zip')]
                    if len(archives) > self.cfg.keep_archives:
                        archives.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                        for old in archives[self.cfg.keep_archives:]:
                            try:
                                os.remove(old)
                            except Exception:
                                pass
                except Exception:
                    pass

            except Exception:
                # don't fail training if archiving or JSON write fails
                pass

    def _on_step(self) -> bool:  # type: ignore[override]
        prev = getattr(self, 'best_mean_reward', -float('inf'))
        result = super()._on_step()
        new = getattr(self, 'best_mean_reward', prev)

        if new is not None and new > prev:
            msg = f"🔔 Improvement detected: mean reward {new:.3f} (was {prev:.3f}) at {datetime.utcnow().isoformat()}Z"
            self._notify(msg)
            # Try to rename the most recent checkpoint produced by SB3's EvalCallback
            # so it includes the run/model name instead of a generic prefix like 'ppo'.
            try:
                zips = [os.path.join(self.trained_models_dir, f) for f in os.listdir(self.trained_models_dir) if f.endswith('.zip')]
                zips.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                latest = zips[0] if zips else None
                if latest:
                    run_name = getattr(self.cfg, 'tb_run_name', None) or 'run'
                    base = os.path.basename(latest)
                    target_name = f"{run_name}.zip"
                    # Only rename if it isn't already the desired target name
                    if base != target_name:
                        dest = os.path.join(self.trained_models_dir, target_name)
                        # If a file with the desired name exists, find a small numeric suffix
                        if os.path.exists(dest):
                            i = 1
                            while True:
                                candidate = os.path.join(self.trained_models_dir, f"{run_name}_{i}.zip")
                                if not os.path.exists(candidate):
                                    dest = candidate
                                    break
                                i += 1
                        try:
                            shutil.move(latest, dest)
                            latest = dest
                            self._notify(f"Renamed checkpoint {base} -> {os.path.basename(dest)}")
                        except Exception:
                            # ignore rename failures
                            pass
            except Exception:
                latest = None

            try:
                self._update_best_json_and_archive(new)
            except Exception:
                pass

        return result


class TimedStopCallback(BaseCallback):
    def __init__(self, max_seconds: int, verbose=0):
        super().__init__(verbose=verbose)
        self.max_seconds = max_seconds
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.max_seconds and self.start_time is not None:
            if time.time() - self.start_time > self.max_seconds:
                if self.verbose:
                    print(f"TimedStopCallback: reached {self.max_seconds}s, stopping training")
                return False
        return True


class CurriculumCallback(BaseCallback):
    """Gradually adjusts environment difficulty (spike chance and speed) over training time.

    The callback expects that the vectorized env exposes an `apply_env_updates`
    method on the underlying Gym env which accepts a dict of env-config updates.
    """

    def __init__(self, total_schedule_steps: int, cfg: TrainingConfig, verbose=0):
        super().__init__(verbose=verbose)
        self.total_schedule_steps = int(total_schedule_steps)
        self.cfg = cfg

    def _on_step(self) -> bool:
        # compute progress in [0,1]
        num_steps = getattr(self.model, 'num_timesteps', 0)
        if self.total_schedule_steps <= 0:
            return True
        progress = min(1.0, float(num_steps) / float(self.total_schedule_steps))

        # linear interpolation for spike chance and speed
        spike = float(self.cfg.curriculum_start_spike + progress * (self.cfg.curriculum_end_spike - self.cfg.curriculum_start_spike))
        speed = float(self.cfg.curriculum_start_speed + progress * (self.cfg.curriculum_end_speed - self.cfg.curriculum_start_speed))

        # call env_method on the VecEnv to apply updates to each sub-env
        try:
            # VecEnv.env_method will call apply_env_updates on each inner env if available
            self.model.get_env().env_method('apply_env_updates', {'spike_chance': spike, 'speed_px_s': speed})
            if self.verbose:
                print(f'CurriculumCallback: progress={progress:.3f} spike={spike:.4f} speed={speed:.2f}')
        except Exception:
            # ignore if method not present
            pass

        return True


class TrainingManager:
    def __init__(self, cfg: TrainingConfig, project_dir: Optional[str] = None):
        self.cfg = cfg
        self.project_dir = project_dir or os.path.dirname(os.path.abspath(__file__))
        self.trained_models_dir = os.path.join(self.project_dir, 'trained_models')
        os.makedirs(self.trained_models_dir, exist_ok=True)

        # remove legacy global_best.zip if present
        try:
            legacy = os.path.join(self.trained_models_dir, 'global_best.zip')
            if os.path.exists(legacy):
                os.remove(legacy)
        except Exception:
            pass

        self.tb_root = get_tb_log_root()

    def _safe_run_name(self, name: Optional[str]) -> str:
        if name is None:
            return datetime.utcnow().strftime('run_%Y%m%d_%H%M%S')
        s = str(name).strip()
        if not s or s.lower() == 'none':
            return datetime.utcnow().strftime('run_%Y%m%d_%H%M%S')
        for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
            s = s.replace(ch, '_')
        return s

    def create_vec_env(self):
        # Build env and reward configs from training config
        try:
            group_sizes = [int(x) for x in str(self.cfg.group_sizes).split(',') if x.strip()]
        except Exception:
            group_sizes = [1, 2, 3]
        env_cfg = EnvConfig(
            width=1000,
            height=600,
            spawn_min=self.cfg.spawn_min,
            spawn_max=self.cfg.spawn_max,
            group_sizes=group_sizes,
            group_internal_gap=0,
            spike_chance=self.cfg.spike_chance,
            speed_px_s=self.cfg.env_speed,
        )
        reward_cfg = RewardConfig(
            frame_reward=self.cfg.reward_frame,
            pass_reward=self.cfg.reward_pass,
            death_penalty=self.cfg.reward_death,
            ground_bonus=self.cfg.reward_ground,
        )

        # Construct per-worker envs with independent seeds so each env sees different levels
        if self.cfg.n_envs > 1:
            print(f'Using SubprocVecEnv with {self.cfg.n_envs} workers (parallel envs)')
            env_fns = []
            for i in range(self.cfg.n_envs):
                seed = None if self.cfg.seed is None else int(self.cfg.seed + i + 1)
                env_fns.append(make_env(headless=self.cfg.headless, env_cfg=env_cfg, reward_cfg=reward_cfg, seed=seed))
            env = SubprocVecEnv(env_fns)
        else:
            print('Using single-process DummyVecEnv')
            seed = None if self.cfg.seed is None else int(self.cfg.seed + 1)
            env = DummyVecEnv([make_env(headless=self.cfg.headless, env_cfg=env_cfg, reward_cfg=reward_cfg, seed=seed)])
        env = VecNormalize(env, norm_obs=self.cfg.norm_obs, norm_reward=self.cfg.norm_reward, clip_obs=self.cfg.clip_obs)
        return env

    def build_model(self, env, tb_run_dir: Optional[str] = None):
        """Build a PPO model and direct TensorBoard logs into tb_run_dir.

        If tb_run_dir is not provided, fall back to the training manager's TB root.
        """
        tb_dir = tb_run_dir or self.tb_root
        try:
            os.makedirs(tb_dir, exist_ok=True)
        except Exception:
            pass

        model = PPO(
            'MlpPolicy',
            env,
            tensorboard_log=tb_dir,
            verbose=self.cfg.verbose,
            policy_kwargs=dict(net_arch=list(self.cfg.policy_net)),
            learning_rate=self.cfg.learning_rate,
            n_steps=self.cfg.n_steps,
            batch_size=self.cfg.batch_size,
            n_epochs=self.cfg.n_epochs,
            gamma=self.cfg.gamma,
            gae_lambda=self.cfg.gae_lambda,
            clip_range=self.cfg.clip_range,
            ent_coef=self.cfg.ent_coef,
            vf_coef=self.cfg.vf_coef,
            max_grad_norm=self.cfg.max_grad_norm,
        )
        return model

    def train(self):
        # lock run name
        old_name = self.cfg.tb_run_name
        self.cfg.tb_run_name = self._safe_run_name(self.cfg.tb_run_name)
        if self.cfg.tb_run_name != old_name:
            print(f"Note: using fallback run name '{self.cfg.tb_run_name}' instead of '{old_name}' to avoid invalid filenames")

        env = self.create_vec_env()
        vecnorm_path = os.path.join(self.trained_models_dir, f'{self.cfg.tb_run_name}_vecnormalize.pkl')

        # Use the name of the trained model as the TensorBoard run folder name.
        # This ensures the TB run shown matches the saved model filename.
        save_basename = f"{self.cfg.tb_run_name}.zip"
        run_name_from_model = os.path.splitext(save_basename)[0]
        # Ensure a run-name-specific folder exists under the TB root so writers
        # used by SB3 and the eval callback write under a predictable path.
        run_log_path = os.path.join(self.tb_root, run_name_from_model)
        try:
            os.makedirs(run_log_path, exist_ok=True)
        except Exception:
            pass

        # Ensure the run folder has at least one TensorBoard event file so the
        # TensorBoard UI lists the run name immediately. Create a tiny SummaryWriter
        # event if a compatible writer is available.
        try:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception:
                # fall back to tensorboardX if present
                try:
                    from tensorboardX import SummaryWriter  # type: ignore
                except Exception:
                    SummaryWriter = None
            if SummaryWriter:
                try:
                    w = SummaryWriter(log_dir=run_log_path)
                    # write a trivial scalar so an events file is created
                    w.add_scalar('run/initialized', 1, 0)
                    w.close()
                except Exception:
                    pass
        except Exception:
            pass

        model = self.build_model(env, tb_run_dir=run_log_path)

        notify_log = os.path.join(self.trained_models_dir, 'last_improvements.log')

        # Recreate the env/reward configs here for the eval env (same defaults as used in create_vec_env)
        try:
            group_sizes = [int(x) for x in str(self.cfg.group_sizes).split(',') if x.strip()]
        except Exception:
            group_sizes = [1, 2, 3]
        env_cfg = EnvConfig(
            width=1000,
            height=600,
            spawn_min=self.cfg.spawn_min,
            spawn_max=self.cfg.spawn_max,
            group_sizes=group_sizes,
            group_internal_gap=0,
            spike_chance=self.cfg.spike_chance,
            speed_px_s=self.cfg.env_speed,
        )
        reward_cfg = RewardConfig(
            frame_reward=self.cfg.reward_frame,
            pass_reward=self.cfg.reward_pass,
            death_penalty=self.cfg.reward_death,
            ground_bonus=self.cfg.reward_ground,
        )

        # Build an eval env with the same config but single-process and its own seed
        eval_seed = None if self.cfg.seed is None else int(self.cfg.seed + 9999)
        eval_env = DummyVecEnv([make_env(headless=self.cfg.headless, env_cfg=env_cfg, reward_cfg=reward_cfg, seed=eval_seed)])
        eval_env = VecNormalize(eval_env, norm_obs=self.cfg.norm_obs, norm_reward=self.cfg.norm_reward, clip_obs=self.cfg.clip_obs)

        eval_callback = NotifyingEvalCallback(
            eval_env=eval_env,
            trained_models_dir=self.trained_models_dir,
            cfg=self.cfg,
            best_model_save_path=self.trained_models_dir,
            log_path=run_log_path,
            eval_freq=self.cfg.eval_freq,
            deterministic=self.cfg.eval_deterministic,
            render=False,
            notify_file=notify_log,
            n_eval_episodes=self.cfg.eval_episodes,
        )

        callbacks = [eval_callback]
        if self.cfg.train_seconds and self.cfg.train_seconds > 0:
            callbacks.append(TimedStopCallback(self.cfg.train_seconds, verbose=1))
        # Curriculum: progressively increase difficulty over cfg.curriculum_steps timesteps
        if getattr(self.cfg, 'curriculum_steps', 0) and self.cfg.curriculum_steps > 0:
            callbacks.append(CurriculumCallback(total_schedule_steps=self.cfg.curriculum_steps, cfg=self.cfg, verbose=0))
        callback = CallbackList(callbacks)

        # Start learning and force the tensorboard writer to use the provided
        # run-specific folder directly by passing an empty tb_log_name.
        # This prevents SB3 from creating a subfolder named after the algorithm
        # (e.g. 'ppo').
        model.learn(total_timesteps=self.cfg.total_timesteps, tb_log_name='', callback=callback)

        # Save final model and VecNormalize stats
        save_path = os.path.join(self.trained_models_dir, f'{self.cfg.tb_run_name}.zip')
        model.save(save_path)
        print('Saved model to', save_path)
        try:
            env.save(vecnorm_path)
            print('Saved VecNormalize stats to', vecnorm_path)
        except Exception as e:
            print('Warning: failed to save VecNormalize stats:', e)


def parse_args(argv=None) -> TrainingConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=None)
    parser.add_argument('--n-envs', type=int, default=None)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--policy-net', type=int, nargs='+', default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--n-steps', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--n-epochs', type=int, default=None)
    parser.add_argument('--gamma', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=None)
    parser.add_argument('--clip-range', type=float, default=None)
    parser.add_argument('--ent-coef', type=float, default=None)
    parser.add_argument('--vf-coef', type=float, default=None)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--eval-freq', type=int, default=None)
    parser.add_argument('--eval-episodes', type=int, default=None)
    parser.add_argument('--train-seconds', type=int, default=None)
    parser.add_argument('--curriculum-steps', type=int, default=None)
    parser.add_argument('--curriculum-start-spike', type=float, default=None)
    parser.add_argument('--curriculum-end-spike', type=float, default=None)
    parser.add_argument('--curriculum-start-speed', type=float, default=None)
    parser.add_argument('--curriculum-end-speed', type=float, default=None)
    parser.add_argument('--keep-archives', type=int, default=None)
    parser.add_argument('--spawn-min', type=int, default=None)
    parser.add_argument('--spawn-max', type=int, default=None)
    parser.add_argument('--spike-chance', type=float, default=None)
    parser.add_argument('--env-speed', type=float, default=None)
    parser.add_argument('--group-sizes', type=str, default=None, help='Comma-separated group sizes, e.g. "1,2,3"')
    parser.add_argument('--seed', type=int, default=None, help='Base seed for env workers')

    # reward knobs
    parser.add_argument('--reward-frame', type=float, default=None)
    parser.add_argument('--reward-pass', type=float, default=None)
    parser.add_argument('--reward-death', type=float, default=None)
    parser.add_argument('--reward-ground', type=float, default=None)
    args = parser.parse_args(argv)

    cfg = TrainingConfig()
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.timesteps is not None:
        cfg.total_timesteps = args.timesteps
    if args.model_name is not None:
        cfg.tb_run_name = args.model_name
    if args.policy_net is not None and len(args.policy_net) > 0:
        cfg.policy_net = tuple(args.policy_net)
    if args.eval_freq is not None:
        cfg.eval_freq = args.eval_freq
    if args.eval_episodes is not None:
        cfg.eval_episodes = args.eval_episodes
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.n_steps is not None:
        cfg.n_steps = args.n_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.n_epochs is not None:
        cfg.n_epochs = args.n_epochs
    if args.gamma is not None:
        cfg.gamma = args.gamma
    if args.gae_lambda is not None:
        cfg.gae_lambda = args.gae_lambda
    if args.clip_range is not None:
        cfg.clip_range = args.clip_range
    if args.ent_coef is not None:
        cfg.ent_coef = args.ent_coef
    if args.vf_coef is not None:
        cfg.vf_coef = args.vf_coef
    if args.max_grad_norm is not None:
        cfg.max_grad_norm = args.max_grad_norm
    if args.train_seconds is not None:
        cfg.train_seconds = args.train_seconds
    if args.keep_archives is not None:
        cfg.keep_archives = args.keep_archives
    if args.spawn_min is not None:
        cfg.spawn_min = args.spawn_min
    if args.spawn_max is not None:
        cfg.spawn_max = args.spawn_max
    if args.spike_chance is not None:
        cfg.spike_chance = args.spike_chance
    if args.env_speed is not None:
        cfg.env_speed = args.env_speed
    if args.group_sizes is not None:
        cfg.group_sizes = args.group_sizes
    if args.seed is not None:
        cfg.seed = args.seed

    # curriculum args
    if args.curriculum_steps is not None:
        cfg.curriculum_steps = args.curriculum_steps
    if args.curriculum_start_spike is not None:
        cfg.curriculum_start_spike = args.curriculum_start_spike
    if args.curriculum_end_spike is not None:
        cfg.curriculum_end_spike = args.curriculum_end_spike
    if args.curriculum_start_speed is not None:
        cfg.curriculum_start_speed = args.curriculum_start_speed
    if args.curriculum_end_speed is not None:
        cfg.curriculum_end_speed = args.curriculum_end_speed

    # reward knobs
    if args.reward_frame is not None:
        cfg.reward_frame = args.reward_frame
    if args.reward_pass is not None:
        cfg.reward_pass = args.reward_pass
    if args.reward_death is not None:
        cfg.reward_death = args.reward_death
    if args.reward_ground is not None:
        cfg.reward_ground = args.reward_ground

    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    mgr = TrainingManager(cfg)
    mgr.train()


if __name__ == '__main__':
    main()
