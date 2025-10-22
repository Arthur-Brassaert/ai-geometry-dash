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

from geometry_dash_env import GeometryDashEnv
from logging_config import get_tb_log_root


@dataclass
class TrainingConfig:
    # environment
    n_envs: int = 8
    headless: bool = True

    # training loop
    total_timesteps: int = 5_000_000
    tb_run_name: str = 'ai_full_game'
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


def make_env(headless: bool = True):
    def _init():
        env = GeometryDashEnv(headless=headless)
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

                # archive the latest model with a timestamped filename
                archive_name = None
                if latest:
                    # Overwrite canonical best_model.zip for easy playback
                    dest_canonical = os.path.join(self.trained_models_dir, 'best_model.zip')
                    try:
                        shutil.copy2(latest, dest_canonical)
                        summary['canonical'] = os.path.basename(dest_canonical)
                        summary['canonical_path'] = dest_canonical
                    except Exception:
                        # ignore copy failure, continue to archive
                        pass

                    # Also create an immutable timestamped archive
                    archive_name = f'best_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.zip'
                    archive_path = os.path.join(self.trained_models_dir, archive_name)
                    shutil.copy2(latest, archive_path)
                    summary['archive'] = archive_name
                    summary['archive_path'] = archive_path

                # write JSON summary
                with open(best_file, 'w', encoding='utf-8') as bf:
                    json.dump(summary, bf, indent=2)

                # notify
                self._notify(f"🏆 New global best! mean reward {mean_reward:.3f} (previous {global_best}) saved to {best_file}")
                if archive_name:
                    self._notify(f"Archived best model -> {archive_name}")

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
        if self.cfg.n_envs > 1:
            print(f'Using SubprocVecEnv with {self.cfg.n_envs} workers (parallel envs)')
            env = SubprocVecEnv([make_env(headless=self.cfg.headless) for _ in range(self.cfg.n_envs)])
        else:
            print('Using single-process DummyVecEnv')
            env = DummyVecEnv([make_env(headless=self.cfg.headless) for _ in range(self.cfg.n_envs)])
        env = VecNormalize(env, norm_obs=self.cfg.norm_obs, norm_reward=self.cfg.norm_reward, clip_obs=self.cfg.clip_obs)
        return env

    def build_model(self, env):
        model = PPO(
            'MlpPolicy',
            env,
            tensorboard_log=self.tb_root,
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

        model = self.build_model(env)

        notify_log = os.path.join(self.trained_models_dir, 'last_improvements.log')
        eval_env = DummyVecEnv([make_env(headless=self.cfg.headless)])
        eval_env = VecNormalize(eval_env, norm_obs=self.cfg.norm_obs, norm_reward=self.cfg.norm_reward, clip_obs=self.cfg.clip_obs)

        eval_callback = NotifyingEvalCallback(
            eval_env=eval_env,
            trained_models_dir=self.trained_models_dir,
            cfg=self.cfg,
            best_model_save_path=self.trained_models_dir,
            log_path=self.tb_root,
            eval_freq=self.cfg.eval_freq,
            deterministic=self.cfg.eval_deterministic,
            render=False,
            notify_file=notify_log,
            n_eval_episodes=self.cfg.eval_episodes,
        )

        callbacks = [eval_callback]
        if self.cfg.train_seconds and self.cfg.train_seconds > 0:
            callbacks.append(TimedStopCallback(self.cfg.train_seconds, verbose=1))
        callback = CallbackList(callbacks)

        model.learn(total_timesteps=self.cfg.total_timesteps, tb_log_name=self.cfg.tb_run_name, callback=callback)

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
    parser.add_argument('--keep-archives', type=int, default=None)
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

    return cfg


def main(argv=None):
    cfg = parse_args(argv)
    mgr = TrainingManager(cfg)
    mgr.train()


if __name__ == '__main__':
    main()
