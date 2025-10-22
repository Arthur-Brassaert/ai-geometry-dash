import pygame
import torch
import numpy as np
from geometry_dash_env import GeometryDashEnv  # jouw environment
from env.geometry_dash_ import DQN           # je DQN model (of dummy random acties)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Load the final model from the models/ directory created by the trainer
MODEL_PATH = "models/dqn_final.pth"

def test(render=True, fps=60):
    pygame.init()
    clock = pygame.time.Clock()

    env = GeometryDashEnv()
    state_size = env.STATE_SIZE
    action_size = env.ACTION_SIZE

    # laad model als die bestaat
    try:
        model = DQN(state_size, action_size).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"[INFO] Model geladen: {MODEL_PATH}")
    except FileNotFoundError:
        model = None
        print(f"[WAARSCHUWING] Modelbestand '{MODEL_PATH}' niet gevonden. AI kiest random acties.")

    # reset environment
    state = env.reset()

    running = True
    while running:
        if render:
            env.render()  # gebruikt de echte Game.render() methode

        # bepaal actie
        if model:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = int(torch.argmax(q_values).item())
        else:
            action = np.random.randint(0, action_size)

        # stap uitvoeren
        next_state, reward, done, _ = env.step(action)
        state = next_state

        if done:
            print("[INFO] Speler overleden, reset environment")
            state = env.reset()

        # FPS limiter
        if render:
            clock.tick(fps)

if __name__ == "__main__":
    test(render=True, fps=60)
