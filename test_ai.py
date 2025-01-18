from eaglecraft_env import EaglecraftEnv
from stable_baselines3 import PPO

# Cargar el entorno
env = EaglecraftEnv()

# Cargar el modelo entrenado
model = PPO.load("eaglecraft_ai")

# Reiniciar el entorno
obs = env.reset()

# Ejecutar acciones de la IA
print("Probando el modelo...")
for _ in range(1000):  # Número de pasos
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break
