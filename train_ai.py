from eaglecraft_env import EaglecraftEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Crear el entorno
env = make_vec_env(lambda: EaglecraftEnv(), n_envs=1)

# Crear el modelo PPO
model = PPO("CnnPolicy", env, verbose=1)

# Entrenar el modelo
print("Comenzando el entrenamiento...")
model.learn(total_timesteps=10000)

# Guardar el modelo
model.save("eaglecraft_ai")
print("Modelo guardado como eaglecraft_ai.zip")
