import pyautogui
import numpy as np
import cv2
import gym
from gym import spaces

class EaglecraftEnv(gym.Env):
    """
    Entorno básico para Eaglecraft.
    """
    def __init__(self):
        super(EaglecraftEnv, self).__init__()

        # Acciones: mover adelante, atrás, izquierda, derecha
        self.action_space = spaces.Discrete(4)

        # Observaciones: Imagen de 84x84x3 (RGB)
        self.observation_space = spaces.Box(low=0, high=255, 
                                            shape=(84, 84, 3), dtype=np.uint8)

    def step(self, action):
        """
        Ejecutar una acción y devolver la nueva observación, recompensa, si terminó y otra info.
        """
        # Ejecutar la acción
        if action == 0:
            pyautogui.keyDown('w')
        elif action == 1:
            pyautogui.keyDown('s')
        elif action == 2:
            pyautogui.keyDown('a')
        elif action == 3:
            pyautogui.keyDown('d')

        # Esperar un momento para que se ejecute la acción
        pyautogui.sleep(0.1)

        # Soltar todas las teclas
        pyautogui.keyUp('w')
        pyautogui.keyUp('s')
        pyautogui.keyUp('a')
        pyautogui.keyUp('d')

        # Capturar la pantalla como nueva observación
        observation = self._get_observation()

        # Recompensa temporal fija
        reward = 1  # Puedes cambiar esto más adelante.

        # Indicar si el episodio terminó
        done = False  # Cambiar esto dependiendo del objetivo.

        return observation, reward, done, {}

    def reset(self):
        """
        Reiniciar el entorno al estado inicial.
        """
        # Recargar el navegador (opcional, depende de cómo uses Eaglecraft)
        pyautogui.hotkey('ctrl', 'r')
        pyautogui.sleep(3)  # Esperar a que cargue el juego

        # Capturar la pantalla inicial
        return self._get_observation()

    def _get_observation(self):
        """
        Capturar la pantalla del juego como entrada para la IA.
        """
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (84, 84))  # Reducir el tamaño para eficiencia
        return frame
