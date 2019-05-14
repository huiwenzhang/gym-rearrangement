"""
Parse fetch xml file
"""

from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
print(base_path)

env_name = 'pick_and_place.xml'
xml_path = os.path.join(base_path, 'gym_rearrangement', 'envs', 'robotics', 'assets', 'fetch', env_name)

model = load_model_from_path(xml_path)
sim = MjSim(model)
viewer = MjViewer(sim)

t = 0
while True:
    t += 1
    sim.step()
    viewer.render()
    if t > 1000:
        break
