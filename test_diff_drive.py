import numpy as np
dx, dy = 0.25, 0.1 # Example target: 25cm forward, 10cm left
distance = np.sqrt(dx**2 + dy**2)
angle = np.arctan2(dy, dx) # positive is left
v = min(abs(dx), 0.25) * np.sign(dx) 
omega = np.clip(angle, -np.radians(15), np.radians(15))
print("v:", v, "omega:", np.degrees(omega))

import habitat_sim
print(dir(habitat_sim.utils.common))
