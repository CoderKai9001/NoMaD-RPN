import habitat_sim
import numpy as np

def test():
    scene_id = "/scratch/aditya.vadali/data/1W61QJVDBqe/1W61QJVDBqe.basis.glb"
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_id
    backend_cfg.enable_physics = False
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    
    pt = sim.pathfinder.get_random_navigable_point()
    pt2 = pt + np.array([1.0, 0.0, 0.0])
    res = sim.pathfinder.try_step(pt, pt2)
    print("try_step result:", res)

test()
