from gymnasium.envs.registration import register

register(
    id="GymAdapter-v0",
    entry_point="gymnasium_env.gymnasium_env.envs.gym_adapter:SquareAdaptedGymSimulation",
)
