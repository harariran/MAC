import jax.numpy as jnp

from utils import map_observation, get_shapes, load_model, checkpoint_dir, get_action_shape
from models import probabilityMultiTaxi

class BCAgent:
    def __init__(self, env, agent_name, checkpoint_name):
        self.env = env
        self.agent_name = agent_name

        symbolic_shape, img_shape = get_shapes(env, agent_name)
        self.num_actions = get_action_shape(env, agent_name)
        model = probabilityMultiTaxi(img_shape, symbolic_shape, self.num_actions)
        self.model = load_model(model, checkpoint_dir, checkpoint_name)

    def get_action(self, obs, pruned_action=None):
        self.model.eval()

        obs = obs['symbolic']
        symbolic_obs, domain_map = map_observation(self.env, obs)
        symbolic_obs = jnp.delete(symbolic_obs, jnp.array([5,9], dtype=jnp.int32))

        probs = self.model(symbolic_obs, domain_map)

        if pruned_action is not None:
            probs = probs.at[pruned_action].set(0.0)
            probs = probs / jnp.sum(probs)

        a = jnp.argmax(probs, axis=-1)
        a = int(a[0])

        return a
