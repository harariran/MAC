from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy
from gym.spaces import Box
from PIL import Image, ImageOps
from PIL import ImageDraw
from multi_taxi import ObservationType
# for making sure display will work:
import os

from src.decision_makers.MA_decision_makers.MA_AddOn_DM import MA_AddOn_DM, MA_prune_DM

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def np_to_pil(img_arr):
    return Image.fromarray(img_arr)
def pil_write_text(pil,text, color=(255,255,0)):
    pic = ImageOps.expand(pil, border=10, fill=(0, 0, 0))
    temp = ImageDraw.Draw(pic)
    temp.text((0, 0), text, fill=color)
    return pic



class Controller(ABC):
    """An abstract controller class, for other controllers
    to inherit from
    """

    # init agents and their observations
    def __init__(self, environment, agents):
        self.environment = environment
        self.agents = agents

    def run(self, render=True, max_iteration=None):
        """Runs the controller on the environment given in the init,
        with the agents given in the init

        Args:
            render (bool, optional): Whether to render while runngin. Defaults to False.
            max_iteration ([type], optional): Number of steps to run. Defaults to infinity.
        """
        done = False
        index = 0
        num_of_agents = len(self.agents)
        env  = self.environment.get_env()
        # obs_table = {agent_name:[] for agent_name in self.agents.keys()}
        # reward_table = {agent_name:[] for agent_name in self.agents.keys()}
        observations=env.reset()
        self.observation_type=env.unwrapped.observation_type
        if render: env.render()
        self.is_paralel = False
        if isinstance(observations,dict):
            self.is_paralel= True

        # if (not self.is_paralel):
        #       selected_agent= env.agent_selection
        #       obs_table[selected_agent].append(observations)
        # env.render()
        self.total_rewards = []
        while done is not True:
            index += 1
            if max_iteration is not None and index > max_iteration:
                break

            # if observation == None:
            #     observation = {agent_name: None for agent_name in self.agents.keys()}

            # get actions for each agent to perform
            joint_action = self.get_joint_action(observations)

            # perform agents actions
            last_observations=observations
            observations, rewards, dones, truncs, infos = self.perform_joint_action(joint_action)
            self.total_rewards.append(rewards)

            # display environment
            if render:
                if ((self.is_paralel) or (index % num_of_agents)==0):
                # re-render after step
                # if agent_name == env.possible_agents[-1]:
                #     # state only changes after both taxis have stepped
                #     time.sleep(0.15)  # sleep for animation speed control
                #     clear_output(wait=True)  # clear previous render for animation effect
                #     env.render()

                    # todo check obs type - if image False, symbolic-True
                    self.render_obs_next_action(joint_action, last_observations, index)
                    env_str = env.render()


            # check if all agents done to finish
            done = all(value == True for value in dones.values())
            if done:
                break

    def perform_joint_action(self, joint_action):
        if self.is_paralel:
            return self.environment.step(joint_action)
        else:
            for i in range( len(joint_action)):
                current_agent = self.environment.get_env().agent_selection
                self.environment.get_env().step(joint_action[current_agent])

            return self.environment.get_env().last()

    def get_joint_action(self, observation):
        pass

    def render_obs_next_action(self, joint_action, observation, step = None):
        i = 1
        fig = plt.figure(figsize=(16, 4), )
        fig.suptitle('Step # : ' + str(step), fontsize=16)
        for agent_name in self.agents.keys():
                a = joint_action[agent_name]
                ob = observation[agent_name]
                is_image=False
                is_symbolic=False
                if self.observation_type[agent_name]==ObservationType.IMAGE:
                    im_ob = ob
                    is_image=True
                elif self.observation_type[agent_name]==ObservationType.SYMBOLIC:
                    sy_ob = ob
                    is_symbolic=True
                else:
                    im_ob = ob['image']
                    sy_ob = ob['symbolic']
                    is_image=True
                    is_symbolic=True

                # if isinstance(ob,Box) or (isinstance(ob,numpy.ndarray) and len(ob.shape)>1):  # case image
                if is_image: # case has image

                    # option 1 - image per agent
                    # pic = np_to_pil(observation[agent_name])
                    # pic = pil_write_text(pic, " " + str(a) + " (" +
                    #                      self.environment.get_env().unwrapped.get_action_meanings(agent_name)[a] + ")")
                    # pic.show()
                    env = self.environment.get_env()
                    color = env.state().taxi_by_name[agent_name].color
                    dm = self.agents[agent_name].decision_maker
                    dm_str = dm.__class__.__name__

                    if isinstance(dm, MA_AddOn_DM) or isinstance(dm, MA_prune_DM):
                        if dm.is_conflict_on():
                            cf = str(env.unwrapped.get_action_meanings(agent_name)[dm.last_pruned_action]) + '* has'
                        else:
                            cf = 'no'
                        dm_str+=' ' + cf + ' conflict ' + dm.last_dm.__class__.__name__

                    title = str(agent_name) + " (" + color + ") : (" + dm_str + ") " + str(env.unwrapped.get_action_meanings(agent_name)[a])

                    ax = fig.add_subplot(1, len(self.agents), i)
                    i += 1
                    plt.title(title)
                    ax.imshow(im_ob)




                if is_symbolic: # case symbolic
                    print(f"{agent_name} obs:  {sy_ob} \n"
                          f"         action: {a} ({self.environment.get_env().unwrapped.get_action_meanings(agent_name)[a]})")
                    # print(f"{agent_name} obs:\n {observation[agent_name]} , action: {self.environment.get_env().index_action_dictionary[joint_action[agent_name]]}")
        plt.show()