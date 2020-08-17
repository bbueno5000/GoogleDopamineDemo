"""
DOCSTRING
"""
import dopamine
import matplotlib.pyplot as pyplot
import numpy
import os
import seaborn

BASE_PATH = '/tmp/colab_dope_run'
GAME = 'Asterix'
LOG_PATH = os.path.join(BASE_PATH, 'basic_agent', GAME)

class BasicAgent:
    """
    This agent randomly selects an action and sticks to it.
    It will change actions with probability switch_prob.
    """
    def __init__(self, sess, num_actions, switch_prob=0.1):
        self._sess = sess
        self._num_actions = num_actions
        self._switch_prob = switch_prob
        self._last_action = numpy.random.randint(num_actions)
        self.eval_mode = False

    def _choose_action(self):
        """
        We define our policy here.
        """
        if numpy.random.random() <= self._switch_prob:
            self._last_action = numpy.random.randint(self._num_actions)
        return self._last_action
    
    def begin_episode(self, unused_observation):
        """
        First action to take.
        """
        return self._choose_action()
  
    def bundle_and_checkpoint(self, unused_checkpoint_dir, unused_iteration):
        """
        When it checkpoints during training, anything we should do?
        """
        pass

    def end_episode(self, unused_reward):
        """
        Cleanup.
        """
        pass

    def step(self, reward, observation):
        """
        We can update our policy here using the reward and observation.
        """
        return self._choose_action()
    
    def unbundle(self, unused_checkpoint_dir, unused_checkpoint_version, unused_data):
        """
        Loading from checkpoint.
        """
        pass
    
def create_basic_agent(sess, environment):
  """
  The Runner class will expect a function of this type to create an agent.
  """
  return BasicAgent(sess, num_actions=environment.action_space.n, switch_prob=0.2)

# Create the runner class with this agent. We use very small numbers of steps
# to terminate quickly, as this is mostly meant for demonstrating how one can
# use the framework. We also explicitly terminate after 110 iterations (instead
# of the standard 200) to demonstrate the plotting of partial runs.
basic_runner = dopamine.atari.run_experiment.Runner(
    LOG_PATH,
    create_basic_agent,
    game_name=GAME,
    num_iterations=200,
    training_steps=10,
    evaluation_steps=10,
    max_steps_per_episode=100)

print('Will train basic agent, please be patient.')
basic_runner.run_experiment()
print('Done training!')
experimental_data = dopamine.colab.utils.load_baselines('/content')
fig, ax = pyplot.subplots(figsize=(16,8))
seaborn.tsplot(
    data=experimental_data[GAME],
    time='iteration',
    unit='run_number',
    condition='agent',
    value='train_episode_returns',
    ax=ax)
pyplot.title(GAME)
pyplot.show()
