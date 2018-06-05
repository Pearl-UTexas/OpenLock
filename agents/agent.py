
from shutil import copytree, ignore_patterns
import time

from logger import SubjectLogger, SubjectWriter


# base class for all agents; each agent has a logger
class Agent(object):
    def __init__(self, data_path):
        self.logger = None
        self.writer = None
        self.subject_id = None
        self.data_path = data_path
        self.human = False

    # default args are for non-human agent
    def setup_subject(self, human=False, participant_id=-1, age=-1, gender='robot', handedness='none', eyewear='no', major='robotics'):
        self.human = human
        self.writer = SubjectWriter(self.data_path)
        self.subject_id = self.writer.subject_id

        print("Starting trials for subject {}. Saving to {}".format(self.subject_id, self.writer.subject_path))
        self.logger = SubjectLogger(subject_id=self.subject_id,
                                    participant_id=participant_id,
                                    age=age,
                                    gender=gender,
                                    handedness=handedness,
                                    eyewear=eyewear,
                                    major=major,
                                    start_time=time.time())

        # copy the entire code base; this is unnecessary but prevents worrying about a particular
        # source code version when trying to reproduce exact parameters
        copytree('./', self.writer.subject_path + '/src/', ignore=ignore_patterns('*.mp4',
                                                                                  '*.pyc',
                                                                                  '.git',
                                                                                  '.gitignore',
                                                                                  '.gitmodules'))

    def get_current_attempt_logged_actions(self, idx):
        results = self.logger.cur_trial.cur_attempt.results
        agent_idx = results[0].index('agent')
        actions = results[idx][agent_idx+1:len(results[idx])]
        action_labels = results[0][agent_idx+1:len(results[idx])]
        return actions, action_labels

    def get_current_attempt_logged_states(self, idx):
        results = self.logger.cur_trial.cur_attempt.results
        agent_idx = results[0].index('agent')
        # frame is stored in 0
        states = results[idx][1:agent_idx]
        state_labels = results[0][1:agent_idx]
        return states, state_labels

    def write_results(self):
        self.writer.write(self.logger, self)

    def write_trial(self, test_trial=False):
        self.writer.write_trial(self.logger, test_trial)

    def finish_trial(self, test_trial):
        self.logger.finish_trial()
        self.write_trial(test_trial)

    def finish_subject(self, strategy, transfer_strategy):
        self.logger.finish(time.time())
        self.logger.strategy = strategy
        self.logger.transfer_strategy = transfer_strategy

        self.write_results()