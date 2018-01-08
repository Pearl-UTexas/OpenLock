
import gym
from gym_lock.settings_scenario import select_scenario, select_random_scenarios
from gym_lock.session_manager import SessionManager
import time


# def exit_handler(signum, frame):
#    print 'saving results.csv'
#    np.savetxt('results.csv', env.results, delimiter=',', fmt='%s')
#    exit()


if __name__ == '__main__':

    # PARAMETERS: todo: make these command-line arguments

    # general params
    # training params
    params = {
        'data_dir': '../OpenLockResults/subjects',
        'num_trials': 6,
        'train_attempt_limit': 10,
        'train_action_limit': 3,
        'test_attempt_limit': 20,
        'test_action_limit': 3
    }

    train_scenario_name, test_scenario_name = select_random_scenarios()
    params['train_scenario_name'] = train_scenario_name
    params['test_scenario_name'] = test_scenario_name

    scenario = select_scenario(params['train_scenario_name'])
    env = gym.make('arm_lock-v0')
    # create session/trial/experiment manager
    manager = SessionManager(env, params)
    manager.update_scenario(scenario)

    for trial_num in range(0, params['num_trials']):
        manager.run_trial_human(params['train_scenario_name'], params['train_action_limit'], params['train_attempt_limit'])

    # testing trial
    # print "INFO: STARTING TESTING TRIAL"
    scenario = select_scenario(params['test_scenario_name'])
    manager.update_scenario(scenario)
    manager.run_trial_human(params['test_scenario_name'], params['test_action_limit'], params['test_attempt_limit'])

    manager.finish_subject()
    manager.write_results()
    print "Thank you for participating."
    time.sleep(3)

