from mcl_toolbox.utils.experiment_utils import Experiment


class ParticipantIterator:
    def __init__(self, participant, click_cost=1):
        self.participant = participant
        self.click_cost = click_cost
        self.clicks = self.participant.clicks
        self.envs = self.participant.envs
        self.rewards = self.modify_scores(participant.scores, participant.clicks)
        self.taken_paths = self.participant.paths
        self.strategies = self.participant.strategies
        self.temperature = self.participant.temperature
        self.current_trial = 0
        self.current_click = 0

    def modify_scores(self, scores, p_clicks):
        p_rewards = []
        for score, clicks in zip(scores, p_clicks):
            num_clicks = len(clicks) - 1
            total_click_cost = self.click_cost * num_clicks
            rewards = [-self.click_cost] * num_clicks + [score + total_click_cost]
            p_rewards.append(rewards)
        return p_rewards

    def get_click(self):
        click_num = self.current_click
        trial_num = self.current_trial
        return self.clicks[trial_num][click_num]

    def make_click(self):
        done = False
        reward = self.rewards[self.current_trial][self.current_click]
        taken_path = None
        self.current_click += 1
        if self.current_click == len(self.participant.clicks[self.current_trial]):
            done = True
            taken_path = self.taken_paths[self.current_trial]
            self.current_trial += 1
            self.current_click = 0
        return reward, taken_path, done

    def get_trial_path(self):
        return self.taken_paths[self.current_trial]

    def reset(self):
        self.current_click = 0
        self.current_trial = 0


if __name__ == "__main__":
    E = Experiment("v1.0")
    p = E.participants[0]
    pi = ParticipantIterator(p)
    print(pi.clicks)
    print(pi.get_click())
    print(pi.make_click())
    print(pi.get_click())
