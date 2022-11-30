from mcl_toolbox.utils.experiment_utils import Experiment


class ParticipantIterator:
    def __init__(self, participant, click_cost=1):
        self.participant = participant
        print(click_cost)
        self.click_cost = click_cost
        self.clicks = self.participant.clicks
        self.envs = self.participant.envs
        self.rewards = self.modify_scores(participant.scores, participant.clicks, participant.rewards_withheld)
        self.taken_paths = self.participant.paths
        self.strategies = self.participant.strategies
        self.temperature = self.participant.temperature
        self.current_trial = 0
        self.current_click = 0

    def modify_scores(self, scores, p_clicks, rewards_withheld):
        p_rewards = []
        for score, clicks, withheld in zip(scores, p_clicks, rewards_withheld):
            num_clicks = len(clicks) - 1
            total_click_cost = self.click_cost * num_clicks
            # Score includes subtracted click costs
            # Add back click costs so that termination reward does not reflect click costs
            adjusted_score = None if withheld else score + total_click_cost
            rewards = [-self.click_cost] * num_clicks + [adjusted_score]
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
            if len(self.taken_paths) > 0:
                taken_path = self.taken_paths[self.current_trial]
            else:
                taken_path = None
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
