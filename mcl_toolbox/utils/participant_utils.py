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
        """

        Args:
            scores:
            p_clicks:

        Returns: a list containing the click costs and external reward, e.g. [-1, -1, 40]

        """
        p_rewards = []
        for score, clicks in zip(scores, p_clicks):
            total_click_cost = []
            click_cost_list = []
            num_clicks = len(clicks) - 1
            # if self.click_cost is a function
            if callable(self.click_cost): #for strategy discovery
                # click 1, 5, 9 have depth; 2, 6, 10 have depth 2; all others have depth 3
                replacement_dict = {0: 0, 1: 1, 5: 1, 9: 1, 2: 2, 6: 2, 10: 2, 3: 3, 4: 3, 7: 3, 8: 3, 11: 3, 12: 3}
                clicks_depth = [replacement_dict[click] for click in clicks]
                for click_depth in clicks_depth:
                    click_cost_list.append(-self.click_cost(click_depth))

                click_cost_list.pop() #remove the 0 cost for termination
                total_click_cost.append(score - sum(click_cost_list))
            else:
                total_click_cost = self.click_cost * num_clicks

            if callable(self.click_cost): #that is if strategy discovery
                rewards = [click_cost_list, total_click_cost]
                flattened_list = [item for sublist in rewards for item in sublist]
                p_rewards.append(flattened_list)

            else:
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
