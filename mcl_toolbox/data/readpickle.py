import pandas as pd
import pickle


hybrid_ssl = pd.read_pickle("hybrid_ssl_features.pkl")
non_learning = pd.read_pickle("non_learning_features.pkl") #non learning
mf = pd.read_pickle("model_free_habitual_features.pkl") #non learning
# set_difference = set(sd_features) - set(all)
print(2)


## Remove feature 49 'trial_level_std' from microscope_features.pkl
# mb = hybrid_ssl.copy()
#
#
# ## Remove the specific item  from the new list
# mb.remove("max_uncertainty")
# mb.remove("uncertainty")
# mb.remove("successor_uncertainty")
# mb.remove("trial_level_std")
# mb.remove("get_level_observed_std")
# #
# # # # remove the habitual features
# mb.remove("constant")
# mb.remove("level_count")
# mb.remove("num_clicks_adaptive")
# mb.remove("branch_count")
# mb.remove("click_count")
#
#
# # # # add feature: termination_after_observing_positive_inner_and_one_outer
# # # all = all_features.copy()
# # # all.append("avoid_first_level")
# # # all.append("avoid_third_level")
# #
# #
# # # # remove all the other termination features
# # # all = all_features.copy()
# # # all.remove("termination_after_at_most_4")
# # # all.remove("termination_not_before_2")
# # # all.remove("terminate_not_before_examined_one_outer_node")
# #
# #
# # ## save as new pickle file
# with open('non_learning_features.pkl', 'wb') as file:
#     pickle.dump(mb, file)


