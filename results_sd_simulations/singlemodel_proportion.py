import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# adaptive [2, 29, 36, 38, 39, 103, 105, 112, 115, 125, 147, 151, 165, 166, 171, 173, 174, 175, 181, 186, 203, 205, 215,
# 224, 234, 239, 241, 249, 251, 256, 259, 260, 261, 265, 271, 279, 281, 282, 289, 294]

# all_reward_data = []
# load all pkl files from dir strategy_discovery_data ending with _10.pkl
# for pid in [28, 35, 48, 63, 75, 195, 320, 335]:
# for pid in [13, 30, 62, 73, 78, 141, 176, 197, 276, 285, 303]:

# change figure size
plt.rcParams["figure.figsize"] = (7, 5)
fig, ax1 = plt.subplots()


# for pid in [13, 62, 78, 141]:
for pid in [13, 78, 141]:
# for pid in [141]:
    all_reward_data = []
    data = pd.read_pickle(f"test6/{pid}_3326_1000.pkl")

    # save them all in a list
    all_reward_data.append(data["r"])

    # flatten all_reward_data
    all_reward_data = [item for sublist in all_reward_data for item in sublist]

    # for list in list, replace 13, 14, 15 with 1 and 0 otherwise
    all_reward_data = [[1 if x in [13, 14, 15, 16] else 0 for x in sublist] for sublist in all_reward_data]

    # calculate the average proportion
    average_proportion = np.mean(all_reward_data, axis=0)

    # ### mk test of trend
    # import pymannkendall as mk
    # mk_results = mk.original_test(average_proportion)
    # print(pid)
    # print(mk_results)

    ## Create a second y-axis to plot the average_proportion
    # ax2 = ax1.twinx()

    # if pid != 141:
    #     # make it more transparent
    #     ax1.plot(average_proportion, label="Hybrid", color="blue", alpha=0.2)
    # else:
    # ax1.plot(average_proportion, label="Hybrid")

    # add average proportion as percentage of first trial and last trial in legend
    ax1.plot(average_proportion, label=f"Hybrid {average_proportion[0]*100:.1f}% to {average_proportion[-1]*100:.1f}%")


    ax1.set_ylabel("Proportion of optimal Strategy", fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis="y", labelsize=14)
    ax1.tick_params(axis="x", labelsize=14)

    ## Add 95% CI of proportion
    ci = 1.96 * np.std(all_reward_data, axis=0) / np.sqrt(len(all_reward_data))
    ax1.fill_between(range(len(average_proportion)), average_proportion - ci, average_proportion + ci, alpha=0.2)


    ### Load MF data
    all_reward_data = []
    data = pd.read_pickle(f"test7/{pid}_491_1000.pkl")

    # save them all in a list
    all_reward_data.append(data["r"])

    # flatten all_reward_data
    all_reward_data = [item for sublist in all_reward_data for item in sublist]

    # for list in list, replace 13, 14, 15 with 1 and 0 otherwise
    all_reward_data = [[1 if x in [13, 14, 15, 16] else 0 for x in sublist] for sublist in all_reward_data]

    # calculate the average proportion
    average_proportion = np.mean(all_reward_data, axis=0)

    ### mk test of trend
    # mk_results = mk.original_test(average_proportion)
    # print(pid)
    # print(mk_results)

    # if pid != 141:
    #     # make it more transparent
    #     ax1.plot(average_proportion, label="MF", color="orange", alpha=0.2)
    # else:
    # ax1.plot(average_proportion, label="MF")

    ax1.plot(average_proportion, label=f"MF {average_proportion[0]*100:.1f}% to {average_proportion[-1]*100:.1f}%")

    ## Add 95% CI of proportion
    ci = 1.96 * np.std(all_reward_data, axis=0) / np.sqrt(len(all_reward_data))
    ax1.fill_between(range(len(average_proportion)), average_proportion - ci, average_proportion + ci, alpha=0.2)

# remove repeating labels
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=16)
plt.legend(loc="upper left", fontsize=14)
plt.xlabel("Trials", fontsize=14)

plt.savefig(f"hybrid_mf_1000_simulations.png")
plt.show()
plt.close()
