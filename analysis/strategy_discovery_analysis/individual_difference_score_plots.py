import matplotlib.pyplot as plt
import pandas as pd
from vars import habitual_pid, mf_pid, hybrid_pid, habitual_not_examined_all_pid, habitual_examined_all_pid

plt.figure(figsize=(8, 5))

# plot the average score of habitual_pid against mfhybrid_pid across each trial
# load the data
df = pd.read_csv('../../data/human/strategy_discovery/mouselab-mdp.csv')
df_habitual_not_all = df[df['pid'].isin(habitual_not_examined_all_pid)]
df_habitual_examined_all = df[df['pid'].isin(habitual_examined_all_pid)]
df_mf = df[df['pid'].isin(mf_pid)]
df_hybrid = df[df['pid'].isin(hybrid_pid)]
# df_mb = df[df['pid'].isin(mb_pid)]

df_habitual_not_all = df_habitual_not_all.groupby('trial_index').score.mean().reset_index()
df_mf = df_mf.groupby('trial_index').score.mean().reset_index()
df_hybrid = df_hybrid.groupby('trial_index').score.mean().reset_index()
df_habitual_examined_all = df_habitual_examined_all.groupby('trial_index').score.mean().reset_index()
# df_mb = df_mb.groupby('trial_index').score.mean().reset_index()

plt.plot(df_habitual_not_all['trial_index'], df_habitual_not_all['score'], label=f'Habitual participant who did not examine all, n={len(habitual_not_examined_all_pid)}', color='red')
plt.plot(df_mf['trial_index'], df_mf['score'], label=f'Model-free participant, n={len(mf_pid)}')
plt.plot(df_hybrid['trial_index'], df_hybrid['score'], label=f'Hybrid participant, n={len(hybrid_pid)}')
plt.plot(df_habitual_examined_all['trial_index'], df_habitual_examined_all['score'], label=f'Habitual participants who examined all, n={len(habitual_examined_all_pid)}')
# plt.plot(df_mb['trial_index'], df_mb['score'], label=f'Model-based, n={len(mb_pid)}')

plt.ylim(-160, 10)
plt.xlabel('Trial', fontsize=12)
plt.ylabel('Average Score', fontsize=12)
plt.legend(loc='lower right', fontsize=14)
plt.savefig(f'plots/cogsci2025/individual_difference_score_2.png')
plt.show()
plt.close()


