import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

with open('results/experiment2_gap_height_informed_results.pkl', 'rb') as f:
    experiment2_gap_height_informed_results = pickle.load(f)

with open('results/experiment2_gap_height_rrt_results.pkl', 'rb') as f:
    experiment2_gap_height_rrt_results = pickle.load(f)

keys = ['informed_rrt_star', 'rrt_star']
values = {
    'informed_rrt_star': experiment2_gap_height_informed_results,
    'rrt_star': experiment2_gap_height_rrt_results}
save_name = "results/experiment2_figure.png"
fig, ax = plt.subplots()
color = iter(cm.brg(np.linspace(0, 1, len(keys))))
ax.set_xlabel("Gap height as percentage of total wall height, hg/h (%)")
ax.set_ylabel("CPU Time(seconds)")
lines = []


gap_height_rates = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0]

for key in keys:

    valuesList = values[key]
    numberOfExperiments = len(valuesList)
    # maxLength = np.max(np.array
    #                    ([len(valueList) for valueList in valuesList]))
    maxLength = np.max(np.array
                       ([len(value) for key, value in valuesList.items()]))
    x = np.array(gap_height_rates)
    # extendList = np.array([None] * maxLength)
    extendList = np.array([None] * len(gap_height_rates))

    for i in range(len(gap_height_rates)):
        map_key = gap_height_rates[i]
        extend = np.empty((maxLength,))
        extend[:] = np.nan
        valuesListNp_i = np.array(valuesList[map_key])
        extend[:valuesListNp_i.shape[0]] = valuesListNp_i
        extendList[i] = extend

    y = np.stack(extendList, axis=0)
    # ci = 1.96 * np.nanstd(y, axis=0)/np.sqrt((numberOfExperiments))
    # mean_y = np.nanmean(y, axis=0)
    ci = 1.96 * np.nanstd(y, axis=1)/np.sqrt((numberOfExperiments))
    mean_y = np.nanmean(y, axis=1)

    c = next(color)
    line, = ax.plot(x, mean_y, color=c)
    plt.xlim(max(x), min(x))
    lines.append(line)
    ax.fill_between(x, (mean_y-ci), (mean_y+ci), color=c, alpha=.1)

ax.legend(lines, keys)
fig.savefig(save_name, format="png", bbox_inches="tight")
