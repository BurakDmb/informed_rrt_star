import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

with open('results/experiment1_map_informed_results.pkl', 'rb') as f:
    experiment1_map_informed_results = pickle.load(f)

with open('results/experiment1_map_rrt_results.pkl', 'rb') as f:
    experiment1_map_rrt_results = pickle.load(f)

keys = ['informed_rrt_star', 'rrt_star']
values = {
    'informed_rrt_star': experiment1_map_informed_results,
    'rrt_star': experiment1_map_rrt_results}
save_name = "results/experiment1_figure.png"
fig, ax = plt.subplots()
color = iter(cm.brg(np.linspace(0, 1, len(keys))))
ax.set_xlabel("Map width as a factor of the" +
              " distance between start and goal, L/dgoal")
ax.set_ylabel("CPU Time(seconds)")
lines = []

map_width_rates = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]

for key in keys:

    valuesList = values[key]
    numberOfExperiments = len(valuesList)
    # maxLength = np.max(np.array
    #                    ([len(valueList) for valueList in valuesList]))
    maxLength = np.max(np.array
                       ([len(value) for key, value in valuesList.items()]))
    x = np.array(map_width_rates)
    # extendList = np.array([None] * maxLength)
    extendList = np.array([None] * len(map_width_rates))

    for i in range(len(map_width_rates)):
        map_key = map_width_rates[i]
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
    lines.append(line)
    ax.fill_between(x, (mean_y-ci), (mean_y+ci), color=c, alpha=.1)

ax.legend(lines, keys)
fig.savefig(save_name, format="png", bbox_inches="tight")
