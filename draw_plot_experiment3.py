import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

with open('results/experiment3_tolerance_informed_results.pkl', 'rb') as f:
    experiment3_tolerance_informed_results = pickle.load(f)

with open('results/experiment3_tolerance_rrt_results.pkl', 'rb') as f:
    experiment3_tolerance_rrt_results = pickle.load(f)

keys = ['informed_rrt_star', 'rrt_star']
values = {
    'informed_rrt_star': experiment3_tolerance_informed_results,
    'rrt_star': experiment3_tolerance_rrt_results}
save_name = "results/experiment3_figure.png"
fig, ax = plt.subplots()
color = iter(cm.brg(np.linspace(0, 1, len(keys))))
ax.set_xlabel("Target path cost as a percentage" +
              " above optimal cost, cbest/c* (%)")
ax.set_ylabel("CPU Time(seconds)")
lines = []


optimal_percentages = [
    3.0, 2.8, 2.6, 2.4, 2.2, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0]

for key in keys:

    valuesList = values[key]
    numberOfExperiments = len(valuesList)
    maxLength = np.max(np.array
                       ([len(value) for key, value in valuesList.items()]))
    x = np.array(optimal_percentages)
    extendList = np.array([None] * len(optimal_percentages))

    for i in range(len(optimal_percentages)):
        map_key = optimal_percentages[i]
        extend = np.empty((maxLength,))
        extend[:] = np.nan
        valuesListNp_i = np.array(valuesList[map_key])
        extend[:valuesListNp_i.shape[0]] = valuesListNp_i
        extendList[i] = extend

    y = np.stack(extendList, axis=0)
    ci = 1.96 * np.nanstd(y, axis=1)/np.sqrt((numberOfExperiments))
    mean_y = np.nanmean(y, axis=1)

    c = next(color)
    line, = ax.plot(x, mean_y, color=c)
    plt.xlim(max(x), min(x))
    lines.append(line)
    ax.fill_between(x, (mean_y-ci), (mean_y+ci), color=c, alpha=.1)

ax.legend(lines, keys)
fig.savefig(save_name, format="png", bbox_inches="tight")
