import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

with open('results/experiment4_random_cost_rrt_results.pkl', 'rb') as f:
    experiment4_random_cost_rrt_results = pickle.load(f)

with open('results/experiment4_random_cost_informed_results.pkl', 'rb') as f:
    experiment4_random_cost_informed_results = pickle.load(f)

keys = ['informed_rrt_star', 'rrt_star']
values = {
    'informed_rrt_star': experiment4_random_cost_informed_results,
    'rrt_star': experiment4_random_cost_rrt_results}
save_name = "results/experiment4_figure.pdf"
fig, ax = plt.subplots()
color = iter(cm.brg(np.linspace(0, 1, len(keys))))
ax.set_xlabel("Timestep")
ax.set_ylabel("Solution Cost")
lines = []


for key in keys:

    valuesList = values[key]
    numberOfExperiments = len(valuesList)
    maxLength = np.max(np.array
                       ([len(valueList) for valueList in valuesList]))
    x = np.arange(1, maxLength+1, 1)
    extendList = valuesList
    for i in range(len(valuesList)):
        extend = np.empty((maxLength,))
        extend[:] = np.nan
        valuesListNp_i = np.array(valuesList[i])
        extend[:valuesListNp_i.shape[0]] = valuesListNp_i
        extendList[i] = extend

    y = np.stack(extendList, axis=0)
    ci = 1.96 * np.nanstd(y, axis=0)/np.sqrt((numberOfExperiments))
    mean_y = np.nanmean(y, axis=0)

    c = next(color)
    line, = ax.plot(x, mean_y, color=c)
    lines.append(line)
    ax.fill_between(x, (mean_y-ci), (mean_y+ci), color=c, alpha=.1)

ax.legend(lines, keys)
fig.savefig(save_name, format="pdf", bbox_inches="tight")
