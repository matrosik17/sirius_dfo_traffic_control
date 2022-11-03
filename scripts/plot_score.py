import sys
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models.optimize import OptimizeResponseBody


if __name__ == "__main__":
    optimization_result_json = json.load(sys.stdin)
    opt_response = OptimizeResponseBody.parse_obj(optimization_result_json)

    times = opt_response.local_search_result.times
    scores = opt_response.local_search_result.scores

    fig, ax = plt.subplots(nrows=1, figsize=(18, 4))
    time = np.array(times)
    scores = pd.Series(scores)
    best_scores = scores.cummax()

    ax.plot(time, scores, label="score локального поиска")
    ax.plot(time, best_scores, label="Лучший текущий score")

    ax.set_title("Динамика оптимизации")
    ax.set_ylabel("Score")
    ax.set_xlabel("Время, с")
    ax.legend()
    ax.grid()

    # period = 50
    # max_mp_iter = 10
    # num_periods = int(np.ceil(len(scores) / period))
    # for i in range(num_periods):
    #     ax.axvline(times[i * period], color='r')
    #     try:
    #         ax.axvline(times[i * period + max_mp_iter], color='r')
    #     except IndexError:
    #         pass

    plt.show()