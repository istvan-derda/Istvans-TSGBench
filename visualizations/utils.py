import matplotlib.pyplot as plt
import random
from pyprojroot import here


def ori_dataset_path(no):
    paths = list(here('data/ori').glob(f"D{no}_*/*train.pkl"))
    assert len(paths) == 1
    return paths[0]


def _read_dataset(dataset_path):
    
    data = np.load(str(dataset_path))

    file_name = dataset_path.name
    return data, file_name


def _plot_timeseries(timeseries, ax=None, channel_nos=None, labels=None, autoscale=False):
    channels = timeseries.T
    if ax == None:
        _, ax = plt.subplots()
    if channel_nos == None:
        channel_nos = range(len(channels))
    if labels == None:
        labels = channel_nos
    for i, channel_no in enumerate(channel_nos):
        channel = channels[channel_no]
        ax.plot(range(len(timeseries)), channel, label=labels[i])
        if not autoscale:
            ax.set_ylim([0, 1])


def plot_dataset(dataset_path, samplesize=3, indexes=None, labels=None, autoscale=False):
    dataset, file_name = _read_dataset(dataset_path)
    fig, axs = plt.subplots(samplesize, 1)
    for i in range(samplesize):
        j = random.randrange(0, len(dataset))
        _plot_timeseries(dataset[j], axs[i], indexes, labels, autoscale)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels)
    fig.suptitle(file_name)
