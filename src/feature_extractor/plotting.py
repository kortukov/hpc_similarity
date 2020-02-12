def plot_time_series_list(time_series_list):
    fig = time_series_list[0].plot(x='t', y='y', label=time_series_list[0].name)
    for ts in time_series_list[1:]:
        ts.plot(ax=fig, x='t', y='y', label=ts.name)
    return fig


def scatter_time_series_list(time_series_list):
    fig = time_series_list[0].plot.scatter(x='t', y='y', label=time_series_list[0].name)
    for ts in time_series_list[1:]:
        ts.plot.scatter(ax=fig, x='t', y='y', label=ts.name)
    return fig


def plot_time_series_dict_together(time_series_dict):
    fig = None
    for sensor_name, time_series in time_series_dict.items():
        if fig is None:
            fig = time_series.plot(x='t', y='y', label=sensor_name)
        else:
            time_series.plot(ax=fig, x='t', y='y', label=sensor_name)
    return fig


# These are most useful


def plot_time_series_dict_subplots(time_series_dict):
    for sensor_name, time_series in time_series_dict.items():
        time_series.plot(x='t', y='y', label=sensor_name)


def plot_time_series_with_change_points(time_series_dict, change_points_times):
    for sensor_name, time_series in time_series_dict.items():
        fig = time_series.plot(x='t', y='y', label=sensor_name)
        for cpt in change_points_times:
            fig.axvline(cpt)
    return fig


def plot_change_points_on_figure(change_points_times, figure):
    new_figure = figure
    for cpt in change_points_times:
        new_figure.axvline(cpt)
    return new_figure.figure


def plot_superstructure(superstructure):
    fig = superstructure.structures[0].get_df().plot(x='t', y='y')
    for subregion in superstructure.structures[1:]:
        subregion.get_df().plot(ax=fig, x='t', y='y')
        return fig


def plot_superstructure_on_fig(superstructure, fig):
    superstructure.structures[0].get_df().plot(ax=fig, x='t', y='y')
    for subregion in superstructure.structures[1:]:
        subregion.get_df().plot(ax=fig, x='t', y='y')
    return fig
