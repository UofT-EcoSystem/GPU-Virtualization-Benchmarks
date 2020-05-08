import altair as alt
import pandas as pd
import numpy as np

import data.scripts.common.constants as const


def prepare_data(pair_series):
    runtime = pair_series['runtime']
    s1_runtime = runtime[1]
    s2_runtime = runtime[2]

    norm_runtime = pair_series['norm_ipc']
    s1_norm = norm_runtime[1]
    s2_norm = norm_runtime[2]

    def get_from_to(duration):
        time_series = [0]

        for d in duration:
            new_elem = time_series[-1] + d
            time_series.append(new_elem)

        time_series = np.array(time_series)
        return time_series[0:-1], time_series[1:]

    s1_from, s1_to = get_from_to(s1_runtime)
    s1 = np.repeat('1: ' + pair_series['1_bench'], s1_from.shape[0])
    k1 = np.arange(1, const.multi_kernel_app[pair_series['1_bench']] + 1)
    k1 = np.resize(k1, s1_from.shape[0])
    k1 = ['{}:{}'.format('1', k) for k in k1]

    s2_from, s2_to = get_from_to(s2_runtime)
    s2 = np.repeat('2: ' + pair_series['2_bench'], s2_from.shape[0])
    k2 = np.arange(1, const.multi_kernel_app[pair_series['2_bench']] + 1)
    k2 = np.resize(k2, s2_from.shape[0]).astype(str)
    k2 = ['{}:{}'.format('2', k) for k in k2]

    col_from = np.concatenate((s1_from, s2_from))
    col_to = np.concatenate((s1_to, s2_to))
    col_stream = np.concatenate((s1, s2))
    col_kernel = np.concatenate((k1, k2))
    col_norm = np.concatenate((s1_norm, s2_norm))

    data = pd.DataFrame()
    data["start"] = col_from
    data["end"] = col_to
    data["stream"] = col_stream
    data["kernel"] = col_kernel
    data['position'] = (data['start'] + data['end']) / 2
    data['norm'] = col_norm
    data['norm'] = data['norm'].round(2)

    return data


def draw_altair_timeline(pair_series, col_title):
    # 1. Prepare data
    data = prepare_data(pair_series)
    chart_title = "{0} = {1}".format(col_title, pair_series[col_title])

    # 2. Draw altair objects
    bars = alt.Chart(data, title=chart_title).mark_bar().encode(
        x=alt.X("start", title='Cycles'),
        x2="end",
        y=alt.Y("stream", sort=None),
        color=alt.Color('kernel', scale=alt.Scale(scheme='pastel1'),
                        legend=None)
    ).properties(
        width=900,
        height=100
    )

    text = alt.Chart(data).mark_text(
        align='center',
        baseline='middle',
        color='black',
        fontSize=12,
        angle=90,
    ).encode(
        y=alt.Y('stream', title=None),
        x='position',
        text='norm'
    )

    return alt.layer(bars, text).configure_axis(
        grid=False
    ).configure_header(
        titleColor='black',
        titleFontSize=14,
    )
