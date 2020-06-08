import altair as alt
import pandas as pd
import numpy as np

import data.scripts.common.constants as const


def _prepare_data(pair_series):
    runtime = pair_series['runtime']
    s1_runtime = runtime[-2]
    s2_runtime = runtime[-1]

    norm_runtime = pair_series['norm_ipc']
    s1_norm = norm_runtime[-2]
    s2_norm = norm_runtime[-1]

    def get_from_to(duration):
        time_series = [0]

        for d in duration:
            new_elem = time_series[-1] + d
            time_series.append(new_elem)

        time_series = np.array(time_series)
        return time_series[0:-1], time_series[1:]

    def get_kernels(stream_id, length):
        bench = stream_id + '_bench'
        kernels = np.arange(1, const.get_num_kernels(pair_series[bench]) + 1)
        kernels = np.resize(kernels, length)
        kernels = ['{}:{}'.format(stream_id, k) for k in kernels]

        return kernels

    s1_from, s1_to = get_from_to(s1_runtime)
    s1 = np.repeat('1: ' + pair_series['1_bench'], s1_from.shape[0])
    k1 = get_kernels('1', s1_from.shape[0])

    s2_from, s2_to = get_from_to(s2_runtime)
    s2 = np.repeat('2: ' + pair_series['2_bench'], s2_from.shape[0])
    k2 = get_kernels('2', s2_from.shape[0])

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


# Required fields in pair_series: 1_bench, 2_bench, runtime, norm_ipc
def draw_altair_timeline(pair_series, col_title=None, title=None):
    # 1. Prepare data
    data = _prepare_data(pair_series)

    if col_title:
        chart_title = "{0} = {1}".format(col_title, pair_series[col_title])
    elif title:
        chart_title = "{0}".format(title)
    else:
        chart_title = ""

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
        y=alt.Y('stream', title=None, sort=None),
        x='position',
        text='norm'
    )

    return alt.layer(bars, text).configure_axis(
        grid=False
    ).configure_header(
        titleColor='black',
        titleFontSize=14,
    )
