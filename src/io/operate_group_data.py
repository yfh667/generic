def slice_group_data(raw_group_data, start, end):
    """
    从 raw_group_data 中裁剪时间区间 [basicSa, end)
    """
    return {
        step: raw_group_data[step]
        for step in range(start, end)
        if step in raw_group_data
    }


def build_stationpair_group_data(station_a_series, station_b_series, start, end, gid_a=0, gid_b=1):
    """
    把两个地面站的可接入卫星时序，转换成 SatelliteViewer 需要的 group_data 格式。

    参数：
      station_a_series: {step: set(sat_id)}
      station_b_series: {step: set(sat_id)}
      start, end: 时间窗口，闭区间 [start, end]
      gid_a, gid_b: 在 group_data["groups"] 里对应的组号（默认 0/1）

    返回：
      {
        step: {
          "groups": {gid_a: set(...), gid_b: set(...)},
          "all_mentioned": set(...)
        },
        ...
      }
    """
    out = {}
    for step in range(start, end + 1):
        a_set = set(station_a_series.get(step, set()))
        b_set = set(station_b_series.get(step, set()))
        out[step] = {
            "groups": {
                gid_a: a_set,
                gid_b: b_set,
            },
            "all_mentioned": a_set | b_set,
        }
    return out
