def slice_group_data(raw_group_data, start, end):
    """
    从 raw_group_data 中裁剪时间区间 [basicSa, end)
    """
    return {
        step: raw_group_data[step]
        for step in range(start, end)
        if step in raw_group_data
    }