
def compute_route_reliability_series(
    df,
    *,
    p_intra=0.999,
    p_inter=0.99,
    intra_col="intra_hops",
    inter_col="inter_hops",
    name=None,
):
    """
    只计算端到端稳定概率，不做绘图，不修改 df。
    返回一个 pandas Series。
    """
    import pandas as pd

    reliability = (float(p_intra) ** df[intra_col].astype(float)) * \
                  (float(p_inter) ** df[inter_col].astype(float))

    if name is None:
        name = f"reliability_pi{p_intra}_pe{p_inter}"

    reliability = pd.Series(reliability, index=df.index, name=name)
    return reliability


