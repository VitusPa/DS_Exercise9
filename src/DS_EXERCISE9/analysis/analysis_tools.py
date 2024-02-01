def analyse_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Produce and analyse raw data.

       Function fits 16th and 84th percentiles.

    Args:
        raw_data: a dataframe containing the raw data.

    Returns:
        pd.DataFrame containing fit results.
    """
    pct16 = []
    pct84 = []
    x_mean = []
    bins = np.arange(raw_data['x'].min(),raw_data['x'].max(), 100)
    for k in range(len(bins) -1):
        idx = (raw_data['x'] >= bins[k]) & (raw_data['x'] < bins[k+1])
        pct16.append(np.percentile(raw_data['y'][idx],16))
        pct84.append(np.percentile(raw_data['y'][idx],84))
        x_mean.append(np.mean(raw_data['x'][idx]))
    return pd.DataFrame({'pct16': pct16, 'pct84': pct84, 'x_mean': x_mean})