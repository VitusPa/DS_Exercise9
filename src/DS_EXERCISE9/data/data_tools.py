def generate_data(n: int = 1000) -> pd.DataFrame:
    """Generate data points.

    Args:
        n: Number of datapoints. Defaults to 1000.

    Returns:
        dataframe with raw data.
    """
    np.random.seed(42)
    x = np.linspace(0,2500, n)
    noise_component = np.random.rand(n)
    y = (x + x*noise_component/3)
    return pd.DataFrame({'x': x, 'y': y})