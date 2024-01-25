def plot_analysis(raw_data: pd.DataFrame,
                  fit_results: pd.DataFrame) -> None:
    """Plot the results of the analysis.

    Args:
        raw_data: a dataframe containing the raw data.
        fit_results: a dataframe containing the results of the analysis.
    """
    ax = plt.subplot(111)
    ax.set_axisbelow(True)
    ax.scatter(raw_data['x'], raw_data['y'], label = 'raw data', color = 'grey', alpha = 0.5)
    ax.plot(fit_results['x_mean'], fit_results['pct16'], label = '16th percentile')
    ax.plot(fit_results['x_mean'], fit_results['pct84'], label = '84th percentile')
    ax.legend(frameon = False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('A Mock Scientific Result')
    ax.set_xlabel('x-variable [arb.]', size = 14)
    ax.set_ylabel('y-variable [arb.]', size = 14)
    ax.grid(True)