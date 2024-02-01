import os.path

# This will produce the path to the test data on any OS and machine,
# if run inside unit_tests.py

# Strictly needed
TEST_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", 'tests', 'test_data')
)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np




def generate_data(n: int = 1000) -> pd.DataFrame:
    
      np.random.seed(42)
      x = np.linspace(0,2500, n)
      noise_component = np.random.rand(n)
      y = (x + x*noise_component/3)
      return pd.DataFrame({'x': x, 'y': y})

def test_data_generator():
    

 raw_data=generate_data()
 #assert raw_data==pd.read_parquet(r'C:\Users\vitus\OneDrive\Desktop\DataScience\DS_Exercise9\tests\test_data\raw_data.parquet')
 pd.testing.assert_frame_equal(raw_data, pd.read_parquet(r'C:\Users\vitus\OneDrive\Desktop\DataScience\DS_Exercise9\tests\test_data\raw_data.parquet'))




def analyse_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    
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

def test_analyse_data():
    
    fit_results = analyse_data(pd.read_parquet(r'C:\Users\vitus\OneDrive\Desktop\DataScience\DS_Exercise9\tests\test_data\raw_data.parquet'))
    pd.testing.assert_frame_equal(fit_results, pd.read_parquet(r'C:\Users\vitus\OneDrive\Desktop\DataScience\DS_Exercise9\tests\test_data\fit_results.parquet'))


def test_full_analysis():
    def plot_analysis(raw_data: pd.DataFrame,
                  fit_results: pd.DataFrame) -> None:
    
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
    
    def generate_data(n: int = 1000) -> pd.DataFrame:
    
      np.random.seed(42)
      x = np.linspace(0,2500, n)
      noise_component = np.random.rand(n)
      y = (x + x*noise_component/3)
      return pd.DataFrame({'x': x, 'y': y})
    
    def analyse_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    
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
    
    raw_data = generate_data()
    fit_results = analyse_data(raw_data)
    plot_analysis(raw_data = raw_data, fit_results = fit_results)
    
    



