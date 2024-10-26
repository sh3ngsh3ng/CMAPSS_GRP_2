## Dataset
- The engine is operating normally at the start of each time series and develops a fault at some point during the series. In the training dataset, the fault grrows in magnitude until system failure. In the test set, the time series ends some time prior to system failure.
- Objective: Predict the number of remaining operational cycles before failure in the test set. I.e. the number of operational cycles after the last cycle that the engine will continue to operate.
- A vector of true RUL values for the test data is provided
- Dataset has 26 columns.
- Reference: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data

## Setting up TSFRESH
- pip install tsfresh
- Reference: https://tsfresh.readthedocs.io/en/latest/text/quick_start.html#quick-start-label