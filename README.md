# The Machine Learning Crypto Trading Package

An end-to-end solution for analyzing and trading in the crypto-currency market, based on traditional technique
indicators and machine learning techniques to help average investor find a customized quant strategy.

A pipeline was developed from raw trading data to technique indicators, machine learning models
(tune and combine indicators smartly) and back-testing performance. It generates buy-sell recommendation based
on back-testing results with historical precision and average returns related statistics. This project hope to help
investors improve performance without losing explicability.

It consists of the following components:

1. **DataGen**: Unified interface for data query and processing. Specific data objects are designed to 
handle various trading data for crypto assets. It contains utility functions to query data from 
different exchanges/data providers, and restore them in a standard OHLCV data format indexed by query
dictionary

2. **Env**: Built from 'spot', 'forward' and 'vol' asset env components, the env class is used to 
hold the data from different types of assets. In each env (component), we attach the corresponding
env configurations as guideline to postprocess and analyze the input data, performance statistical 
inference.

3. **Analyze**: Generate analysis from environments and provide interface functions for a large set 
of market signals (e.g. bear/bull, moving average crossing, and etc)

4. **Ledger**: Standardized holding class, which provides the basic logics of buy/sell and set aside
functions to track the changes on the asset holdings and cash

5. **Strategy**: Valuation engine that combines Env and Ledger. It takes the signals generated from
Env, and follow the strategy logic to update the Ledger. Back testing is provided to test the 
strategy against the historical data in the Env. New strategies be derived from the base class simply
and one can implement their own trading logic

6. **Execution**: It provides linkage with certain exchange API, which can execute the strategy based
on the ledger information

