# bitcoin-and-index-fund-prophet-time-series-modelling
Exploratory additive time series modelling using Prophet 
<h1> Bitcoin Time Series Modelling</h1>

The purpose of this project is to explore the application of the library Prophet on Bitcoin’s price action (from CryptoWatch API), to gain familiarity with the library and to create useful functions for both retrieving cryptocurrency data, and evaluating prophet models. The repository has two smaller .py files which were used to create the functions created, and the notebook consolidates all outcomes and findings into one easy to follow file.

Attempting to predict BTC or index fund market price using Prophet for real money is not the aim here. These time series' have only been selected as an interesting case study/thought experiment which also exemplifies the strengths and weaknesses of Prophets models. Despite the lack of actionable insights, the project has achieved its goal in providing a welcome introduction to FBProphet and useful functions for future time series analysis. 

<h3> Key Outcomes</h3>
<ul>
<li>Built foundational methodology of using Prophet for additive time series analysis  </li>
<li>Created a function to call cryptocurrnecy price data at specified granularity using CryptoWatch API </li>
<li>Created a function to find rolling Pearson’s correlation coefficient of two time series' over specified timeframe  </li>
<li>Evaluated Prophet change point relevancy to Google trends interest </li>
<li>Created a function to compare different Prophet models via changepoint_prior_scale hyper parameter tuning </li>
<li>Evaluated the efficacy of Prophet models using the rolling pearsons correlation coefficient function</li>
<li>Used the yahoo finance library to call historical index function price action</li>

    
