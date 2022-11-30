
""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                        helper functions for efficiently creating seaborn graphics                                  |
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure

import seaborn as sns
sns.set_theme(style="whitegrid")


from sqlalchemy import create_engine

# Build dummy data
# engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
# cnxn = engine.connect()
# df = pd.read_sql("select * from CompFunBase", cnxn)[['ticker','name','calendardate','sector','industry','revenue', 'fcfmargin','eps','oppmargin','profitmargin', 'netmargin', 'pe','roc', 'roe',]].dropna(axis=0)
# df = df.sort_values(by=['sector','industry','revenue'], ascending = False)
# industry_leaders = df.drop_duplicates(subset=['industry'], keep = 'first')
# df = df[df.ticker.isin(industry_leaders.ticker)]
# df.to_csv('./equity_fundamentals_revenue_leaders.csv')
# print(df)

df = pd.read_csv('./equity_fundamentals_revenue_leaders.csv')

#       ticker                              name calendardate           sector                     industry       revenue  fcfmargin   eps  oppmargin  profitmargin  netmargin       pe       roc       roe
# 21383    CEG         CONSTELLATION ENERGY CORP   2022-09-30        Utilities        Utilities - Renewable  6.051000e+09  -0.245249 -0.57  -0.006776     -0.031069   0.036688 -180.068 -0.011651 -0.017436
# 21381    CEG         CONSTELLATION ENERGY CORP   2022-03-31        Utilities        Utilities - Renewable  5.591000e+09   0.168306  0.32   0.077804      0.018959   0.125022   26.477  0.006255  0.009526
# 21382    CEG         CONSTELLATION ENERGY CORP   2022-06-30        Utilities        Utilities - Renewable  5.465000e+09  -0.087466 -0.34   0.049771     -0.020311   0.100823   29.048 -0.006988 -0.010097
# 11863    AWK  AMERICAN WATER WORKS COMPANY INC   2021-09-30        Utilities  Utilities - Regulated Water  1.092000e+09   0.026557  1.53   0.453297      0.254579   0.600733   40.210  0.015418  0.040489
# 11884    AWK  AMERICAN WATER WORKS COMPANY INC   2022-09-30        Utilities  Utilities - Regulated Water  1.082000e+09  -0.064695  1.63   0.463956      0.274492   0.615527   17.952  0.015093  0.038283
# ...      ...                               ...          ...              ...                          ...           ...        ...   ...        ...           ...        ...      ...       ...       ...
# 77683    NTR                       NUTRIEN LTD   2018-09-30  Basic Materials          Agricultural Inputs  3.990000e+09  -0.132581 -1.70  -0.364912     -0.261654   0.289474  123.331 -0.029491 -0.047379
# 77684    NTR                       NUTRIEN LTD   2018-12-31  Basic Materials          Agricultural Inputs  3.875000e+09   0.354581  5.13   0.140387      0.826323   0.324903    8.089  0.095162  0.131095
# 77685    NTR                       NUTRIEN LTD   2019-03-31  Basic Materials          Agricultural Inputs  3.719000e+09  -0.226674  0.07   0.050282      0.011024   0.262167    8.881  0.001190  0.001730
# 77681    NTR                       NUTRIEN LTD   2018-03-31  Basic Materials          Agricultural Inputs  3.666000e+09  -0.157665  0.00   0.040371     -0.000273   0.231042  171.885 -0.000029 -0.000043
# 77688    NTR                       NUTRIEN LTD   2019-12-31  Basic Materials          Agricultural Inputs  3.503000e+09   0.520411 -0.08   0.043677     -0.013703   0.306880   27.668 -0.001413 -0.002099

melt = df.melt(id_vars = ['ticker', 'name', 'calendardate', 'sector', 'industry'], value_vars = ['fcfmargin','eps','oppmargin','profitmargin', 'netmargin', 'roc', 'roe'])

#       ticker                       name calendardate           sector               industry variable         value
# 0        CEG  CONSTELLATION ENERGY CORP   2022-09-30        Utilities  Utilities - Renewable  revenue  6.051000e+09
# 1        CEG  CONSTELLATION ENERGY CORP   2022-03-31        Utilities  Utilities - Renewable  revenue  5.591000e+09
# 2        CEG  CONSTELLATION ENERGY CORP   2021-03-31        Utilities  Utilities - Renewable  revenue  5.559000e+09
# 3        CEG  CONSTELLATION ENERGY CORP   2021-12-31        Utilities  Utilities - Renewable  revenue  5.532000e+09
# 4        CEG  CONSTELLATION ENERGY CORP   2022-06-30        Utilities  Utilities - Renewable  revenue  5.465000e+09
# ...      ...                        ...          ...              ...                    ...      ...           ...
# 12843    NTR                NUTRIEN LTD   2019-12-31  Basic Materials    Agricultural Inputs      rnd  0.000000e+00
# 12844    NTR                NUTRIEN LTD   2017-09-30  Basic Materials    Agricultural Inputs      rnd  0.000000e+00
# 12845    NTR                NUTRIEN LTD   2017-06-30  Basic Materials    Agricultural Inputs      rnd  0.000000e+00
# 12846    NTR                NUTRIEN LTD   2017-03-31  Basic Materials    Agricultural Inputs      rnd  0.000000e+00
# 12847    NTR                NUTRIEN LTD   2017-12-31  Basic Materials    Agricultural Inputs      rnd  0.000000e+00

# print(melt)

class Plotme:

    def __init__(self,  data, x, y, figsize=(8, 5)):

        self.data = data

        self.x = x

        self.y = y

        self.figsize = figsize

        

    def box_plot(self, title, swarm = False):
        
        fig = figure(figsize=self.figsize, dpi=80)

        g = sns.boxplot(data=self.data, x = self.x, y = self.y, palette="Set2", showfliers  = False, orient = 'v')

        if swarm:
            sns.swarmplot(data=self.data, x = self.x, y = self.y, color="black", size = 2 )

        g.set_title(title)
        # chart.set_xticklabels(chart.get_xticklabels(), rotation=10)
        g.tick_params(labelrotation=10)

        # if unit_divisor == 1000000000:
        #     ylabels = ['{:,.0f}'.format(x) + 'B' for x in chart.get_yticks()/unit_divisor]

        # chart.set_yticklabels(ylabels)

        plt.show()

        return g

p = Plotme( data = melt[melt.calendardate == '2022-09-30'][-200:], x = 'variable', y = 'value')
p.box_plot(title = ' a title', swarm = True)



# def clustered_bar():
#   # Clustered bar with horizontal labels; 
#   pass


# def facet_grid():
#   # Facetgrid (param: melted) color coded axis labels/bars;
#   pass


# def waterfall():
#   pass


# def candle_stick():
#   pass # Bokeh


# def equity_technicals():
#   # figure with subplots for candlestick chart; rsi; macd; bollinger bands; volume
#   pass


# def spider_web():
#   pass


# def dual_axis_facet_grid():
#   ''' dual axis line chart in a seaborn facet grid'''
#   pass


# def dual_axis():
#   ''' standard dual axis line chart '''
#   pass


# def exposures():
#   # option exposure; P/L
#   pass






