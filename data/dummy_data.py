import pandas as pd
from sqlalchemy import create_engine


def query():

    # Build dummy data
    engine = create_engine('sqlite:///C:\data\industry_fundamentals.db', echo=False)
    cnxn = engine.connect()
    df = pd.read_sql("select * from CompFunBase", cnxn)[['ticker','name','calendardate','sector','industry','revenue', 'fcfmargin','eps','oppmargin','profitmargin', 'netmargin', 'pe','roc', 'roe','assets', 'liabilities']].dropna(axis=0)
    df = df.sort_values(by=['sector','industry','revenue'], ascending = False)
    industry_leaders = df.drop_duplicates(subset=['industry'], keep = 'first')
    df = df[df.ticker.isin(industry_leaders.ticker)]
    df.to_csv('./data/equity_fundamentals_revenue_leaders.csv')
    print(df)


    df = pd.read_csv('./data/equity_fundamentals_revenue_leaders.csv')

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

    melt = df.melt(id_vars = ['ticker', 'name', 'calendardate', 'sector', 'industry'], value_vars = ['fcfmargin','eps','oppmargin','profitmargin', 'netmargin', 'roc', 'roe', 'assets', 'liabilities'])

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

    print(melt)

    print(melt.variable.unique())

    return melt

# query()

df = pd.read_csv('./data/equity_fundamentals_revenue_leaders.csv')

melt = df.melt(id_vars = ['ticker', 'name', 'calendardate', 'sector', 'industry'], value_vars = ['fcfmargin','eps','oppmargin','profitmargin', 'netmargin', 'roc', 'roe', 'revenue', 'assets', 'liabilities'])

# ...Sample Data...

melt['calendardate'] = pd.to_datetime(melt['calendardate'], format = '%Y-%m-%d %H:%M:%S' ) # infer_datetime_format=True

sample1 = melt[(melt.calendardate == '2022-09-30') & (melt.variable.isin(['fcfmargin','oppmargin','profitmargin', 'netmargin']))]

sample2 = melt[ (melt.sector == 'Utilities') & (melt.calendardate > pd.to_datetime('2020-09-30')) & (melt.variable == 'revenue') ]
sample2 = sample2[sample2.ticker != 'CEG']
# sample2.calendardate = sample2.calendardate.apply(lambda x : x.strftime('%Y-%m-%d'))
sample2 = sample2.sort_values(by = ['ticker', 'calendardate'], ascending=True).reset_index(drop=True)
sample2['calendardate'] =  sample2['calendardate'].apply(lambda x : x.strftime('%Y-%m-%d'))

sample3 = melt[ (melt.sector == 'Utilities') & (melt.calendardate > pd.to_datetime('2015-03-31')) & (melt.variable.isin(['fcfmargin','oppmargin','profitmargin', 'netmargin'])) ]
sample3 = sample3[sample3.ticker != 'CEG']
# sample3.calendardate = sample3.calendardate.apply(lambda x : x.strftime('%Y-%m-%d'))
sample3 = sample3.sort_values(by = ['ticker', 'calendardate'], ascending=True).reset_index(drop=True)

sample4 = melt[ (melt.sector == 'Utilities') & (melt.calendardate == pd.to_datetime('2022-03-31')) & (melt.variable.isin(['fcfmargin','oppmargin','profitmargin', 'netmargin'])) ]
sample4 = sample4[sample4.ticker != 'CEG']
# sample4.calendardate = sample4.calendardate.apply(lambda x : x.strftime('%Y-%m-%d'))
sample4 = sample4.sort_values(by = ['ticker', 'calendardate'], ascending=True).reset_index(drop=True)

sample5 = melt[ (melt.sector == 'Utilities') & (melt.calendardate > pd.to_datetime('2021-12-31')) & (melt.variable.isin(['fcfmargin','oppmargin','profitmargin', 'netmargin'])) ]
sample5 = sample5[sample5.ticker != 'CEG']
# sample4.calendardate = sample4.calendardate.apply(lambda x : x.strftime('%Y-%m-%d'))
sample5 = sample5.sort_values(by = ['ticker', 'calendardate'], ascending=True).reset_index(drop=True)
sample5['calendardate'] =  sample5['calendardate'].apply(lambda x : x.strftime('%Y-%m-%d'))

sample6 = melt[ (melt.sector == 'Utilities') & (melt.calendardate > pd.to_datetime('2021-12-31')) & (melt.variable.isin(['assets', 'liabilities'])) ]
sample6['calendardate'] =  sample6['calendardate'].apply(lambda x : x.strftime('%Y-%m-%d'))
assets = sample6[sample6.variable == 'assets'].set_index(['ticker','name', 'calendardate','sector','industry',]).rename(columns = {'value':'assets'}).drop('variable', axis=1)
liabilities = sample6[sample6.variable == 'liabilities'].set_index(['ticker','name', 'calendardate','sector','industry']).rename(columns = {'value':'liabilities'}).drop('variable', axis=1)
sample6 = assets.merge(liabilities, how = 'outer', left_index=True, right_index=True).reset_index()
sample6.sort_values(by = ['calendardate','ticker'], ascending = True, inplace = True)
