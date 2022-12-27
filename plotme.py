
""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                        helpers for efficiently creating seaborn graphics                                           |
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt 
import mplfinance as fplt
import data.dummy_data as dummy
import yfinance as yf
import plotly
import plotly.graph_objects as go
import talib
from talib import RSI, BBANDS, MACD
import waterfall_chart


sns.set(rc={'figure.figsize':(20,8)})
sns.set_style('whitegrid')
COLOR_PALETTE = sns.color_palette("Set2", 20)
COLOR_PALETTE_STRING = "Set2"
sns.set_palette(COLOR_PALETTE_STRING)


class Yax:
    """ Y-Axis
    """    

    def __init__(self, g):
        """[Summary]
        
        :param [g]: A matplotlib/seaborn graph object
        
        """     

        self.g = g


    def annotate(self, ax, annotation_series:pd.Series=None, precision=2):

        patch_labels = []
        for i in range(annotation_series.size): # data must have the same number of records for each annotation
            patch_labels.append(annotation_series[i]) 

        if isinstance(patch_labels[i], str):
            # annotate bar centering around half of it's height adjusting for the lenght of the text being written
            for i, p in enumerate(ax.patches):  
                    ax.annotate( patch_labels[i], (p.get_x() + (p.get_width() / 2.0), (p.get_height() - ( p.get_height() * (len(str(patch_labels[i])) + 1) / 100 ) )/ 2 ), ha = 'center', va = 'center', rotation = 90, size = 10 ) #  xytext = (1, 0), textcoords = 'offset points',

        elif isinstance(patch_labels[i], float):
            # annotate datapoint value atop the point
            if hasattr(self, 'divisor'):
                patch_labels = [(p/self.divisor) for p in patch_labels] # divisor is set by set_ylabels() which should be called first

            for i, p in enumerate(ax.patches):  
                ax.annotate( "{:,.1f}".format(patch_labels[i]), (p.get_x() + p.get_width() / 2., p.get_height() ), ha = 'center')


    def set_ylabels(self, ax, units=None, format_code = '{:,.2f}'):

        units_map = {None:1, 'K':1000, 'M':1000000, 'B':1000000000}

        divisor = units_map[units]

        self.divisor = divisor

        ylabels = [format_code.format(x) + str(units or '') for x in ax.get_yticks() / divisor]

        ax.set_yticklabels(ylabels)


    def set_ylim(self, lower, upper):

        plt.ylim([lower, upper]) # NOTE: Must specify ylim values before calling set_yabels


    def set_yticks(self, data):
        
        y = np.array([0.650, 0.660, 0.675, 0.685]) # TODO: calculate best 
        
        plt.yticks(np.arange(y.min(), y.max(), 0.005))



class Xax:
    """ X-Axis
    """    

    def __init__(self, g):
        """[Summary]
        
        :param [g]: A matplotlib/seaborn graph object
        
        """        

        self.g = g


    def color_tick_labels(self, ax):
       
        for ix, tick_label in enumerate(ax.get_xticklabels()):
            
            tick_label.set_color(matplotlib.colors.to_hex(COLOR_PALETTE[ix]))

            tick_label.set_fontsize("10")          


    def every_other(self, ax):
        
        for label in ax.xaxis.get_ticklabels()[::2]:
           
            label.set_visible(False)


    def rotate(self, ax, degrees):

        if ax is not None:

            ax.set_xticklabels( ax.get_xticklabels(), rotation=degrees, horizontalalignment='center')
        


class Plotme:

    def __init__(self, data, x, y, y2 = None) -> None:
        self.data = data
        self.x = x
        self.y = y
        self.y2 = y2


    def facet_grid(
        self,
        col=None,
        row=None,
        height = 5,
        map_type='line',
        axis_hue = None,
        annotation = None,
        **kwargs
    ):
        """[Summary]
        
        :param [hue]: a Seaborn property to color code plot objects
        :param [axis_hue]: a custom property used to color code x axis labels according to data groupings
        
        :raises [ErrorType]: 
        """    

        g = sns.FacetGrid(self.data, col = col, row = row, height = height, **kwargs)

        if map_type == 'line':
            g.map(sns.lineplot, self.x, self.y, palette=COLOR_PALETTE_STRING,  ci = False)
        
        elif map_type == 'bar':
            g.map(sns.barplot, self.x, self.y, palette=COLOR_PALETTE_STRING, dodge=True, ci = False)

        xax, yax = Xax(g), Yax(g)

        for ax in g.axes.flat:

            xax.rotate(ax, 90)

            if axis_hue == self.x:
                
                xax.color_tick_labels(ax)

            if len(self.data[self.x].unique()) > 10:
                
                xax.every_other(ax)

            if annotation is not None:
                yax.annotate(ax, annotation_series=self.data[annotation])

        g.set_titles(col_template = '{col_name}', row_template='{row_name}')

        plt.show()

        return g



    def facet_grid_dual_axis_line(
        self,
        col=None,
        row=None, 
        yax_units = 'M',
        hspace=0.7, 
        wspace=0.7,  
        **kwargs

    ):
        """[Summary]

        :param [data]: data is a flat table (row by columns); not melted to a tall table
        
        """
        def facetgrid_two_axes(*args, **kwargs):
            yax = Yax(g)

            data = kwargs.pop('data')
            kwargs.pop('dual_axis')
            kwargs.pop('color')
            
            ax = plt.gca()
            ax.plot(data[self.x],data[self.y], **kwargs, color='g',label = self.y)
            ax.set_ylabel(self.y, color='g')
            yax.set_ylabels(ax = ax, units = yax_units, format_code = '{:,.1f}')

            ax2 = ax.twinx()
            ax2.plot(data[self.x],data[self.y2], **kwargs, color='b', label = self.y2)
            ax2.set_ylabel(self.y2, color='b')
            yax.set_ylabels(ax = ax2, units = yax_units, format_code = '{:,.1f}')
        
        g = sns.FacetGrid(self.data, col=col, row = row, **kwargs)
        g.map_dataframe(facetgrid_two_axes, dual_axis=True)

        xax = Xax(g)
        for ax in g.axes.flat:

            xax.rotate(ax, 90)

            if len(self.data[self.x].unique()) > 10:
                
                xax.every_other(ax)

        g.set_titles(col_template = '{col_name}', row_template='{row_name}')

        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        plt.show()



    def clustered_bar(
        self,
        yax_units = 'M',
        **kwargs
    ):
        g = sns.barplot(data =self.data, x = self.x, y = self.y, **kwargs)

        g.set_title('Title')

        xax, yax = Xax(g), Yax(g) 
        
        yax.set_ylabels(ax = g.axes, units = yax_units)
        
        yax.annotate(ax=g.axes, annotation_series=dummy.sample2['value'])
        
        xax.rotate(ax=g.axes, degrees = 45)
        
        plt.show()

        return g



    def box_swarm_plot(
        self,
        box_kwargs,
        swarm_kwargs,
        **kwargs
    ):
        g = sns.boxplot(data=self.data, x = self.x, y = self.y, **box_kwargs)

        sns.swarmplot(data=self.data,  x = self.x, y = self.y, **swarm_kwargs)

        g.set_title('Title')

        yax = Yax(g)
        
        yax.set_ylim(-1, 1.5)

        yax.set_ylabels(ax = g.axes, units=None)

        plt.show()



    #FIXME
    def candle_stick(
        self,
        engine = 'fplt',
        
    ):
        sample6 =  yf.download("AMZN", start="2022-06-01", end="2022-10-30")

        if engine == 'fplt':
            sample6["SMA"] = talib.SMA(sample6.Close, timeperiod=3)
            sample6["RSI"] = talib.RSI(sample6.Close, timeperiod=3)
            sample6["EMA"] = talib.EMA(sample6.Close, timeperiod=3)
            rsi = fplt.make_addplot(sample6["RSI"], color="grey", width=1.5, ylabel="RSI", secondary_y=True, linestyle='dashdot')
            volume = fplt.make_addplot(sample6["Volume"], color="purple", panel=1)
            sma = fplt.make_addplot(sample6[["SMA", "EMA"]])
            ema = fplt.make_addplot(sample6["EMA"], color="dodgerblue", width=1.5)
            fplt.plot(
                        sample6,
                        type='candle',
                        style='charles',
                        title='Title',
                        ylabel='Price ($)',
                        addplot = [sma, ema, rsi, volume],
                        volume=True,
                        ylabel_lower='Shares\nTraded',            
                    )
            plt.show()
            plt.savefig("img/fplt_candle.png")

        elif engine == 'plotly':
            candlestick = go.Candlestick(x=sample6.index,
                                        open=sample6['Open'],
                                        high=sample6['High'],
                                        low=sample6['Low'],
                                        close=sample6['Close']
                                        )
            fig = go.Figure(data=[candlestick])
            fig.update_layout(
                width=800, height=600,
                title="Title",
                yaxis_title='Axis Title'
            )
            fig.update_layout(xaxis_rangeslider_visible=False)
            fig.write_image("img/fig1.png")
            fig.show()



    #FIXME
    def technical_analysis(
        self
    ):
        data =  yf.download("AMZN", start="2021-01-01", end="2022-10-30")

        print(data.iloc[-1])

        up, mid, low = BBANDS(data['Adj Close'], timeperiod=21, nbdevup=2, nbdevdn=2, matype=0)

        rsi = RSI(data['Adj Close'], timeperiod=14)

        macd, macdsignal, macdhist = MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)

        data = pd.DataFrame({'Price': data['Adj Close'], 'BB Up': up, 'BB Mid': mid, 'BB down': low, 'RSI': rsi, 'MACD': macd})

        fig, axes= plt.subplots(nrows=3, figsize=(15, 10), sharex=True)

        data.drop(['RSI', 'MACD'], axis=1).plot(ax=axes[0], lw=1, title='Bollinger Bands')

        data['RSI'].plot(ax=axes[1], lw=1, title='Relative Strength Index')

        axes[1].axhline(70, lw=1, ls='--', c='k')

        axes[1].axhline(30, lw=1, ls='--', c='k')

        data.MACD.plot(ax=axes[2], lw=1, title='Moving Average Convergence/Divergence', rot=0)

        axes[2].set_xlabel('')

        fig.tight_layout()

        sns.despine()

        plt.show()



    #FIXME
    def spider_plot(
        self
    ):

        feats = ['Service Games Won','Total Service Points Won', 'Return Games Won', 'Return Points Won', 'Break Points Converted']
        _sr = sr.copy()
        for c in _sr.columns:
            _sr[c] = [int(str(x).replace('%','')) for x in _sr[c]]
            
        p1 = _sr.transpose()[feats].iloc[0].values.tolist()
        p2 = _sr.transpose()[feats].iloc[1].values.tolist()
        p1 = [*p1, p1[0]]
        p2 = [*p2, p2[0]]
        feats = [*feats, feats[0]]

        label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(p1))

        plt.figure(figsize=(8, 5))
        plt.subplot(polar=True)
        plt.plot(label_loc, p1, label=self.p1_lname)
        plt.plot(label_loc, p2, label=self.p2_lname)
        plt.title(f'{self.p1_lname} vs. {self.p2_lname}', size=20)
        lines, labels = plt.thetagrids(np.degrees(label_loc), labels=feats)
        plt.legend()
        plt.show()



    #FIXME
    def waterfall(
        self
    ):
        a = ['mon','tue','wen','thu','fri','sat','sun']
        b = [10,-30,-7.5,-25,95,-7,45]
        waterfall_chart.plot(a, b, net_label='Total', rotation_value=360)
        plt.show()

""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Examples:                                                                                                         │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

# Box & Swarm Plot
# Plotme(data = dummy.sample1, x = 'variable', y = 'value').box_swarm_plot(box_kwargs = {'showfliers':False, 'orient':'v', 'color':"seagreen"}, swarm_kwargs= {'size':4, 'hue':'sector', "palette":"deep"})

# Clustered Bar Plot
# Plotme(data = dummy.sample2, x = 'calendardate', y = 'value').clustered_bar(ci=None, hue = 'industry', yax_units = 'B', dodge = True,  palette="Set2")

# Facet Grid Line
# Plotme(data = dummy.sample3, x = 'calendardate', y = 'value').facet_grid(col = 'variable', row = None, map_type='line', axis_hue = 'ticker', annotation = None, hue = 'ticker', sharex=False, sharey=False, margin_titles = True)

# Facet Grid Bar Hue X Axis
# Plotme(data = dummy.sample4, x = 'variable', y = 'value').facet_grid(col = 'ticker', row = None, map_type='bar', axis_hue = 'variable', annotation = 'value', sharex=False, sharey=False, margin_titles = True)

# Facet Grid Bar Date X Axis
# Plotme(data = dummy.sample5, x = 'calendardate', y = 'value').facet_grid(col = 'ticker', row = None, map_type='bar', axis_hue = 'variable', annotation = 'value', sharex=False, sharey=False, margin_titles = True)

# Facet Grid Dual Axis Lines
# Plotme(data = dummy.sample6, x = 'calendardate', y='assets', y2='liabilities').facet_grid_dual_axis_line( col = 'ticker', row = None, yax_units = 'B', height=5, col_wrap = 3, sharex = False, sharey=False)


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Todo                                                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

# def dual_axis():
#   ''' standard dual axis line chart '''
#   pass

# def derivative_exposures():
#   # option exposure; P/L
#   pass






