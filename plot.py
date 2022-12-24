
""" 
  ┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │                        helpers for efficiently creating seaborn graphics                                           |
  └────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from matplotlib import dates
import seaborn as sns
from sqlalchemy import create_engine
import dummy_data as dummy

sns.set_theme(style="whitegrid")
sns.set_palette("Set2")
COLOR_PALETTE = sns.color_palette("Set2", 20)


class Yax:
    """ Y-Axis
    """    

    def __init__(self, g):
        """[Summary]
        
        :param [g]: A matplotlib/seaborn graph object
        
        """     

        self.g = g


    def annotate(self, ax, annotation_series:pd.Series=None):

        # TODO: needs an is_clustered parameter to distinguish between clustered plot annotations
        print(annotation_series)

        patch_labels = []
        for i in range(annotation_series.size): # data must have the same number of records for each annotation
            patch_labels.append(annotation_series[i]) 

        if isinstance(patch_labels[i], str):
            # annotate bar centering around half of it's height adjusting for the lenght of the text being written
            for i, p in enumerate(ax.patches):  
                    
                    ax.annotate( patch_labels[i], (p.get_x() + (p.get_width() / 2.0), (p.get_height() - ( p.get_height() * (len(str(patch_labels[i])) + 1) / 100 ) )/ 2 ), ha = 'center', va = 'center', rotation = 90, size = 10 ) #  xytext = (1, 0), textcoords = 'offset points',

        elif isinstance(patch_labels[i], float):
            # annotate datapoint value atop the point
            for i, p in enumerate(ax.patches):  
               
                ax.annotate( round(patch_labels[i], 2), (p.get_x() + p.get_width() / 2., p.get_height() ), ha = 'center',  textcoords = 'offset points')


    def set_ylabels(self, units=None, format_code = '{:,.2f}'):

        units_map = {None:1, 'M':1000000, 'B':1000000000}

        divisor = units_map[units]

        ylabels = [format_code.format(x) + str(units or '') for x in self.g.get_yticks() / divisor]

        self.g.set_yticklabels(ylabels)


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
        

        # plt.xticks(rotation = degrees, ha = 'center')


    def date_format(self, ax):
        
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d'))



def _facet_grid(
    g,
    data,
    x, 
    y,
    hue,
    annotation,
    palette = "Set2",
    legend = False,
):

    xax = Xax(g)
    
    yax = Yax(g)

    for ax in g.axes.flat:

        if legend:
            ax.legend(loc = 'best')

        xax.rotate(ax, 90)

        if hue == x:
            
            xax.color_tick_labels(ax)

        if len(data[x].unique()) > 10:
            
            xax.every_other(ax)

        if annotation is not None:
            yax.annotate(ax, annotation_series=data[annotation])

    g.set_titles(col_template = '{col_name}', row_template='{row_name}')

    color_palette = sns.color_palette(palette, 20)

    plt.show()

    return g


""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Examples:                                                                                                         │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """



# ...Examples...

# Box & Swarm Plot
# g = sns.boxplot(data=dummy.sample1, x = 'variable', y = 'value', showfliers = False, orient = 'v')
# sns.swarmplot(data=dummy.sample1, x = 'variable', y = 'value', color="black", size = 2 )
# g.set_title('Title')
# yax = Yax(g)
# yax.set_ylabels(units=None)
# plt.show()


# Clustered Bar Plot
# g = sns.barplot(data =dummy.sample2, x = 'calendardate', y = 'value', hue = None, palette="Set2")
# g.set_title('Title')
# xax, yax = Xax(g), Yax(g) 
# yax.set_ylabels(units = 'M')
# xax.rotate(ax=g.axes, degrees = 45)
# plt.show()


# # # Facet Grid Line
# g = sns.FacetGrid(data = dummy.sample3, col = 'variable', row = None, hue = 'ticker', height = 5, sharex=False, sharey=False, despine=True, margin_titles = True)
# g.map(sns.lineplot, 'calendardate', 'value',  palette="Set2")
# _facet_grid(g, data = dummy.sample3, x = 'calendardate', y = 'value',  hue = 'ticker', annotation = None)


# # Facet Grid Bar Hue X Axis
g = sns.FacetGrid(data = dummy.sample4, col = 'ticker', row = None, hue = 'variable', height = 5, sharex=False, sharey=False, despine=True, margin_titles = True)
g.map(sns.lineplot, 'variable', 'value',  palette="Set2")
_facet_grid(g, data = dummy.sample3, x = 'variable', y = 'value',  hue = 'variable', annotation = None)


# # Facet Grid Bar Date X Axis
# p = Plotme( data = dummy.sample5, x = 'calendardate', y = 'value', hue = 'variable', annotation=None, )
# p.facet_grid(col = 'ticker', row = None, map = 'bar', legend = True, dodge=True, title = 'Facet Grid of Line Charts')


## **kwargs
# g = sns.FacetGrid(data = dummy.sample5, col = 'ticker')
# g.map(sns.lineplot, 'calendardate',  'value')
# plt.show()




""" 
  ┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │ Todo                                                                                                             │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 """

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






