"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module containing utility functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.patheffects as path_effects
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from PIL import Image
from scipy.stats import poisson

class FontManager:
    """Utility to load fun fonts from https://fonts.google.com/ for matplotlib.
    Find a nice font at https://fonts.google.com/, and then get its corresponding URL
    from https://github.com/google/fonts/.
    The FontManager is taken from the ridge_map package by Colin Carroll (@colindcarroll).
    Parameters
    ----------
    url : str, default is the url for Roboto-Regular.ttf
        Can really be any .ttf file, but probably looks like
        'https://github.com/google/fonts/blob/master/ofl/cinzel/static/Cinzel-Regular.ttf?raw=true'
        Note make sure the ?raw=true is at the end.
    Examples
    --------
    >>> from mplsoccer import FontManager
    >>> import matplotlib.pyplot as plt
    >>> font_url = 'https://github.com/google/fonts/blob/master/ofl/abel/Abel-Regular.ttf?raw=true'
    >>> fm = FontManager(url=font_url)
    >>> fig, ax = plt.subplots()
    >>> ax.text("Good content.", fontproperties=fm.prop, size=60)
    """

    def __init__(self,
                 url=('https://github.com/google/fonts/blob/master/'
                      'apache/roboto/static/Roboto-Regular.ttf?raw=true')):
        self.url = url
        with NamedTemporaryFile(delete=False, suffix=".ttf") as temp_file:
            temp_file.write(urlopen(self.url).read())
            self._prop = fm.FontProperties(fname=temp_file.name)

    @property
    def prop(self):
        """Get matplotlib.font_manager.FontProperties object that sets the custom font."""
        return self._prop

    def __repr__(self):
        return f'{self.__class__.__name__}(font_url={self.url})'

def add_image(image, fig, left, bottom, width=None, height=None, **kwargs):
    """
    -----> The method is taken from mplsoccer package (from github) <-----
    -----> Andy Rowlinson(@numberstorm) <-----
    Adds an image to a figure using fig.add_axes and ax.imshow
    Args:
        image (str): image path.
        fig (matplotlib.figure.Figure): figure object
        left (float): The left dimension of the new axes.
        bottom (float): The bottom dimension of the new axes.
        width (float, optional): The width of the new axes. Defaults to None.
        height (float, optional): The height of the new axes. Defaults to None.
        **kwargs: All other keyword arguments are passed on to matplotlib.axes.Axes.imshow.
    Returns:
        matplotlib.figure.Figure: figure object.
    """    
    ## open image
    image = Image.open(image)

    ## height, width, channel of shape
    shape = np.array(image).shape
    
    image_height, image_width =  shape[0], shape[1]
    image_aspect = image_width / image_height
    
    figsize = fig.get_size_inches()
    fig_aspect = figsize[0] / figsize[1]
    
    if height is None:
        height = width / image_aspect * fig_aspect
    
    if width is None:   
        width = height*image_aspect/fig_aspect
    
    ## add image
    ax_image = fig.add_axes((left, bottom, width, height))
    ax_image.axis('off')  # axis off so no labels/ ticks
    
    ax_image.imshow(image, **kwargs)
    
    return fig

def frequency_dict(df, is_shot=False):
    """
    Function to create frequency-dict
    for number of goals.

    Args:
        df (pandas.DataFrame): required dataframe.

    Returns:
        dict: frequency dictionary.
    """    
    # dataframe with home and away goals for all matches
    goals_per_game = df[["FTHG", "FTAG"]].sum(axis=1)

    # fetch unique values
    all_nums = np.unique(goals_per_game)

    # get frequency
    frequency = dict(goals_per_game.value_counts())

    if not is_shot:
        # fetch max and min number of goals
        max_num, min_num = max(all_nums), min(all_nums)

        # to see all the values are present b/w max_num and min_num
        for goal in range(min_num, max_num + 1):
            if frequency.get(goal) is None:
                frequency[goal] = 0

    return frequency, np.mean(goals_per_game.values)


def create_axis(
    xticks, yticks, xlim, ylim,
    fontproperties=None, size=None
):
    """
    Function to create axis.
    
    Args:
        xticks (numpy.array): xtick values.
        yticks (numpy.array): ytick values.
        xlim (tuple): x-limit.
        ylim (tuple): y-limit.
        fontproperties(FontManager, optional): fontproperty for the font. Defaults to None.
        size (float, optional): size of font. Defaults to None.
    
    Returns:
        figure.Figure: figure object.
        axes.Axes: axes object.
    """
    ## create subplot
    fig, ax = plt.subplots(facecolor="#222222", figsize=(16,12))
    ax.set_facecolor("#222222")

    ## hide the all the spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ## change color
    ax.spines['bottom'].set_color("#F5F5F5")
    ax.spines['left'].set_color("#F5F5F5") 

    ## change color of tick params
    ax.tick_params(axis='x', colors="#F5F5F5")
    ax.tick_params(axis='y', colors="#F5F5F5")

    
    if fontproperties and size:
        for i in ax.get_xticklabels():
            i.set_font_properties(fontproperties)
            i.set_size(size)

        for i in ax.get_yticklabels():
            i.set_font_properties(fontproperties)
            i.set_size(size)

    ## set ticks
    ax.set_xticks(np.round(xticks, 2))
    ax.set_yticks(np.round(yticks, 2))

    ## setting the limit
    ax.set(xlim=xlim, ylim=ylim)
    
    return fig, ax

def make_histogram(
    hist_dict, xticks, yticks, xlim, ylim, hist_color,
    background, text_color, font_normal, font_bold,
    font_italic, x_label, y_label, 
    title, sub_title, credit, image_path=None,
    save_path=None, dpi=500, title_x=0, image_x=0,
    plot_poission=False, mean_value=None, total_matches=380
):
    """
    Function to make required histogram.

    Args:
        hist_dict (dict): containing values to plot histogram.
        xticks (numpy.array): x-tick values.
        yticks (numpy.array): y-tick values.
        xlim (tuple): x-axis limit.
        ylim (tuple): y-axis limit.
        hist_color (str): color for histogram.
        background (str): background color.
        text_color (str): text color.
        font_normal (FontManager): normal font.
        font_bold (FontManager): bold font.
        font_italic (FontManager): italic font.
        x_label (str): x-label.
        y_label (str): y-label.
        title (str): title string.
        sub_title (str): sub-title string.
        credit (str): credit string.
        image_path (str): path to image.
        save_path (str): path where plot should be saved. Default to None.
        dpi (float): dots per inch. Defaults to 500.
        title_x (float, optional): to shift the title in x-direction. Defaults to 0.
        image_x (float, optional): to shift the image in x-direction. Defaults to 0.
        plot_poission (bool, optional): to plot poission distribution. Defaults to False.
        mean_value (float, optional): mean value for poission distribution. Defaults to None.
        total_matches (float, optional): total number of matches in a season. Defaults to 380.

    Returns:
        tuple: containing figure.Figure and axes.Axes objects.
    """
    # define path-effect
    path_effect = [path_effects.withStroke(linewidth=3, foreground=background)]

    # create axis for histogram
    fig, ax = create_axis(
        xticks, yticks, xlim, ylim,
        fontproperties=font_normal.prop, size=13
    )

    # x and y axis values
    x_axis = list(hist_dict.keys())
    y_axis = list(hist_dict.values())

    # plot bars
    ax.bar(
        x_axis, y_axis, color=hist_color, label="Goals Scored",
        edgecolor="#121212", linewidth=1, zorder=2
    )
    
    if plot_poission:
        # calculate value
        poission_value = total_matches * poisson.pmf(xticks, mu=mean_value)
        
        # plot poission
        ax.plot(
            xticks, poission_value, lw=2, color="#F6F6F6",
            marker='o', markersize=8, markeredgecolor="#101010", 
            label="Poission Distribution"
        )

        ax.legend(
            facecolor="none", labelcolor=text_color, fontsize=14
        )

    # set labels
    ax.set_xlabel(
        x_label, color=text_color, 
        fontproperties=font_normal.prop, size=22,
        path_effects=path_effect, labelpad=10

    )
    ax.set_ylabel(
        y_label, color=text_color, 
        fontproperties=font_normal.prop, size=22,
        path_effects=path_effect
    )

    # grid
    ax.grid(b=True, axis="both", alpha=0.1)

    # add credits
    fig.text(
        0.9, 0.065, credit,
        fontsize=12, color=text_color, fontproperties=font_italic.prop,
        ha="right", va="center", path_effects=path_effect
    )

    # add image
    if image_path is not None:
        fig = add_image(
            image_path, fig, 0.06 + image_x, 0.91, 0.1, 0.1
        )

    # add title
    fig.text(
        0.145 + title_x, 0.97, title, path_effects=path_effect,
        fontsize=26, color=text_color, fontproperties=font_bold.prop,
    )
    fig.text(
        0.145 + title_x, 0.94, sub_title, path_effects=path_effect,
        fontsize=22, color=text_color, fontproperties=font_bold.prop,
    )

    if save_path is not None:
        # save the plot
        fig.savefig(
            save_path, dpi=dpi, bbox_inches="tight"
        )

    return fig, ax