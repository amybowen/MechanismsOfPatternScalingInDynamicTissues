""" A module containing functions useful for plotting.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
import colorsys
import numpy as np


# set default plotting parameters
plt.rcParams['figure.figsize'] = [5,3.7]
figsize_cbar = [4.5,3.7]
# plt.rcParams['figure.figsize'] = [4,3]

plt.rcParams['lines.linewidth'] = 3.0
# fontsizes=[20,22,24]
fontsize = 50
smallfontsize = 36
cbarlabelfontsize = 44

plt.rc('font', size=fontsize)          # controls default text sizes
plt.rc('axes', titlesize=fontsize)     # fontsize of the axes title
plt.rc('axes', labelsize=fontsize)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=smallfontsize)    # fontsize of the tick labels
plt.rc('ytick', labelsize=smallfontsize)    # fontsize of the tick labels
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['lines.linewidth'] = 5
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3


N1_colour='#DC257E'
N2_colour='#6C63AC'
M_colour='#FCAF16'
orange_colour = '#EA5711'
brat_colour = '#89cc04'
green_colour = '#7EDC25'
blue_colour = '#257EDC'
brown_colour = '#AC6C63'
dark_green_colour = '#63AC6C'
tea_green = '#BDD9BF'
dark_purple = '#412234'
bittersweet_colour = '#F05D5E'
carribean_crush = '#16697A'
camblue = '#9ABCA7'
alpha1color = '#D17B88'
alpha2color = '#F6AE2D'
alpha3color = '#5887FF'
olivine_colour = '#ABC798'
rose_colour = '#F560C3'
indigo_colour = '#473198'
monostable_colour = '#D7FFAB'
bistable_colour = '#D7C0D0'
accessible_bistable_colour = '#AF9B46'
violet_colour = '#462255'
columbia_blue_colour = '#BDD5EA'
moss_green_colour = '#758E4F'
hunyadi_yellow_colour = '#F6AE2D'
pantone_orange_colour = '#F26419'
lapis_lazuli_colour = '#33658A'
carolina_blue_colour = '#86BBD8'
bright_pink_crayola_colour = '#E85D75'
dynamic_bistable_colour = '#CAAAAFFF'
N2_sg_colour = '#B02B3D'
bistable_colour_sg = "#CAAAAFFF"
n2_colour_sg = "#B02B3D"
tangerine_color = '#EFA48B'

def setDefaultParams():
    """ Set default plotting parameters for matplotlib """
    plt.rcParams['figure.figsize'] = [5,3.7]
    plt.rcParams['lines.linewidth'] = 3.0
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.linewidth'] = 1.2
    return 



def make_darker_shades(hex_code, n=5, min_factor=0.5, max_factor=1.0):
    """ Function to generate n shades of a given hex code colour with increasing darkness """
    rgb = mcolors.to_rgb(hex_code)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Create shades by decreasing lightness
    shades = []
    for i in range(n):
        # scale lightness from max_factor to min_factor
        new_l = l * (max_factor - min_factor * i / (n - 1))
        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
        shades.append(new_rgb)
    return shades



def make_lighter_shades(hex_code, n=5, min_factor=0.5, max_factor=1.0):
    """ Function to generate n shades of a given hex code colour with increasing lightness """
    rgb = mcolors.to_rgb(hex_code)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    
    # Create shades by increasing lightness
    shades = []
    for i in range(n):
        # scale lightness from max_factor to min_factor
        new_l = l * (max_factor - min_factor * i / (n - 1))
        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
        shades.append(new_rgb)
    return shades

def make_saturation_gradient(hex_colour, n, min_s=0.3, max_s=1.0):
    rgb = mcolors.to_rgb(hex_colour)
    h, l, s = colorsys.rgb_to_hls(*rgb)

    sats = np.linspace(max_s, min_s, n)
    return [colorsys.hls_to_rgb(h, l, s_) for s_ in sats]


def make_ticks_minimal(ax=None, x_n_ticks=4, y_n_ticks=4):
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(x_n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(y_n_ticks))


def sci_notation(num, precision=0):
    s = f"{num:.{precision}e}"
    base, exp = s.split("e")
    exp = int(exp)
    return fr"{base}\times 10^{{{exp}}}"