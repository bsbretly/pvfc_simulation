import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.style


def setupPlotParams():
    # https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html
    # mpl.rcParams['text.usetex'] = True
    # mpl.style.use('seaborn-poster')  # seems to only work on linux
    mpl.style.use('seaborn-v0_8-poster')

    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.color'] = 'k'
    mpl.rcParams['grid.linestyle'] = ':'
    mpl.rcParams['grid.linewidth'] = 0.5

    # mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['font.size'] = 30

    mpl.rcParams['legend.fontsize'] = 30
    mpl.rcParams['figure.titlesize'] = 30

    # plot
    # mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.dashed_pattern'] = [6, 6]
    mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
    mpl.rcParams['lines.dotted_pattern'] = [1, 3]
    mpl.rcParams['lines.scale_dashes'] = False

    # font
    mpl.rcParams['mathtext.fontset'] = 'cm'
    mpl.rcParams['mathtext.rm'] = 'serif'
