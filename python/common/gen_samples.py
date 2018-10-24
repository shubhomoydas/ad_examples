from scipy.stats import multivariate_normal as mvn

from common.data_plotter import *
from common.utils import *

"""
pythonw -m common.gen_samples
"""


class MVNParams(object):
    def __init__(self, mu, mcorr, dvar):
        self.mu = mu
        self.mcorr = mcorr
        self.dvar = dvar


def interpolate_2D_line_by_point_and_vec(x, mu, m):
    """
    (y - mu_y) / (x - mu_x) = m_y / m_x = m_
    => y = m_*(x - mu_x) + mu_y

    :param x: np.array
        range of x-axis
    :param mu: np.array
        point through which the line should pass
    :param m: np.array
        vector that represents the line's direction
    :return:
    """
    m_ = m[1] / m[0]
    y = m_ * (x - mu[0]) + mu[1]
    # logger.debug("y:\n%s" % str(list(y)))

    return cbind(x, y)


def interpolate_2D_line_by_slope_and_intercept(x, m, c):
    """
    y = m * x + c

    :param x: np.array
    :param m: float
    :param c: float
    :return:
    """
    y = m * x + c
    # logger.debug("y:\n%s" % str(list(y)))

    return cbind(x, y)


def generate_dependent_normal_samples(n, mu, mcorr, dvar):
    d = mcorr.shape[1]
    mvar = np.copy(mcorr)
    np.fill_diagonal(mvar, 1.)
    # logger.debug("mvar:\n%s" % str(mvar))
    if d > 1:
        for i in range(0, d-1):
            for j in range(i+1, d):
                mvar[j, i] = mvar[i, j]
        p = np.diag(np.sqrt(dvar))
        mvar = p.dot(mvar).dot(p)
        rv = mvn(mu, mvar)
        s = rv.rvs(size=n)
    else:
        s = np.random.normal(loc=mu[0], scale=np.sqrt(dvar[0]), size=n)
        # logger.debug(str(list(s)))
        s = np.reshape(s, (-1, 1))

    if n == 1:
        # handle the case where numpy automatically makes column vector if #rows is 1
        s = np.reshape(s, (n, len(s)))
    return s


def get_sphere_samples(sampledef):
    x = np.zeros(shape=(0, 2))
    y = np.zeros(0, dtype=int)
    for n, label, start_angle, end_angle in sampledef:
        thetas = np.random.uniform(start_angle, end_angle, n)
        samples = np.hstack([np.transpose([np.cos(thetas)]), np.transpose([np.sin(thetas)])])
        x = np.vstack([x, samples])
        y = np.append(y, np.ones(n, dtype=int)*label)
    return x, y


def get_sample_defs(stype=1):
    if stype == 1:
        ns = [200, 250, 20, 15,  0,   0,  0,   0,   0]
    elif stype == 2:
        ns = [200, 250, 20,  0, 15,   0,  0,   0,   0]
    elif stype == 3:
        ns = [  0,   0,  0,  0,  0, 150,  1,   0,   0]
    elif stype == 4:
        ns = [  0,   0,  1,  1,  0,   0,  0, 100,  30]
    elif stype == 5:
        ns = [  0, 200, 10,  0,  0,   0,  0,   0,   0]
    else:
        raise ValueError("Incorrect sample type %d" % stype)

    label_order = ["nominal", "nominal", "anomaly", "anomaly", "anomaly",
                   "nominal", "anomaly", "nominal", "nominal"]
    sampledefs = list([
        MVNParams(
        mu=np.array([2., 3.]),
        mcorr=np.array([
            [1, 0.5],
            [0, 1.0]]),
            dvar=np.array([3., 2.])
        ),
        MVNParams(
        mu=np.array([1., 1.]),
        mcorr=np.array([
          [1, 0.5],
          [0, 1.0]]),
        dvar=np.array([2., 3.])
        ),
        MVNParams(
            mu=np.array([6., 3.]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.5, 1])
        ),
        MVNParams(
            mu=np.array([4.5,-1]),  # for first toy data
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.5, 1])
        ),
        MVNParams(
            mu=np.array([0.5,5]),  # for second toy data
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.5, 1])
        ),
        MVNParams(
            mu=np.array([2., 3.]),  # for third toy data
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([3., 2.])
        ),
        MVNParams(
            mu=np.array([8.0, -20.5]),  # for third toy data
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.5, 0.5])
        ),
        MVNParams(
            mu=np.array([2., 3.]),
            mcorr=np.array([
                [1, 0.0],
                [0, 1.0]]),
            dvar=np.array([1.5, 1.5])
        ),
        MVNParams(
            mu=np.array([-2., -3.]),
            mcorr=np.array([
                [1, 0.0],
                [0, 1.0]]),
            dvar=np.array([0.15, 0.15])
        )
    ])
    return sampledefs, label_order, ns


def get_synthetic_samples(sampledefs=None, label_order=None, ns=None, stype=1):

    if sampledefs is None:
        sampledefs, label_order, ns = get_sample_defs(stype=stype)

    s = np.zeros(shape=(0,2))
    for i in range(0, len(sampledefs)):
      n = ns[i]
      if n > 0:
          sampledef = sampledefs[i]
          si = generate_dependent_normal_samples(n, sampledef.mu, sampledef.mcorr, sampledef.dvar)
          s = rbind(s, si)

    label_cls = []
    anomaly_dataset = False
    for i in range(len(ns)):
        if ns[i] > 0:
            if type(label_order[i]) == str and (label_order[i] == "anomaly" or
                                                label_order[i] == "nominal"):
                anomaly_dataset = True
            label_cls.extend([label_order[i]] * ns[i])
    if anomaly_dataset:
        labels = [1 if ll == "anomaly" else 0 for ll in label_cls]
    else:
        labels = label_cls

    # logger.debug("labels:\n%s" % labels)
    # logger.debug("#s: %d, #labels: %d" % (s.shape[0], len(labels)))

    return s, np.array(labels)


def get_hard_samples(stype=1):
    if stype == 1:
        ns = [200, 250, 6, 8, 5, 5, 5, 3, 2]
    else:
        ns = [200, 250, 6, 8, 5, 5, 5, 3, 2]

    label_order = ["nominal", "nominal", "anomaly", "anomaly", "anomaly",
                   "anomaly", "anomaly", "anomaly", "anomaly"]

    sampledefs = list([
        MVNParams(
            mu=np.array([1.0, 3.5]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([3., 2.])
        ),
        MVNParams(
            mu=np.array([3.25, 1.75]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([2., 3.])
        ),
        MVNParams(
            mu=np.array([0, 4.25]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.05, 0.05])
        ),
        MVNParams(
            mu=np.array([2.5, 4.5]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.1, 0.1])
        ),
        MVNParams(
            mu=np.array([1., 2.5]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.04, 0.04])
        ),
        MVNParams(
            mu=np.array([2.5, 2.25]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.03, 0.03])
        ),
        MVNParams(
            mu=np.array([2.25, 0.25]),
            mcorr=np.array([
                [1, 0.5],
                [0, 1.0]]),
            dvar=np.array([0.05, 0.05])
        ),
        MVNParams(
            mu=np.array([5.0, 3.25]),
            mcorr=np.array([
                [1, 0.0],
                [0, 1.0]]),
            dvar=np.array([0.05, 0.05])
        ),
        MVNParams(
            mu=np.array([4.0, 0.]),
            mcorr=np.array([
                [1, 0.0],
                [0, 1.0]]),
            dvar=np.array([0.05, 0.05])
        )
    ])

    x, y = get_synthetic_samples(sampledefs=sampledefs, label_order=label_order, ns=ns)
    return x, y, ns


def load_donut_data(with_diffusion=False):
    if not with_diffusion:
        df = read_csv("../datasets/donut-shape.csv", header=True)
        data = np.array(df)
        y = np.zeros(data.shape[0])
        y[len(y)-1] = 1  # the last point at the middle of the donut is known anomaly
    else:
        df = read_csv("../datasets/donut_with_labeldiffusion.csv", header=True)
        data = np.array(df)
        y = data[:, 0]
        data = data[:, 1:data.shape[1]]
    logger.debug(data.shape)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_dif = data_max - data_min
    data = data / data_dif
    return data, y


class AnomalyDataOpts(object):
    def __init__(self, dataset, datafile=None):
        """ Reads one of the standard anomaly detection datasets included with the codebase

        :param dataset: string
            name of one of the standard anomaly datasets included with the codebase
        """
        self.datafile = datafile
        if self.datafile is None or self.datafile == "":
            self.datafile = "../datasets/anomaly/%s/fullsamples/%s_1.csv" % (dataset, dataset)
        self.labelindex = 1 # Note: this is 1-indexed (*not* 0-indexed)
        self.startcol = 2 # Note: this is 1-indexed (*not* 0-indexed)
        self.header = 0


def read_anomaly_dataset(dataset, datafile=None):
    """ Returns a standard dataset included with the codebase

    Supported datasets:
        abalone
        ann_thyroid_1v3
        cardiotocography_1
        covtype
        covtype_sub
        kddcup
        kddcup_sub
        mammography
        mammography_sub
        shuttle_1v23567
        shuttle_sub
        yeast
        weather
        toy
        toy2

    :param dataset: string
    :return: numpy.ndarray, numpy.array
    """
    opts = AnomalyDataOpts(dataset, datafile=datafile)
    x, y = read_data_as_matrix(opts)
    # logger.debug("x: %s" % str(x.shape))
    # logger.debug("x:\n%s" % str(x[0:2, :]))
    return x, y


def normalize_and_center_by_feature_range(x):
    x_mean = np.mean(x, axis=0)
    x_ = x - x_mean
    x_min = x_.min(axis=0)
    x_max = x_.max(axis=0)
    x_dif = x_max - x_min
    x_ = x_ / x_dif
    return x_


def load_face_data(with_diffusion=False):
    if not with_diffusion:
        df = read_csv("../datasets/face_with_anoms.csv", header=True)
    else:
        df = read_csv("../datasets/face_with_labeldiffusion.csv", header=True)
    data = np.array(df)
    y = data[:, 0]
    data = data[:, 1:data.shape[1]]
    logger.debug(data.shape)
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    data_dif = data_max - data_min
    data = data / data_dif
    return data, y


def get_demo_samples(sample_type):
    if sample_type == "donut_":
        x, y = load_donut_data()
    elif sample_type == "donut_diff_":
        x, y = load_donut_data(with_diffusion=True)
    elif sample_type == "face_":
        x, y = load_face_data()
    elif sample_type == "face_diff_":
        x, y = load_face_data(with_diffusion=True)
    elif sample_type == "1_":
        x, y = get_synthetic_samples(stype=1)
    elif sample_type == "4_":
        x, y = get_synthetic_samples(stype=4)
    else:
        raise ValueError("Invalid sample type %s" % sample_type)
    return x, y


def plot_samples_and_lines(x, lines=None, line_colors=None, line_legends=None,
                           top_anoms=None, pdfpath=None,
                           line_widths=None, samplescol="grey",
                           labels=None, lbl_color_map=None,
                           marker='o', s=15, xlim=None, ylim=None):
    if pdfpath is None:
        raise ValueError("Need valid pdf path...")
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('x')
    plt.ylabel('y')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    dp.plot_points(x, pl, labels=labels, lbl_color_map=lbl_color_map,
                   marker=marker, s=s, facecolors='none', defaultcol=samplescol)
    if top_anoms is not None:
        pl.scatter(x[top_anoms, 0], x[top_anoms, 1], marker='o', s=35,
                   edgecolors='red', facecolors='none')
    if lines is not None:
        for i, line in enumerate(lines):
            if line is None:
                continue
            legend = None if line_legends is None else line_legends[i]
            color = 'blue'
            linewidth = 2
            if line_colors is not None:
                color = line_colors[i]
            if line_widths is not None:
                linewidth = line_widths[i]
            pl.plot(line[:, 0], line[:, 1], '--', color=color, linewidth=linewidth, label=legend)
        if line_legends is not None:
            pl.legend(loc='best')
    dp.close()


def plot_sample(x, y, pdfpath):
    dp = DataPlotter(pdfpath=pdfpath, rows=1, cols=1)
    pl = dp.get_next_plot()
    plt.xlabel('x')
    plt.ylabel('y')
    dp.plot_points(x, pl, labels=y, lbl_color_map={0: "blue", 1: "red"},
                   marker='o', s=35, facecolors='none')
    dp.close()
