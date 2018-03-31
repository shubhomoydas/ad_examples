import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
from scipy.stats import poisson
from pandas import Series
from common.utils import *
from common.data_plotter import *

"""
pythonw -m timeseries.simulate_timeseries
"""


# transition matrix
T = np.array([[0.8, 0.2],   # A1->{A1, A2}
              [0.6, 0.4]])  # A2->{A1, A2}


class SimTs(object):
    def __init__(self):
        pass

    def get_samples(self, curr, dur):
        """
        :param curr: float
            Current time series value. Used for continuity
        :param dur: int
            Total duration for which to run this time series (length of output)
        :return: np.array
        """
        pass


class Sinusoidal(SimTs):
    def __init__(self, freq=5):
        """
        :param freq: int
            Average angle (in degrees) per time step
        """
        super(SimTs, self).__init__()
        self.freq = freq

    def get_samples(self, curr, dur):
        c = max(min(curr, 1.0), -1.0)
        start_angle = np.arcsin(c)
        angle_incr = np.cumsum(poisson.rvs(self.freq, size=dur) + 1.)  # each duration is 5 degrees on avg
        angle_noise = rnd.normal(0, 0.1, dur)  # some noise to the angle
        angle = (np.pi * self.freq / 180.) * angle_incr
        samples = np.sin(start_angle + angle + angle_noise)
        return samples


class MA1(SimTs):
    def __init__(self, beta=1.):
        """
        :param freq: int
            Average angle (in degrees) per time step
        """
        super(SimTs, self).__init__()
        self.beta = beta

    def get_samples(self, curr, dur):
        e = rnd.normal(0, 1, dur + 1)
        e[0] = curr
        samples = self.beta * e[:-1] + e[1:]
        return samples


def write_to_file(arr, file=None, add_row_index=False, fmt=None):
    n = len(arr)
    fmts = ["%3.6f"] if fmt is None else fmt
    if add_row_index:
        fmts.insert(0, "%d")
        data = np.hstack([np.arange(n).reshape((n, 1)), np.transpose([arr])])
    else:
        data = np.transpose([arr])
    np.savetxt(file, data, fmt=fmts, delimiter=",")


def generate_synthetic_activity_data():

    actions = ["Sin_5", "MA1", "Sin_1"]
    # transition matrix
    T = np.array([[0.4, 0.3, 0.3],
                  [0.7, 0.2, 0.1],
                  [0.4, 0.1, 0.5]])

    acts = [Sinusoidal(freq=5), MA1(), Sinusoidal(freq=2)]

    act_samples = []
    for i in range(len(acts)):
        act_samples.append(acts[i].get_samples(0.0, 100))

    iters = 2000
    avg_durs = [30, 30, 30]
    activity = 0  # start with A1
    starts = list()
    activities = list()
    samples = np.zeros(0)
    start = 0
    curr = 0.
    for i in range(iters):
        starts.append(start)
        activities.append(activity)
        dur = poisson.rvs(avg_durs[activity])
        curr_samples = acts[activity].get_samples(curr, dur)
        samples = np.append(samples, curr_samples)
        curr = curr_samples[-1]
        start += dur
        if False:
            activity += 1  # maybe lookup transition matrix later
            activity = 0 if activity >= len(actions) else activity
        else:
            activity = np.argmax(rnd.multinomial(1, T[activity, :]))

    n = len(samples)
    logger.debug("n: %d" % n)
    logger.debug("samples:\n%s" % str(list(samples)))

    # save the generated data to file(s)
    output_path = "./temp/timeseries"
    write_to_file(samples, "%s/samples_%d.csv" % (output_path, iters), add_row_index=True, fmt=["%3.6f"])
    write_to_file(np.array(activities), "%s/activities_%d.csv" % (output_path, iters), add_row_index=True, fmt=["%d"])
    write_to_file(np.array(starts), "%s/starts_%d.csv" % (output_path, iters), add_row_index=True, fmt=["%d"])

    pdfpath = "temp/timeseries/timeseries_simulation.pdf"
    dp = DataPlotter(pdfpath=pdfpath, rows=3, cols=1)

    for i in range(len(acts)):
        ns = len(act_samples[i])
        pl = dp.get_next_plot()
        plt.title("Simulated Time Series (%s)" % actions[i], fontsize=8)
        pl.set_xlim([0, ns])
        pl.plot(np.arange(0, ns), act_samples[i], 'b-')

    pl = dp.get_next_plot()
    plt.title("Simulated Time Series", fontsize=8)
    pl.set_xlim([0, n])
    pl.plot(np.arange(0, n), samples, 'b-')
    if False:
        for x in starts:
            pl.axvline(x, color='red', linestyle='solid')

    dp.close()


class ActivityData(object):
    def __init__(self, samples, y=None, activities=None, starts=None):
        self.samples = samples
        self.activities = activities
        self.starts = starts
        self.y = y

        if self.y is None and self.activities is not None and self.starts is not None:
            # populate the activity labels
            n = self.samples.shape[0]
            n_acts = self.activities.shape[0]
            self.y = np.zeros(shape=(n, 1), dtype=int)
            for i in range(n_acts):
                s = self.starts[i, 0]
                e = n if i == n_acts-1 else self.starts[i+1, 0]
                self.y[s:e, 0] = self.activities[i, 0]

    def get_batches(self, n_lags, batch_size):
        n = self.samples.shape[0]
        d = self.samples.shape[1]
        for i in xrange(0, n, batch_size):
            x = np.zeros(shape=(batch_size, n_lags, d), dtype=np.float32)
            e = min(n, i + batch_size)
            sz = e - i
            # logger.debug("i, e, sz: %d, %d, %d" % (i, e, sz))
            for t in range(n_lags):
                st = max(0, i - t)  # maximum time we can go back in the past
                et = e - t
                # logger.debug("st, et: %d, %d" % (st, et))
                if et >= st:
                    x[(sz-(et-st)):sz, t, :] = self.samples[st:et, :]
                else:
                    break
            yield x[0:sz, :, :], None if self.y is None else self.y[i:e, :]

    def get_shingles(self, window_size, skip_size=None, batch_size=100):
        skip_size = window_size if skip_size is None else skip_size
        n = self.samples.shape[0]
        d = self.samples.shape[1]
        if batch_size < 0:
            batch_size = 1 + n // skip_size
        x = np.zeros(shape=(batch_size, window_size, d), dtype=np.float32)
        w = np.zeros(batch_size, dtype=np.int)  # window id
        y = None
        if self.y is not None: y = np.zeros(batch_size, dtype=np.int)
        l = 0
        for i in xrange(0, n, skip_size):
            st = max(0, i - window_size)
            if i < window_size: st = None  # zero indexing in reverse requires this
            et = min(i + 1, window_size)
            # logger.debug("i, l, st, et: %d %d, %d, %d" % (i, l, 0 if st is None else st, et))
            x[l, 0:et, :] = self.samples[i:st:-1, :]
            w[l] = i
            if self.y is not None:
                y[l] = self.y[i]
            l += 1
            if l == batch_size or i + skip_size >= n:
                # logger.debug("l: %d" % l)
                yield x[0:l, :, :], None if self.y[0:l] is None else self.y[0:l], w[0:l]
                if i + skip_size < n:
                    l = 0
                    x = np.zeros(shape=(batch_size, window_size, d), dtype=np.float32)
                    w = np.zeros(batch_size, dtype=np.int)
                    if self.y is not None: y = np.zeros(batch_size, dtype=np.int)


def read_activity_data():
    activities = pd.read_csv("../datasets/simulated_timeseries/activities_2000.csv",
                             header=None, sep=",", usecols=[1]
                             )
    activities = np.asarray(activities, dtype=np.int)
    samples = pd.read_csv("../datasets/simulated_timeseries/samples_2000.csv",
                          header=None, sep=",", usecols=[1]
                          )
    samples = np.asarray(samples, dtype=np.float32)
    starts = pd.read_csv("../datasets/simulated_timeseries/starts_2000.csv",
                         header=None, sep=",", usecols=[1]
                         )
    starts = np.asarray(starts, dtype=np.int)
    return ActivityData(samples, y=None, activities=activities, starts=starts)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)

    args = get_command_args(debug=True, debug_args=["--debug",
                                                    "--plot",
                                                    "--log_file=temp/timeseries/timeseries_sim.log"])
    # print "log file: %s" % args.log_file
    configure_logger(args)

    random.seed(42)
    rnd.seed(42)

    generate_synthetic_activity_data()
