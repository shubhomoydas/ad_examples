import os
import pandas as pd
import collections
import zipfile
from six.moves import urllib

from ..common.utils import logger


def maybe_download_casas(zipfilename, url, filepathinzip, localpath):
    """Download the CASAS file if not present, and extract the relevant txt file."""
    if os.path.exists(os.path.join(localpath, filepathinzip)):
        return os.path.join(localpath, filepathinzip)

    local_filename = os.path.join(localpath, zipfilename)
    if not os.path.exists(local_filename):
        local_filename, _ = urllib.request.urlretrieve(url + zipfilename,
                                                       local_filename)
    statinfo = os.stat(local_filename)
    if os.path.exists(local_filename) and statinfo.st_size > 0:
        with zipfile.ZipFile(local_filename) as f:
            casasfile = f.extract(filepathinzip, localpath)
        return casasfile
    return None


def write_sensor_data_as_document(cas):
    localpath = "/Users/moy/work/git_public/word2vec-data/sensor"
    with open(localpath + '/sensor_data.txt', 'w') as f:
        f.write(" ".join(cas.sensor_seq))


class Casas(object):
    def __init__(self, sensor_seq=None, sensor_enc=None,
                 sensors=None, sensor2code=None, code2sensor=None,
                 dataset_path=None):
        self.sensor_seq = sensor_seq
        self.sensor_enc = sensor_enc
        self.sensors = sensors
        self.sensor2code = sensor2code
        self.code2sensor = code2sensor

        if dataset_path is not None:
            self.load_data(dataset_path)

    def load_data(self, dataset_path):
        self.sensor_seq, self.sensor_enc, self.sensors, self.sensor2code, self.code2sensor = \
            Casas.read_sensor_data(dataset_path)

    @staticmethod
    def build_dataset(words, n_words=10000):
        """Process raw inputs into a dataset."""
        # count = [['UNK', -1]]
        count = []
        count.extend(collections.Counter(words).most_common(n_words - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        data = list()
        unk_count = 0
        for word in words:
            index = dictionary.get(word, 0)
            # if index == 0:  # dictionary['UNK']
            #     unk_count += 1
            data.append(index)
        # count[0][1] = unk_count
        reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return data, count, dictionary, reversed_dictionary

    @staticmethod
    def read_sensor_data(dataset_path):
        logger.debug("dataset_path: %s" % dataset_path)
        data = pd.read_csv(dataset_path,
                           header=None, sep='\t',
                           names=['x0', 'x1', 'x2', 'x3', 'x4'], dtype={'x2': str}
                           # parse_dates=dataset_def.date_columns, index_col=dataset_def.index_column,
                           # squeeze=True, date_parser=date_parser
                           )
        logger.debug("Raw data shape: %s" % str(data.shape))

        # retain only the sensor ON/OFF readings
        sensor_data = data[data['x1'].str.startswith(("M", "MA"))]
        # logger.debug(sensor_data)
        logger.debug("filtered sensor_data.shape: %s" % str(sensor_data.shape))

        # list of all sensors
        sensors = sensor_data.x1.unique()
        logger.debug("All sensors:\n%s" % str(sensors))

        # sequential training data
        sensor_seq = list(sensor_data.x1.values)
        logger.debug("len(sensor_seq): %d" % len(sensor_seq))

        if False:
            sensor2code = {}
            code2sensor = {}
            for i, sensor in enumerate(sorted(sensors)):
                sensor2code[sensor] = i
                code2sensor[i] = sensor

            sensor_enc = [sensor2code[sensor] for sensor in sensor_seq]
        else:
            sensor_enc, count, sensor2code, code2sensor = Casas.build_dataset(sensor_seq)
            logger.debug(count)

        return sensor_seq, sensor_enc, sensors, sensor2code, code2sensor


