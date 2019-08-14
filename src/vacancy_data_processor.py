import abc
import collections
import csv
import io
import logging
import os
import math
import multiprocessing as mp
import queue
import time
import pandas as pd
import click

logging.basicConfig(level=logging.DEBUG)

STOP_WRITER_SIGNAL = 'STOP_WRITER'
STOP_WORKER_SIGNAL = 'STOP_WORKER'

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, 'data'))
OUT_DIR = os.path.normpath(os.path.join(BASE_DIR, 'out'))

INPUT_TRAIN_FILE = os.path.join(DATA_DIR, 'train.tsv')
INPUT_FILE = os.path.join(DATA_DIR, 'test.tsv')
OUTPUT_FILE = os.path.join(OUT_DIR, f'test_proc_{time.strftime("%m_%d_%Y_%H_%M_%S")}.tsv')



def iterable_to_stream(iterable, buffer_size=io.DEFAULT_BUFFER_SIZE):

    class IterStream(io.RawIOBase):
        def __init__(self):
            self.leftover = None

        def readable(self):
            return True

        def readinto(self, b):
            try:
                l = len(b)
                chunk = self.leftover or next(iterable)
                output, self.leftover = chunk[:l], chunk[l:]
                b[:len(output)] = output.encode()
                return len(output)
            except StopIteration:
                return 0
    return io.BufferedReader(IterStream(), buffer_size=buffer_size)

class StatisticsComputingStartegy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_statistics():
        pass

class InMemoryStrategy(StatisticsComputingStartegy):
    def __init__(self, train_file):
        self.mean_dict = {}
        self.std_dict = {}
        self.train_file = train_file
    
    def get_statistics(self):
        tsv_stream = self._TsvToCsvStreamer(self.train_file)

        df = pd.read_csv(iterable_to_stream(tsv_stream.read()), header=None)
        grouped_df = df.groupby(1)
        group_code_set = set()
        for g in grouped_df.groups:
            group_code_set.add(g)
            current_df = grouped_df.get_group(g)

            for n, el in enumerate(current_df.iloc[:, 2:].mean(), 1):
                self.mean_dict[f'f_{g}_{n}'] = el

            for n, el in enumerate(current_df.iloc[:, 2:].std(), 1):
                self.std_dict[f'f_{g}_{n}'] = el
        
        return self.mean_dict, self.std_dict, group_code_set

    class _TsvToCsvStreamer:
        def __init__(self, file):
            self.file = file

        def read(self):
            with open(self.file) as tsv:
                for line in csv.DictReader(tsv, dialect='excel-tab'):
                    csv_line = ','.join([line['id_job'], line['features']])
                    yield csv_line + '\n'
                        
class StreamingStrategy(StatisticsComputingStartegy):
    def __init__(self, train_file):
        self.file = train_file
        self.std_dict = collections.defaultdict(float)
        self.mean_dict = collections.defaultdict(float)
        self.code_to_q = collections.defaultdict(int)
    
    def read_file(self, file):
        with open(file) as tsv:
            for line in csv.DictReader(tsv, dialect='excel-tab'):
                yield [int(el) for el in line['features'].split(',')]

    def welford_alg(self, row):
        if row is None:
            return

        self.code_to_q[row[0]] += 1
        for n,  el in enumerate(row[1:],1):
            key = f'f_{row[0]}_{n}'
            new_mean = self.mean_dict[key] + (el - self.mean_dict[key]) * 1. /self.code_to_q[row[0]]
            new_std = self.std_dict[key] + (el - self.mean_dict[key]) * (el - new_mean)
            self.mean_dict[key] , self.std_dict[key] = new_mean , new_std

    def get_statistics(self):
        for row in self.read_file(self.file):
            self.welford_alg(row)

        for k, v in self.std_dict.items():
            code_quantity = self.code_to_q[int(k[2])]
            if code_quantity == 1:
                self.std_dict[k] = float('nan')
            else:
                self.std_dict[k] = math.sqrt(v / (code_quantity-1))
        return self.mean_dict , self.std_dict, self.code_to_q.keys()

class TrainDataPreprocessor:
    def __init__(self, train_file=INPUT_TRAIN_FILE):
        self.train_file = train_file
        self.in_memory_limit = 100 * 2 ** 20 

    def choose_computing_strategy(self):
        file_size = os.stat(self.train_file).st_size
        logging.debug(f'TRAIN FILE SIZE: {file_size}')

        if file_size  < self.in_memory_limit:
            logging.debug('CHOOSE IN-MEMORY STRATEGY')
            return InMemoryStrategy(self.train_file)
        else:
            logging.debug('CHOOSE STREAMING STRATEGY')
            return StreamingStrategy(self.train_file)

    def get_statistics(self):
        strategy = self.choose_computing_strategy()
        return strategy.get_statistics()


class ZscoreNormalizator:
    def __init__(self, train_file):
        self.preprocess_data(train_file)

    def preprocess_data(self, train_file, preprocessor=TrainDataPreprocessor):
        logging.debug('Start train data preprocessing')
        self.mean_dict, self.std_dict, self.code_set = preprocessor(train_file).get_statistics()

    def normalize_value(self, value, column):
        try:
            zscore = (value-self.mean_dict[column]) / self.std_dict[column]
        except ArithmeticError:
            zscore = float('nan')
        return zscore

    def get_column_names(self):
        return self.mean_dict.keys(),  self.code_set
    
    def get_mean(self,key):
        return self.mean_dict[key]
        


class VacancyDataManager:
    def __init__(self,rq, wq, src, dst, train, workers=3):
        self.train_file = train
        self.input_file = src
        self.output_file = dst
        self.worker_num = workers
        self.rq = rq
        self.wq  = wq
        self.workers = []

    def set_normalizator(self, normalizator):
        self._normalizator = normalizator(self.train_file)

    def generate_fieldnames(self):
        fieldnames = []
        fieldnames.append('id_job')
        name_keys =  self._normalizator.get_column_names()

        for key in name_keys[0]:
            fieldnames.append(key + '_stand')
        for key in name_keys[1]:
            fieldnames.append(f'max_feature_{key}_index')
            fieldnames.append(f'max_feature_{key}_abs_mean_diff')


        return fieldnames

    def start_processing(self, producer, worker, writer):
        producerp = mp.Process(target=producer,args=(self.rq, self.input_file),  name=f'producer')
        producerp.start()

        writer_process = mp.Process(target=writer, args=(self.wq, self.output_file, self.generate_fieldnames()), name=f'writer')
        writer_process.start()

        for i in range(self.worker_num):
            worker_process = mp.Process(target=worker,args = (self.rq, self.wq, self._normalizator), name=f'worker {i}')
            worker_process.start()
            self.workers.append(worker_process)

        producerp.join()
        writer_process.join()

        for p in self.workers:
            logging.debug(f'Send stop signal to {p} ')
            self.rq.put(STOP_WORKER_SIGNAL)
        for p in self.workers:
            p.join()
        

def producer(read_queue, input_file):
        logging.debug(f'Producer PID: {os.getpid()} started')
        def chunk_generator(chunksize):
            chunk = []
            with open(input_file) as file:
                for i, line in enumerate(csv.DictReader(file, dialect='excel-tab')):
                    if (i % chunksize == 0 and i > 0):
                        yield chunk
                        del chunk[:]
                    chunk.append(line)
                yield chunk 

        for chunk in chunk_generator(chunksize=5000):
            read_queue.put(chunk)
        read_queue.put(STOP_WRITER_SIGNAL)

        

def worker(read_queue, writer_queue, normalizator):
    while True:
        try:
            chunk = read_queue.get_nowait()
        except queue.Empty:
            pass
        except EOFError:
            break
        else:
            if chunk == STOP_WRITER_SIGNAL:
                writer_queue.put(chunk)
                continue
            elif chunk == STOP_WORKER_SIGNAL:
                break

            chunk_val = []
            for row in chunk:
                feature_row = row['features'].split(',')
                feature_set_code = feature_row[0]
                factors = list(map(float, feature_row[1:]))
                max_factor = max(factors)
                values = {}
                values['id_job'] = row['id_job']
                for i, val in enumerate(factors , 1):
                    train_column_name  = f'f_{feature_set_code}_{i}'
                    new_column_name = f'f_{feature_set_code}_{i}_stand'
                    values[new_column_name] = normalizator.normalize_value(val, train_column_name)
                    if val == max_factor:
                        values[f'max_feature_{feature_set_code}_index'] = i
                        values[f'max_feature_{feature_set_code}_abs_mean_diff'] = abs(val - normalizator.get_mean(train_column_name))

                chunk_val.append(values)
            writer_queue.put(chunk_val)

def file_writer(writer_queue , output_file, fieldnames):
    logging.debug(f'Writer PID: {os.getpid()} started')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'a+', newline='') as out_tsv:
        writer = csv.DictWriter(out_tsv,dialect='excel-tab', fieldnames=fieldnames)
        writer.writeheader()
        while True:
            rows = writer_queue.get()
            if rows == STOP_WRITER_SIGNAL:
                logging.debug('Writter finished')
                break
            bytes_write = writer.writerows(rows)



@click.command()
@click.option('--src', default=INPUT_FILE, type=click.Path(exists=True), help='Path to file with input data')
@click.option('--dst', default=OUTPUT_FILE, type=click.Path(exists=False), help='Path to file where store the output')
@click.option('--train', default=INPUT_TRAIN_FILE, type=click.Path(exists=True) , help='Path to file with train data')
@click.option('--workers', default=3, type=click.INT , help='Number of workers')
def start_processing(src, dst, train, workers):
    manager = mp.Manager()
    rq = manager.Queue(5)
    wq  = manager.Queue(3)
    logging.debug(train)
    manager = VacancyDataManager(rq, wq, src, dst, train, workers)
    manager.set_normalizator(ZscoreNormalizator)
    manager.start_processing(producer, worker, file_writer)




if __name__ ==  '__main__':
    start_processing()
    
