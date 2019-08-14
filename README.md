jooble-test is a simple Python module to normalizing vacancy data using A-score normalization

In script also implemented two strategy for preprocessing train data:

- In memory strategy for files with size < 100 * 2^20
- For bigger files Welford`s online algorithm will be used (https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)

jooble-test requires:

python >= 3.7.0

Click==7.0
numpy==1.17.0
pandas==0.25.0
python-dateutil==2.8.0
pytz==2019.2
six==1.12.0



## Getting started

First, create new virtual environment and activate, of course:

```console
$  git clone https://github.com/andreytsykh/jooble-test.git
$  cd jooble-test
$  virtualenv venv
$  source venv/bin/activate
```
Then install requirements:

```console
(venv) $ pip install -r requirements.txt
```

Program has folloqing console interface:
```console
(venv) $  python src/vacancy_data_processor.py --help
*** Usage: vacancy_data_processor.py [OPTIONS]
Options:
  --src PATH         Path to file with input data
  --dst PATH         Path to file where store the output
  --train PATH       Path to file with train data
  --workers INTEGER  Number of workers
  --help             Show this message and exit.
***
```
By default :
- src = jooble-test/data/test.tsv
- dst = jooble-test/out/test_proc_{current_date}.tsv  (First time directory will be created)
- train = jooble-test/data/train/tsv
- workers = 3


That's about it.

