## Simple time-series model

> In progress

### Installation

All Python requirements should be specified in `requirements.txt` file.
After you set up and activate a [virtual environment](https://docs.python.org/3/library/venv.html), just install everything with:

```
pip install -r requirements.txt
```

### Usage

Use `time_series.py` to run the whole process. The usage is following: 
```
time_series.py [-h] -i INPUT -q QUANTITY [-o OUTPUT] [-v] [-m {VAR,LGBM}]
```

If the output path is given, plot is exported as PNG. Otherwise, it is only opened in a console. The evaluation metric details are printed at the end. Use `python .\time_series.py -h` for further details.

You can run the whole program with the example data as:
```
python .\time_series.py -i .\data.csv -q "Consumption" -o "consumption-forecast.png"
```

Additionaly, some initial data exploration and visualization is provided in `explore.ipynb`.


### Tests

Simple tests are provided in `unit_test.py`. Run them with:
```
python -m unittest .\unit_test.py -v
```