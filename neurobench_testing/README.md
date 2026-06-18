# Set Up

all packages needed (maybe some extra oopsie) are listed in the requirement.txt file
except for memtorch which needs to be cloned and compiled locally on linux systems,
I am not sure for windows or mac.

```
pip install -r requirements.txt

git clone --recursive https://github.com/coreylammie/MemTorch
python setup.py install
```

the memtorch testing file is building off of the example neurobench using memtorch
to simulate the model on a memristive crossbar.
