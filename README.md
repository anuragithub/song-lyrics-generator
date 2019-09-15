# Song lyrics Generator
This repo focusses on using LSTMs to generate song lyrics.

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install tensorflow
pip install keras
pip install nltk
pip install pandas
```

## Data

Data used for the project was obtained from Kaggle. Link to the dataset: https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics

## Usage
* Download dataset from the above source and replace the empty file in data directory named as "lyrics.csv".
* Run the project using the command below
* You can tweak around the hyper-parameters being used for training the model.
* Since the dataset is huge, currently for paging overflow, only 10k rows are being used.
```bash
python main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
