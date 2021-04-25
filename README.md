# News generation

Fake news generation using Markov chains (n-gram language modeling) and long-short term memory network (LSTM).

### Usage

Install dependencies (keras, tensorflow, numpy, pandas):

```bash
# Upgrade your pip3
python3 -m pip install --upgrade pip

# Install deps
pip3 install -U -r src/requirements.txt
```

Then, clean up your CSV containing tweets (e.g., remove stopwords):

```bash
python3 src/data_processing <input.csv> <output_clean.csv>
```

If you want to run n-gram model, run the following command with the output csv file from the previous one:

```bash
python3 src/model_ngrams.py <output_clean.csv>
```

If you would like to run LSTM model based on characters, run the following the same way:

```bash
python3 src/model_lstm_chars.py <output_clean.csv>
```

These will either output results to command line, or generate a file called `generated_text_rnn_chars.txt`. Feel free to modify, play with the code, and contribute if you find a bug.

### Project report

To see the report, click [here](./results/Batyr_report.pdf).

### Data clean-up

Code [here](./src/data_processing.py)

### Examples of generated text

Using [n-grams (markov chain)](./results/5gram_200size.txt)

Using [LSTM](./results/generated_text_rnn_chars.txt)
