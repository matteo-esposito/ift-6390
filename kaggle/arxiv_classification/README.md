# Arxiv Classification
## Fundamentals of Machine Learning

### Usage

To run the file, provide the type of algorithm ["random" or "NBbernoulli"], absolute path of the raw training and testing csv files as command line arguments to this following line.

```bash
python3 Esposito-Main.py <algorithm> <path_to_training> <path_to_testing> <output_path>

# Example
python3 Esposito-Main.py NBbernoulli data/train.csv data/test.csv data/NBBsubmission.csv
```

This will output the submission file in the output_path.