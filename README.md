# end-to-end-ml-suite
A machine learning model that predicts insurance costs, classify scenery images, and classifies a blog into one of 5 classes.
Using these datasets: 
1. Nurmerical (https://www.kaggle.com/datasets/mirichoi0218/insurance)
2. Images (https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
3. Text (https://github.com/derekgreene/bbc-datasets)

**Using python 3.12** 

# How to get started with development:
*For linux / macOS*

```
git clone https://github.com/RealAhmedKhairi/end-to-end-ml-suite.git
cd end-to-end-ml-suite
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*For windows*

```
git clone https://github.com/RealAhmedKhairi/end-to-end-ml-suite.git
cd end-to-end-ml-suite
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

# Commit conventions to use: 
```
1) feat(feature): 
when adding new code or a new feature.
2) fix: 
fixing a bug or an issue.
3) perf(performance): 
Use this when your code change makes the program run faster or use less memory.
4) style(formatting): 
5) chore(maintainance): 
Use this for background tasks that don't change the actual Python source code, like updating .gitignore, requirements.txt, or Docker configs.Example: chore: pin scikit-learn to version 1.4.2
formatted existing code.
```