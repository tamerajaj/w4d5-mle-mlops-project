# MLOps Project


Please do not fork this repository, but use this repository as a template for your MLOps project. Make Pull Requests to your own repository even if you work alone and mark the checkboxes with an x, if you are done with a topic in the pull request message.

## Project for today
The task for today you can find in the [project-description.md](project-description.md) file.



## Environment

You need the google cloud sdk installed and configured. If you don't have it installed follow the instructions here or use homebrew on mac:
```bash
brew install --cask google-cloud-sdk
```

Python environment:

```bash
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```