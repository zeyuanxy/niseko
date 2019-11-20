python3 -m flake8 --config=tools/build-support/flake8 niseko setup.py
python3 -m pylint --rcfile=tools/build-support/pylintrc --output-format=parseable --jobs=4 niseko
