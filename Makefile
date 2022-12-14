# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* brief-news/*.py

black:
	@black scripts/* brief-news/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr brief-news-*.dist-info
	@rm -fr brief-news.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      PACKAGE ACTIONS
# ----------------------------------

run_api_scraper:
	python -c 'from brief_news.interface.main import get_articles; get_articles("business")'

run_transformer_summarizer:
	python -c 'from brief_news.interface.main import transfomer_summaries; transfomer_summaries("business")'

run_preprocess:
	python -c 'from brief_news.ml_logic.main_train import preprocess; preprocess("train")'

run_test_model_train:
	python -c 'from brief_news.ml_logic.NN_model import test_train_model; test_train_model()'

run_model_train:
	python -c 'from brief_news.ml_logic.NN_model import full_train; full_train()'

run_telebot:
	python -c 'from brief_news.telebot.main import main; main()'
