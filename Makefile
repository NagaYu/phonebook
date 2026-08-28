# Phonebook - full pipeline
#
# For real data pass CSV=... ENCODING=cp932.
# With no arguments everything runs end to end on synthetic data (whose numbers
# are NOT evidence for any claim; RESULTS.md carries a warning in that case).

PY ?= python3
DATA ?= data/processed
MODEL ?= artifacts/model
CSV ?= data/raw/synthetic.csv
ENCODING ?= utf-8
PRESET ?= small
EPOCHS ?= 12
N ?= 60000

.PHONY: all synthetic dataset en2kana train eval figures export fill demo test clean

all: dataset train eval figures export

synthetic:
	$(PY) scripts/make_synthetic.py --out data/raw/synthetic.csv --n $(N)

dataset:
	@test -f $(CSV) || $(MAKE) synthetic
	$(PY) scripts/build_dataset.py --csv $(CSV) --encoding $(ENCODING) --out $(DATA) \
		$(if $(findstring synthetic,$(CSV)),--synthetic,)

en2kana:
	$(PY) scripts/build_en2kana_data.py --data $(DATA) --out $(DATA)/en2kana

train:
	$(PY) scripts/train.py --data $(DATA) --out $(MODEL) --preset $(PRESET) --epochs $(EPOCHS)

eval:
	$(PY) scripts/evaluate.py --data $(DATA) --model $(MODEL) --out benchmarks

figures:
	$(PY) scripts/make_figures.py --results benchmarks/results.json --out figures

export:
	$(PY) scripts/export.py --model $(MODEL) --out artifacts/export

fill:
	$(PY) scripts/fill_missing.py --data $(DATA) --model $(MODEL) \
		--out artifacts/derived/furigana_estimates.jsonl

demo:
	PHONEBOOK_MODEL=$(MODEL) $(PY) app.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf $(DATA) artifacts/export artifacts/derived benchmarks/results.json benchmarks/RESULTS.md figures/*.png
