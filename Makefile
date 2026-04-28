.PHONY: sync-pretrained sync-full check-pretrained check-full notebook-pretrained notebook-full evaluate evaluate-zero-shot

sync-pretrained:
	uv sync --group pretrained --group notebook

sync-full:
	uv sync --group full --group notebook

check-pretrained:
	uv run python scripts/check_environment.py --mode pretrained

check-full:
	uv run python scripts/check_environment.py --mode full

notebook-pretrained:
	uv run jupyter lab notebooks/pretrained_laguagebind.ipynb

notebook-full:
	uv run jupyter lab notebooks/complete_languagebind.ipynb

evaluate:
	uv run python scripts/evaluate_model.py \
		--index-dir indexes/proxy_720p30_w15_s5 \
		--mode manual \
		--ground-truth data/ground_truth/ground_truth_dataset.json \
		--adapter data/models/helldivers_adapter.pth

evaluate-zero-shot:
	uv run python scripts/evaluate_model.py \
		--index-dir indexes/proxy_720p30_w15_s5 \
		--mode manual \
		--ground-truth data/ground_truth/ground_truth_dataset.json \
		--adapter missing_zero_shot_adapter.pth
