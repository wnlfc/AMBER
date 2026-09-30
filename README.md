# AMBER

Reference code for **AMBER: Multi-View Adaptive Budget Allocation for Listwise Vision-Language Reranking**.

`elo_rerank.py` combines online Elo updates, head-aware observation-margin view selection, and view-conditioned pairwise-entropy query scheduling. It uses one initial VLM call per query, then allocates the remaining global budget adaptively. The VLM is reached through an OpenAI-compatible API.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Serve `Qwen3-VL-8B-Instruct` through an OpenAI-compatible endpoint, or set `MODEL` to the name exposed by your server. The default endpoint is `http://localhost:8005/v1`. For an authenticated endpoint, set `OPENAI_API_KEY` in the environment.

## Inputs

Retrieval results and images are **not distributed in this repository**. Supply your own first-stage retrieval JSON and local image directory. The expected JSON has a `config` object and a `cases` array. Each case contains `query_id`, `query`, and a candidate list named `retrieved_top50` or `retrieved_top100`; composed image retrieval also contains `reference`. Candidate entries are image filenames. `--image_dir` overrides any path in `config`.

The paper uses `Qwen3-VL-Embedding-8B` for first-stage retrieval, a pool of 50 candidates, and a view of 10. Reproducing its numbers requires the same retrieval results, image files, serving setup, and evaluation protocol.

## Run

For CIRR, CIRCO, or one PhotoBench subset/language split:

```bash
RETRIEVAL_JSON=/path/to/retrieval.json \
IMAGE_DIR=/path/to/images \
MODEL=qwen3-vl-8b-instruct \
bash examples/run_cirr.sh
```

Replace the script with `examples/run_circo.sh` or `examples/run_photobench.sh` as appropriate. Set `OUTPUT` to change the output path. Extra command-line flags pass through, for example `bash examples/run_cirr.sh --budget_per_query 8`. For a small smoke run, add `--num_queries 2 --budget_per_query 4`.

The scripts set only dataset-specific image preprocessing: CIRR long side 768 px, CIRCO original bytes, PhotoBench long side 1120 px. Each PhotoBench split needs a separate invocation with its matching image directory.

## Evaluation

CIRR and CIRCO use their official evaluation servers. Convert a complete AMBER run to the submission format with the same retrieval file used for reranking:

```bash
python3 tools/make_submission.py --dataset cirr \
  --retrieval /path/to/cirr-retrieval.json \
  --rerank outputs/cirr/amber.jsonl \
  --output submissions/cirr.json
```

Use `--dataset circo` and the matching files for CIRCO. The converter checks that every query has a complete top-50 ranking and then writes the dataset-specific format.

PhotoBench metrics are computed locally from the rerank outputs and their retrieval JSONs:

```bash
python3 tools/evaluate_photobench.py \
  --retrieval /path/to/subset03-cn.json \
  --rerank outputs/photobench/amber-subset03-cn.jsonl
```

You can pass multiple paired files after `--retrieval` and `--rerank` to aggregate album subsets for one language. Report CN and EN separately. Recall@K is the fraction of ground-truth images in the top K; NDCG@K uses binary relevance. Queries with empty ground truth stay in the denominator and score zero. The evaluator rejects missing queries and failed runs.

## Main configuration

| Parameter | Default |
| --- | --- |
| Candidate pool / view | `--top_n 50 --view_size 10` |
| Candidate selection | `--cand_sv head_obs_margin --cand_margin_c 50 --cand_margin_k 2` |
| Query scheduling | `--query_sv view_pair_entropy --batch_size 16` |
| Elo | `--elo_base_rating 1500 --elo_rating_range 400 --elo_k 16 --elo_scale 400 --elo_max_pair_gap 0` |
| Budget | `--budget_per_query 24`; per-query cap defaults to `min(top_n, 2 × average budget)` |
| Presentation order | `--shuffle_seed 42` |

These defaults follow the recorded main-run protocol in the later experiment package. A positive `--total_budget` overrides `--budget_per_query`; `--max_per_query 0` removes the per-query cap. Run `python3 elo_rerank.py --help` for other controls. The output is JSONL, with checkpoint and scheduling sidecar files next to it.
