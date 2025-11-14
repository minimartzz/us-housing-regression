import math
from pathlib import Path
from joblib import load

from src.training_pipeline.train import train_model
from src.training_pipeline.eval import evaluate_model

TRAIN_PATH = Path("data/clean/feature_engineered_train.csv")
EVAL_PATH = Path("data/clean/feature_engineered_valid.csv")

def _assert_metrics(m):
  assert set(m.keys()) == {'mae', 'rmse', 'r2'}
  assert all(isinstance(v, float) and math.isfinite(v) for v in m.values())


# ---------- Test train.py ----------
def test_train_creates_model_and_metrics(tmp_path):
  out_path = tmp_path / "xbg_model.pkl"
  _, metrics = train_model(
    train_path=TRAIN_PATH,
    eval_path=EVAL_PATH,
    model_output=out_path,
    model_params={"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
    sample_frac=0.02,
  )
  assert out_path.exists()
  _assert_metrics(metrics)
  model = load(out_path)
  assert model is not None
  print("✅ train_model test passed")


# ---------- Test eval.py ----------
def test_eval_works_with_saved_model(tmp_path):
  model_path = tmp_path / "xbg_model.pkl"
  train_model(
    train_path=TRAIN_PATH,
    eval_path=EVAL_PATH,
    model_output=model_path,
    model_params={"n_estimators": 20},
    sample_frac=0.02,
  )
  metrics = evaluate_model(model_path=model_path, eval_path=EVAL_PATH, sample_frac=0.02)
  _assert_metrics(metrics)
  print("✅ evaluate_model test passed")