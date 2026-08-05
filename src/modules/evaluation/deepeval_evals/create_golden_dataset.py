from deepeval.dataset import EvaluationDataset

from src.settings import DATASETS_DIR


EVALUATION_DATASET_PATH = DATASETS_DIR / "ai_helpdesk" / "evaluation.csv"

dataset = EvaluationDataset()
dataset.add_goldens_from_csv_file(EVALUATION_DATASET_PATH, input_col_name="question", expected_output_col_name="answer")
dataset.save_as(file_type="json", directory="./dataset", file_name="golden_dataset", include_test_cases=True)
