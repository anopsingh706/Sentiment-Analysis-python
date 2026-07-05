import unittest

import pandas as pd

from csv_upload import build_batch_results, extract_text_rows, find_text_column
from sentiment_utils import analyze_text


def fake_analyzer(text):
    return {
        "label": "Positive" if "great" in text.lower() else "Negative",
        "polarity": 0.5 if "great" in text.lower() else -0.5,
        "subjectivity": 0.3,
    }


class CsvUploadTests(unittest.TestCase):
    def test_finds_xquik_tweet_text_column(self):
        dataframe = pd.DataFrame(
            {
                "Tweet Created At": ["2026-07-05T00:00:00Z"],
                "Tweet Text": ["Great service"],
            }
        )

        self.assertEqual(find_text_column(dataframe), "Tweet Text")

    def test_extracts_non_empty_rows(self):
        dataframe = pd.DataFrame({"feedback": ["Great app", " ", None, "Slow queue"]})

        text_column, text_rows = extract_text_rows(dataframe)

        self.assertEqual(text_column, "feedback")
        self.assertEqual(text_rows.tolist(), ["Great app", "Slow queue"])

    def test_builds_batch_results(self):
        dataframe = pd.DataFrame({"Tweet Text": ["Great app", "Slow queue"]})

        results = build_batch_results(dataframe, analyzer=fake_analyzer)

        self.assertEqual(results["sentiment"].tolist(), ["Positive", "Negative"])
        self.assertEqual(results["polarity"].tolist(), [0.5, -0.5])
        self.assertEqual(results["source_text_column"].tolist(), ["Tweet Text", "Tweet Text"])

    def test_analyze_text_returns_neutral_for_blank_text(self):
        result = analyze_text("")

        self.assertEqual(result["label"], "Neutral")
        self.assertEqual(result["polarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
