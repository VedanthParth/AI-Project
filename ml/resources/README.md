# Junyi concept names in English

`junyi_concepts_en.csv` gives an English name for each of the 1,330 exercises in Junyi's
`Info_Content.csv`, keyed by `ucid`.

| Column | Meaning |
| --- | --- |
| `ucid` | Junyi's exercise id |
| `name_en` | English name; the level, when the original has one, is in parentheses |
| `level` | `basic` (基礎), `standard` (一般), `advanced` (進階) or `worked example` (例題) |
| `retiring` | 1 if Junyi marked the exercise as about to be retired (即將下架) |
| `name_zh` | The original `content_pretty_name` |

The names were machine-translated once by an AI model (Claude) and have not been reviewed by a
native speaker; treat them as display labels. The paper's results don't depend on them.

Source: the [Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy)
by Junyi Academy, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
This file is derived from it and is shared under the same license.
