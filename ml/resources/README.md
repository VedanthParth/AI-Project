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

`junyi_topics_en.csv` names Junyi's topic hierarchy, which the dataset only gives as hashed ids:
the 10 strands (`level2_id`, e.g. elementary *Word Problems*, junior *Algebra*) and their 42
units (`level3_id`, e.g. *Fractions and Ratios*, *Quadratic Equations*). The app shows strands as
"areas of maths" and units as "chapters" in the new-student questionnaire. The names were
written by an AI model (Claude) from each group's exercise names, in the style of Indian (NCERT)
maths chapter titles, e.g. *Data Handling*, *Mensuration*, *Order of Operations (BODMAS)*.

Both files use Indian English: British spellings (litre, centimetre, centre, factorisation) and
Indian terms (HCF and LCM, Pythagoras theorem). The app also labels Junyi's school stages by the
Indian classes they roughly match: elementary (Taiwan's grades 1–6) as Class 1–6, junior high
(grades 7–9) as Class 7–9, senior high as Class 10–12.

| Column | Meaning |
| --- | --- |
| `kind` | `strand` (level 2) or `unit` (level 3) |
| `id` | Junyi's `level2_id` or `level3_id` |
| `parent` | For a unit, its strand's id |
| `stage` | `learning_stage` of its exercises: `elementary`, `junior` or `senior` |
| `exercises` | How many exercises it holds |
| `name_en` | English name |

Source: the [Junyi Academy Online Learning Activity Dataset](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy)
by Junyi Academy, licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
This file is derived from it and is shared under the same license.
