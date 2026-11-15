# Detector comparison

## gap test

F1 (higher is better)

| detector     |    en |    hi |   hinglish |    mr |   all |
|:-------------|------:|------:|-----------:|------:|------:|
| protectai-v2 | 0.82  | 0.69  |      0.783 | 0.682 | 0.735 |
| baseline     | 0.921 | 0.909 |      0.897 | 0.929 | 0.914 |
| kavach-small | 0.952 | 0.983 |      0.867 | 0.949 | 0.938 |

False alarm rate (lower is better)

| detector     |    en |   hi |   hinglish |    mr |   all |
|:-------------|------:|-----:|-----------:|------:|------:|
| protectai-v2 | 0.2   |  0.9 |      0.4   | 0.867 | 0.592 |
| baseline     | 0.133 |  0   |      0.067 | 0     | 0.05  |
| kavach-small | 0.1   |  0   |      0.133 | 0.033 | 0.067 |

<details><summary>Full table</summary>

| detector     | lang     | attacks_caught   |   recall | false_alarms   |   false_alarm_rate |    f1 |   auroc |
|:-------------|:---------|:-----------------|---------:|:---------------|-------------------:|------:|--------:|
| protectai-v2 | en       | 25/30            |    0.833 | 6/30           |              0.2   | 0.82  |   0.884 |
| protectai-v2 | hi       | 30/30            |    1     | 27/30          |              0.9   | 0.69  |   0.684 |
| protectai-v2 | hinglish | 27/30            |    0.9   | 12/30          |              0.4   | 0.783 |   0.791 |
| protectai-v2 | mr       | 29/30            |    0.967 | 26/30          |              0.867 | 0.682 |   0.784 |
| protectai-v2 | all      | 111/120          |    0.925 | 71/120         |              0.592 | 0.735 |   0.775 |
| baseline     | en       | 29/30            |    0.967 | 4/30           |              0.133 | 0.921 |   0.947 |
| baseline     | hi       | 25/30            |    0.833 | 0/30           |              0     | 0.909 |   0.989 |
| baseline     | hinglish | 26/30            |    0.867 | 2/30           |              0.067 | 0.897 |   0.956 |
| baseline     | mr       | 26/30            |    0.867 | 0/30           |              0     | 0.929 |   0.981 |
| baseline     | all      | 106/120          |    0.883 | 6/120          |              0.05  | 0.914 |   0.962 |
| kavach-small | en       | 30/30            |    1     | 3/30           |              0.1   | 0.952 |   0.962 |
| kavach-small | hi       | 29/30            |    0.967 | 0/30           |              0     | 0.983 |   1     |
| kavach-small | hinglish | 26/30            |    0.867 | 4/30           |              0.133 | 0.867 |   0.964 |
| kavach-small | mr       | 28/30            |    0.933 | 1/30           |              0.033 | 0.949 |   0.998 |
| kavach-small | all      | 113/120          |    0.942 | 8/120          |              0.067 | 0.938 |   0.984 |

</details>

## test split

F1 (higher is better)

| detector     |    en |    hi |   hinglish |    mr |   all |
|:-------------|------:|------:|-----------:|------:|------:|
| protectai-v2 | 0.841 | 0.292 |      0.447 | 0.296 | 0.531 |
| baseline     | 0.887 | 0.491 |      0.693 | 0.517 | 0.787 |
| kavach-small | 0.919 | 0.753 |      0.775 | 0.84  | 0.873 |

False alarm rate (lower is better)

| detector     |    en |    hi |   hinglish |    mr |   all |
|:-------------|------:|------:|-----------:|------:|------:|
| protectai-v2 | 0.052 | 0.771 |      0.197 | 0.787 | 0.446 |
| baseline     | 0.083 | 0.037 |      0.069 | 0.037 | 0.056 |
| kavach-small | 0.088 | 0.064 |      0.069 | 0.059 | 0.069 |

<details><summary>Full table</summary>

| detector     | lang     | attacks_caught   |   recall | false_alarms   |   false_alarm_rate |    f1 |   auroc |
|:-------------|:---------|:-----------------|---------:|:---------------|-------------------:|------:|--------:|
| protectai-v2 | en       | 146/191          |    0.764 | 10/193         |              0.052 | 0.841 |   0.927 |
| protectai-v2 | hi       | 31/36            |    0.861 | 145/188        |              0.771 | 0.292 |   0.608 |
| protectai-v2 | hinglish | 21/36            |    0.583 | 37/188         |              0.197 | 0.447 |   0.812 |
| protectai-v2 | mr       | 32/36            |    0.889 | 148/188        |              0.787 | 0.296 |   0.688 |
| protectai-v2 | all      | 237/316          |    0.75  | 340/763        |              0.446 | 0.531 |   0.763 |
| baseline     | en       | 165/191          |    0.864 | 16/193         |              0.083 | 0.887 |   0.959 |
| baseline     | hi       | 14/36            |    0.389 | 7/188          |              0.037 | 0.491 |   0.936 |
| baseline     | hinglish | 26/36            |    0.722 | 13/188         |              0.069 | 0.693 |   0.961 |
| baseline     | mr       | 15/36            |    0.417 | 7/188          |              0.037 | 0.517 |   0.943 |
| baseline     | all      | 233/316          |    0.737 | 43/763         |              0.056 | 0.787 |   0.958 |
| kavach-small | en       | 177/191          |    0.927 | 17/193         |              0.088 | 0.919 |   0.985 |
| kavach-small | hi       | 29/36            |    0.806 | 12/188         |              0.064 | 0.753 |   0.961 |
| kavach-small | hinglish | 31/36            |    0.861 | 13/188         |              0.069 | 0.775 |   0.975 |
| kavach-small | mr       | 34/36            |    0.944 | 11/188         |              0.059 | 0.84  |   0.985 |
| kavach-small | all      | 286/316          |    0.905 | 53/763         |              0.069 | 0.873 |   0.981 |

</details>
