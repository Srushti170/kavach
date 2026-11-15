# Dataset stats

Total examples: 8513. Removed while cleaning: {'duplicate': 1}.

## By split

| split   |   normal |   attack |
|:--------|---------:|---------:|
| train   |     4267 |     2183 |
| val     |      725 |      259 |
| test    |      763 |      316 |

## By split and language

|                       |   normal |   attack |
|:----------------------|---------:|---------:|
| ('test', 'de')        |        6 |       17 |
| ('test', 'en')        |      193 |      191 |
| ('test', 'hi')        |      188 |       36 |
| ('test', 'hinglish')  |      188 |       36 |
| ('test', 'mr')        |      188 |       36 |
| ('train', 'de')       |       45 |       43 |
| ('train', 'en')       |     1081 |     1216 |
| ('train', 'hi')       |     1047 |      308 |
| ('train', 'hinglish') |     1047 |      308 |
| ('train', 'mr')       |     1047 |      308 |
| ('val', 'de')         |        7 |        5 |
| ('val', 'en')         |      181 |      164 |
| ('val', 'hi')         |      179 |       30 |
| ('val', 'hinglish')   |      179 |       30 |
| ('val', 'mr')         |      179 |       30 |

## By source

| source              |   normal |   attack |
|:--------------------|---------:|---------:|
| deepset             |      399 |      263 |
| english-to-hinglish |      900 |        0 |
| gandalf             |        0 |      999 |
| opus-100            |     2400 |        0 |
| seed                |     2056 |     1496 |

## Shortcut check (seed examples, train split)

Surface features should look similar for attacks and normal texts.

| label   |   avg_chars |   has_email_% |   has_link_% |   devanagari_% |   html_tag_% |
|:--------|------------:|--------------:|-------------:|---------------:|-------------:|
| attack  |       237.4 |          44.8 |         32.7 |           54.3 |         29.1 |
| normal  |       202.9 |          32.5 |         29.4 |           54.3 |         24.8 |

## Licenses

- Seeds: written for this project.
- deepset/prompt-injections: Apache-2.0.
- Lakera/gandalf_ignore_instructions: MIT.
- findnitai/english-to-hinglish: Apache-2.0.
- Helsinki-NLP/opus-100: mixed licenses from the OPUS corpora (listed as unknown); drop `source == "opus-100"` rows before redistributing.
