# WiCER reproduction report

Profile: `paper`  ·  judge: `anthropic/claude-haiku-5.5`  ·  compiler: `anthropic/claude-haiku-5.5`

Reader: `models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`, n_ctx=98304, KV=q8_0/q8_0


## Per topic and condition

| Topic | Condition | n | Mean | %>=4 | %=1 | TTFT med (s) | tok/s | ctx tok |
|---|---|---|---|---|---|---|---|---|
| company_policies | fc_raw | 400 | 3.58 | 61.0 | 8.5 | 0.147 | 28.3 | 54779 |
| company_policies | wiki_light | 400 | 2.95 | 37.2 | 17.8 | 0.054 | 62.3 | 10144 |
| company_policies | wiki_moderate | 400 | 2.93 | 34.2 | 15.8 | 0.056 | 59.4 | 11134 |
| company_policies | wiki_aggressive | 400 | 2.91 | 36.5 | 16.2 | 0.054 | 61.7 | 9576 |
| company_policies | wicer_iter2 | 400 | 3.20 | 44.8 | 12.5 | 0.057 | 59.0 | 11370 |
| company_policies | random_blind | 400 | 2.94 | 36.8 | 17.0 | 0.056 | 60.2 | 10849 |
| company_policies | random_pin | 400 | 2.90 | 36.5 | 19.2 | 0.056 | 59.8 | 10792 |
| cybersecurity_news | fc_raw | 400 | 3.70 | 64.2 | 8.2 | 0.239 | 19.4 | 90765 |
| cybersecurity_news | wiki_light | 400 | 3.11 | 43.2 | 18.0 | 0.072 | 48.5 | 20409 |
| cybersecurity_news | wiki_moderate | 400 | 3.03 | 40.8 | 19.0 | 0.082 | 42.7 | 26306 |
| cybersecurity_news | wiki_aggressive | 400 | 2.94 | 40.5 | 23.5 | 0.077 | 45.1 | 23334 |
| cybersecurity_news | wicer_iter1 | 400 | 3.01 | 39.0 | 19.0 | 0.068 | 50.5 | 17722 |
| cybersecurity_news | random_blind | 400 | 2.85 | 39.8 | 28.0 | 0.061 | 55.3 | 14178 |
| cybersecurity_news | random_pin | 400 | 2.90 | 37.8 | 23.8 | 0.073 | 47.7 | 20285 |
| incident_report | fc_raw | 400 | 3.40 | 56.8 | 18.2 | 0.241 | 19.8 | 90327 |
| incident_report | wiki_light | 400 | 2.65 | 30.2 | 30.5 | 0.072 | 49.3 | 20084 |
| incident_report | wiki_moderate | 400 | 2.56 | 30.0 | 32.5 | 0.061 | 55.4 | 14693 |
| incident_report | wiki_aggressive | 400 | 2.77 | 35.0 | 28.0 | 0.066 | 52.0 | 16592 |
| incident_report | wicer_iter2 | 400 | 2.86 | 39.2 | 28.0 | 0.071 | 48.8 | 19708 |
| incident_report | random_blind | 400 | 2.69 | 33.2 | 30.0 | 0.073 | 46.9 | 21194 |
| incident_report | random_pin | 400 | 3.04 | 41.8 | 18.8 | 0.114 | 33.6 | 41202 |
| local_arts_and_culture | fc_raw | 400 | 3.56 | 60.0 | 13.2 | 0.245 | 19.4 | 93628 |
| local_arts_and_culture | wiki_light | 400 | 2.26 | 22.5 | 42.5 | 0.060 | 56.8 | 13709 |
| local_arts_and_culture | wiki_moderate | 400 | 2.70 | 30.8 | 28.5 | 0.071 | 49.6 | 19112 |
| local_arts_and_culture | wiki_aggressive | 400 | 2.71 | 32.5 | 26.2 | 0.069 | 51.0 | 17710 |
| local_arts_and_culture | wicer_iter1 | 400 | 2.46 | 26.0 | 35.5 | 0.068 | 51.6 | 17125 |
| local_arts_and_culture | random_blind | 400 | 2.64 | 31.2 | 30.2 | 0.070 | 49.5 | 18915 |
| local_arts_and_culture | random_pin | 400 | 2.42 | 27.8 | 39.2 | 0.062 | 54.3 | 14614 |
| local_economy | fc_raw | 400 | 3.63 | 62.2 | 11.0 | 0.236 | 19.8 | 91642 |
| local_economy | wiki_light | 400 | 2.71 | 33.8 | 29.0 | 0.063 | 54.7 | 15256 |
| local_economy | wiki_moderate | 400 | 2.75 | 32.2 | 26.0 | 0.077 | 46.5 | 22408 |
| local_economy | wiki_aggressive | 400 | 2.48 | 26.0 | 33.8 | 0.061 | 55.5 | 14253 |
| local_economy | wicer_iter1 | 400 | 2.83 | 36.2 | 25.0 | 0.072 | 48.0 | 20932 |
| local_economy | random_blind | 400 | 3.06 | 41.8 | 18.2 | 0.072 | 49.4 | 19835 |
| local_economy | random_pin | 400 | 2.73 | 33.2 | 25.8 | 0.065 | 52.7 | 16764 |
| local_education | fc_raw | 400 | 3.50 | 59.5 | 15.8 | 0.232 | 20.0 | 88904 |
| local_education | wiki_light | 400 | 2.70 | 32.8 | 28.8 | 0.070 | 50.0 | 18948 |
| local_education | wiki_moderate | 400 | 2.78 | 34.5 | 26.2 | 0.065 | 52.1 | 16753 |
| local_education | wiki_aggressive | 400 | 2.77 | 34.2 | 26.2 | 0.065 | 51.6 | 16986 |
| local_education | wicer_iter1 | 400 | 2.88 | 38.0 | 23.2 | 0.062 | 53.5 | 15422 |
| local_education | random_blind | 400 | 2.85 | 36.0 | 25.0 | 0.066 | 52.0 | 17213 |
| local_education | random_pin | 400 | 2.44 | 25.8 | 36.8 | 0.059 | 56.4 | 13075 |
| local_env_issues | fc_raw | 400 | 3.57 | 59.5 | 11.2 | 0.239 | 19.5 | 91218 |
| local_env_issues | wiki_light | 400 | 2.89 | 37.0 | 22.2 | 0.071 | 49.1 | 19542 |
| local_env_issues | wiki_moderate | 400 | 2.60 | 29.5 | 30.5 | 0.059 | 55.7 | 13606 |
| local_env_issues | wiki_aggressive | 400 | 2.96 | 37.8 | 20.2 | 0.067 | 50.8 | 17274 |
| local_env_issues | wicer_iter2 | 400 | 3.00 | 40.5 | 21.5 | 0.069 | 49.4 | 18361 |
| local_env_issues | random_blind | 400 | 2.94 | 39.5 | 21.5 | 0.066 | 51.2 | 16490 |
| local_env_issues | random_pin | 400 | 2.50 | 27.8 | 35.0 | 0.061 | 56.6 | 13838 |
| local_health | fc_raw | 400 | 3.61 | 62.8 | 15.2 | 0.227 | 20.1 | 88018 |
| local_health | wiki_light | 400 | 3.05 | 43.5 | 21.2 | 0.070 | 50.0 | 19055 |
| local_health | wiki_moderate | 400 | 2.73 | 34.2 | 28.0 | 0.066 | 51.0 | 17514 |
| local_health | wiki_aggressive | 400 | 2.83 | 38.5 | 28.5 | 0.067 | 50.6 | 18217 |
| local_health | wicer_iter1 | 400 | 3.02 | 41.2 | 20.5 | 0.068 | 51.1 | 17809 |
| local_health | random_blind | 400 | 2.65 | 30.0 | 28.5 | 0.063 | 52.9 | 15919 |
| local_health | random_pin | 400 | 2.51 | 28.8 | 36.2 | 0.062 | 53.6 | 15356 |
| local_news | fc_raw | 400 | 3.60 | 63.2 | 13.8 | 0.231 | 19.7 | 88733 |
| local_news | wiki_light | 400 | 2.76 | 34.5 | 27.5 | 0.062 | 55.4 | 14537 |
| local_news | wiki_moderate | 400 | 2.84 | 35.5 | 25.5 | 0.068 | 50.7 | 18268 |
| local_news | wiki_aggressive | 400 | 2.87 | 35.8 | 23.5 | 0.062 | 54.6 | 14854 |
| local_news | wicer_iter1 | 400 | 2.70 | 33.0 | 28.5 | 0.063 | 53.2 | 15150 |
| local_news | random_blind | 400 | 2.75 | 33.5 | 25.5 | 0.065 | 51.8 | 16942 |
| local_news | random_pin | 400 | 2.86 | 37.2 | 25.0 | 0.068 | 50.6 | 17827 |
| local_politics | fc_raw | 400 | 3.62 | 62.2 | 14.0 | 0.240 | 19.9 | 91359 |
| local_politics | wiki_light | 400 | 2.71 | 33.0 | 28.2 | 0.065 | 53.6 | 16531 |
| local_politics | wiki_moderate | 400 | 2.87 | 37.5 | 25.2 | 0.070 | 49.9 | 18934 |
| local_politics | wiki_aggressive | 400 | 2.77 | 33.8 | 26.2 | 0.065 | 53.9 | 15630 |
| local_politics | wicer_iter1 | 400 | 2.62 | 30.0 | 31.2 | 0.062 | 55.2 | 14478 |
| local_politics | random_blind | 400 | 2.95 | 41.0 | 23.8 | 0.075 | 47.8 | 21150 |
| local_politics | random_pin | 400 | 2.52 | 28.5 | 34.2 | 0.065 | 52.7 | 16760 |
| local_sports | fc_raw | 400 | 3.78 | 68.2 | 11.5 | 0.236 | 19.9 | 92502 |
| local_sports | wiki_light | 400 | 3.02 | 43.2 | 25.2 | 0.062 | 55.8 | 15207 |
| local_sports | wiki_moderate | 400 | 3.04 | 42.2 | 24.2 | 0.063 | 55.0 | 15184 |
| local_sports | wiki_aggressive | 400 | 2.75 | 36.0 | 31.0 | 0.058 | 57.4 | 12446 |
| local_sports | wicer_iter1 | 400 | 3.09 | 44.0 | 20.0 | 0.068 | 51.8 | 18187 |
| local_sports | random_blind | 400 | 2.97 | 42.5 | 26.2 | 0.063 | 54.9 | 15785 |
| local_sports | random_pin | 400 | 2.70 | 34.2 | 34.0 | 0.061 | 54.4 | 14751 |
| local_tech | fc_raw | 400 | 3.79 | 68.5 | 11.5 | 0.242 | 19.6 | 92916 |
| local_tech | wiki_light | 400 | 2.88 | 39.2 | 26.0 | 0.064 | 54.8 | 15784 |
| local_tech | wiki_moderate | 400 | 3.04 | 42.5 | 20.5 | 0.066 | 53.2 | 16563 |
| local_tech | wiki_aggressive | 400 | 3.06 | 44.2 | 20.5 | 0.068 | 52.6 | 17465 |
| local_tech | wicer_iter2 | 400 | 3.25 | 51.2 | 17.2 | 0.069 | 51.4 | 18466 |
| local_tech | random_blind | 400 | 3.05 | 43.2 | 21.5 | 0.069 | 50.7 | 18071 |
| local_tech | random_pin | 400 | 2.98 | 40.0 | 22.2 | 0.067 | 52.6 | 17144 |
| neighborhood | fc_raw | 400 | 3.62 | 61.5 | 10.5 | 0.238 | 19.8 | 90554 |
| neighborhood | wiki_light | 400 | 2.68 | 32.5 | 31.0 | 0.069 | 51.4 | 18295 |
| neighborhood | wiki_moderate | 400 | 2.85 | 35.8 | 25.0 | 0.068 | 52.3 | 17278 |
| neighborhood | wiki_aggressive | 400 | 2.69 | 31.2 | 28.5 | 0.063 | 54.6 | 15168 |
| neighborhood | wicer_iter1 | 400 | 2.87 | 37.0 | 24.5 | 0.067 | 51.4 | 17340 |
| neighborhood | random_blind | 400 | 2.79 | 35.0 | 25.5 | 0.064 | 54.2 | 15321 |
| neighborhood | random_pin | 400 | 2.56 | 28.5 | 31.8 | 0.062 | 55.2 | 14321 |
| news_stories | fc_raw | 400 | 3.72 | 65.5 | 9.2 | 0.249 | 19.7 | 92887 |
| news_stories | wiki_light | 400 | 2.94 | 39.5 | 22.0 | 0.069 | 53.1 | 16967 |
| news_stories | wiki_moderate | 400 | 2.77 | 32.0 | 25.5 | 0.073 | 48.5 | 19861 |
| news_stories | wiki_aggressive | 400 | 2.78 | 36.8 | 26.8 | 0.064 | 55.1 | 14989 |
| news_stories | wicer_iter1 | 400 | 2.87 | 36.5 | 25.0 | 0.069 | 51.7 | 17399 |
| news_stories | random_blind | 400 | 2.49 | 26.2 | 34.5 | 0.058 | 59.0 | 12025 |
| news_stories | random_pin | 400 | 2.73 | 34.0 | 28.0 | 0.067 | 52.8 | 16120 |
| regional_cuisine | fc_raw | context_overflow | -- | -- | -- | -- | -- | -- |
| regional_cuisine | wiki_light | 400 | 2.95 | 39.8 | 23.2 | 0.070 | 52.2 | 18546 |
| regional_cuisine | wiki_moderate | 400 | 2.85 | 37.8 | 26.8 | 0.074 | 49.7 | 19956 |
| regional_cuisine | wiki_aggressive | 400 | 2.77 | 35.8 | 28.5 | 0.076 | 47.0 | 21091 |
| regional_cuisine | wicer_iter1 | 400 | 2.55 | 30.2 | 34.2 | 0.070 | 50.1 | 18610 |
| regional_cuisine | random_blind | 400 | 2.79 | 35.8 | 26.2 | 0.068 | 52.7 | 16976 |
| regional_cuisine | random_pin | 400 | 2.78 | 35.0 | 27.5 | 0.070 | 51.8 | 17966 |
| regional_folklore | fc_raw | context_overflow | -- | -- | -- | -- | -- | -- |
| regional_folklore | wiki_light | 400 | 2.48 | 26.5 | 33.8 | 0.074 | 50.0 | 19322 |
| regional_folklore | wiki_moderate | 400 | 2.25 | 22.5 | 43.2 | 0.061 | 56.7 | 13359 |
| regional_folklore | wiki_aggressive | 400 | 2.70 | 31.2 | 26.5 | 0.075 | 49.6 | 19446 |
| regional_folklore | wicer_iter1 | 400 | 2.69 | 31.8 | 26.2 | 0.075 | 48.9 | 19973 |
| regional_folklore | random_blind | 400 | 2.80 | 33.0 | 21.0 | 0.077 | 48.6 | 19897 |
| regional_folklore | random_pin | 400 | 2.46 | 25.8 | 33.2 | 0.066 | 54.5 | 15608 |
| small_medium_ent | fc_raw | 400 | 3.64 | 61.0 | 11.2 | 0.237 | 19.7 | 90575 |
| small_medium_ent | wiki_light | 400 | 2.88 | 38.0 | 26.5 | 0.069 | 50.1 | 19371 |
| small_medium_ent | wiki_moderate | 400 | 3.01 | 41.8 | 23.8 | 0.070 | 49.1 | 19064 |
| small_medium_ent | wiki_aggressive | 400 | 2.67 | 33.5 | 32.0 | 0.060 | 56.1 | 13929 |
| small_medium_ent | wicer_iter1 | 400 | 3.02 | 40.5 | 18.5 | 0.075 | 46.3 | 22173 |
| small_medium_ent | random_blind | 400 | 2.98 | 39.5 | 21.8 | 0.069 | 49.0 | 19061 |
| small_medium_ent | random_pin | 400 | 2.79 | 35.2 | 29.5 | 0.066 | 50.1 | 17301 |


## Compilation: compression and quality (paper Table 3)

| Level | Target | Actual | Quality | Score-1 % | vs. FC raw | Quality (answerable) | vs. FC raw (answerable) |
|---|---|---|---|---|---|---|---|
| FC raw (80 docs) | -- | -- | 3.62 | 12.2% | -- | 4.01 | -- |
| Wiki-light | 75% | 14.6% | 2.80 | 26.7% | -0.82 | 2.99 | -1.02 |
| Wiki-moderate | 50% | 14.6% | 2.80 | 26.2% | -0.82 | 2.97 | -1.04 |
| Wiki-aggressive | 25% | 14.1% | 2.79 | 26.2% | -0.83 | 2.98 | -1.03 |


> The answerable columns leave out RepLiQA's unanswerable questions (gold: "The answer is not found in the document."), about a fifth of each topic. A compiled wiki makes the reader abstain more, which those questions reward, so they shrink the gap the other columns measure.


## WiCER across topics, probe set (paper Table 4)

| Topic | Blind (mean / score-1) | Best WiCER | FC raw (probes) | Recovery | Iter | Stop |
|---|---|---|---|---|---|---|
| company_policies | 2.83 / 15.0% | 3.45 / 7.5% | 3.41 | 106% | 2 | insufficient_improvement |
| cybersecurity_news | 3.10 / 10.0% | 3.30 / 11.2% | 3.74 | 31% | 1 | insufficient_improvement |
| incident_report | 2.60 / 26.2% | 3.04 / 25.0% | 3.33 | 60% | 2 | insufficient_improvement |
| local_arts_and_culture | 2.67 / 30.0% | 2.88 / 25.0% | 3.83 | 17% | 1 | insufficient_improvement |
| local_economy | 2.62 / 28.8% | 3.26 / 15.0% | 3.92 | 49% | 1 | insufficient_improvement |
| local_education | 2.91 / 26.2% | 3.35 / 12.5% | 3.65 | 59% | 1 | insufficient_improvement |
| local_env_issues | 2.69 / 27.5% | 3.52 / 6.2% | 3.73 | 81% | 2 | max_iterations |
| local_health | 2.98 / 21.2% | 3.27 / 15.0% | 3.67 | 43% | 1 | insufficient_improvement |
| local_news | 2.91 / 26.2% | 3.00 / 22.5% | 3.65 | 12% | 1 | insufficient_improvement |
| local_politics | 2.96 / 25.0% | 2.99 / 25.0% | 3.75 | 3% | 1 | insufficient_improvement |
| local_sports | 3.01 / 26.2% | 3.30 / 13.8% | 3.90 | 32% | 1 | insufficient_improvement |
| local_tech | 2.89 / 23.8% | 3.40 / 11.2% | 3.46 | 89% | 2 | insufficient_improvement |
| neighborhood | 2.79 / 23.8% | 2.94 / 20.0% | 3.85 | 14% | 1 | insufficient_improvement |
| news_stories | 2.73 / 23.8% | 3.16 / 17.5% | 3.84 | 39% | 1 | insufficient_improvement |
| regional_cuisine | 2.95 / 27.5% | 3.04 / 26.2% | -- | -- | 1 | insufficient_improvement |
| regional_folklore | 2.38 / 38.8% | 2.96 / 25.0% | -- | -- | 1 | insufficient_improvement |
| small_medium_ent | 3.10 / 21.2% | 3.14 / 15.0% | 3.74 | 6% | 1 | insufficient_improvement |
| **Mean** | 2.83 | 3.18 / 17.3% | 3.70 | 43% | -- | -- |


> FC raw is scored on the same probes as Blind and Best WiCER, so recovery compares one question set throughout. The paper divides by FC over every question, which agrees with this only when the probes are as hard as the full set.


### Iteration trajectory

| Topic | Iter | Wiki | Probe mean | Score-1 % | Failures | Pinned facts | Pinned words | Kept verbatim | Actual ratio | Improvement |
|---|---|---|---|---|---|---|---|---|---|---|
| company_policies | 0 | blind_moderate | 2.83 | 15.0 | 12 | 0 | 0 | -- | 16.5% | -- |
| company_policies | 1 | wicer_iter1 | 3.20 | 10.0 | 8 | 12 | 1105 | 8% | 16.7% | +13.3% |
| company_policies | 2 | wicer_iter2 | 3.45 | 7.5 | 6 | 20 | 1739 | 0% | 18.2% | +7.8% |
| cybersecurity_news | 0 | blind_moderate | 3.10 | 10.0 | 8 | 0 | 0 | -- | 20.9% | -- |
| cybersecurity_news | 1 | wicer_iter1 | 3.30 | 11.2 | 9 | 8 | 748 | 0% | 14.9% | +6.5% |
| incident_report | 0 | blind_moderate | 2.60 | 26.2 | 21 | 0 | 0 | -- | 12.5% | -- |
| incident_report | 1 | wicer_iter1 | 2.89 | 22.5 | 18 | 21 | 1827 | 0% | 20.6% | +11.1% |
| incident_report | 2 | wicer_iter2 | 3.04 | 25.0 | 20 | 39 | 3449 | 0% | 18.4% | +5.2% |
| local_arts_and_culture | 0 | blind_moderate | 2.67 | 30.0 | 24 | 0 | 0 | -- | 13.1% | -- |
| local_arts_and_culture | 1 | wicer_iter1 | 2.88 | 25.0 | 20 | 24 | 2032 | 25% | 13.1% | +7.5% |
| local_economy | 0 | blind_moderate | 2.62 | 28.8 | 23 | 0 | 0 | -- | 17.3% | -- |
| local_economy | 1 | wicer_iter1 | 3.26 | 15.0 | 12 | 23 | 1887 | 0% | 15.8% | +24.3% |
| local_economy | 2 | wicer_iter2 | 3.23 | 13.8 | 11 | 35 | 2891 | 0% | 14.4% | -1.1% |
| local_education | 0 | blind_moderate | 2.91 | 26.2 | 21 | 0 | 0 | -- | 14.6% | -- |
| local_education | 1 | wicer_iter1 | 3.35 | 12.5 | 10 | 21 | 1755 | 10% | 13.5% | +15.0% |
| local_education | 2 | wicer_iter2 | 3.19 | 21.2 | 17 | 31 | 2571 | 0% | 14.7% | -4.9% |
| local_env_issues | 0 | blind_moderate | 2.69 | 27.5 | 22 | 0 | 0 | -- | 11.2% | -- |
| local_env_issues | 1 | wicer_iter1 | 3.02 | 16.2 | 13 | 22 | 1809 | 5% | 13.1% | +12.6% |
| local_env_issues | 2 | wicer_iter2 | 3.52 | 6.2 | 5 | 35 | 2883 | 3% | 16.2% | +16.5% |
| local_env_issues | 3 | wicer_iter3 | 3.05 | 20.0 | 16 | 40 | 3286 | 0% | 12.7% | -13.5% |
| local_health | 0 | blind_moderate | 2.98 | 21.2 | 17 | 0 | 0 | -- | 15.3% | -- |
| local_health | 1 | wicer_iter1 | 3.27 | 15.0 | 12 | 17 | 1398 | 0% | 16.6% | +10.1% |
| local_health | 2 | wicer_iter2 | 3.01 | 20.0 | 16 | 29 | 2488 | 3% | 12.7% | -8.0% |
| local_news | 0 | blind_moderate | 2.91 | 26.2 | 21 | 0 | 0 | -- | 14.1% | -- |
| local_news | 1 | wicer_iter1 | 3.00 | 22.5 | 18 | 21 | 1723 | 52% | 13.1% | +3.0% |
| local_politics | 0 | blind_moderate | 2.96 | 25.0 | 20 | 0 | 0 | -- | 16.0% | -- |
| local_politics | 1 | wicer_iter1 | 2.99 | 25.0 | 20 | 20 | 1793 | 0% | 11.8% | +0.8% |
| local_sports | 0 | blind_moderate | 3.01 | 26.2 | 21 | 0 | 0 | -- | 12.8% | -- |
| local_sports | 1 | wicer_iter1 | 3.30 | 13.8 | 11 | 21 | 1872 | 52% | 15.3% | +9.5% |
| local_tech | 0 | blind_moderate | 2.89 | 23.8 | 19 | 0 | 0 | -- | 13.3% | -- |
| local_tech | 1 | wicer_iter1 | 3.21 | 16.2 | 13 | 19 | 1606 | 0% | 15.5% | +11.3% |
| local_tech | 2 | wicer_iter2 | 3.40 | 11.2 | 9 | 32 | 2715 | 0% | 15.2% | +5.8% |
| neighborhood | 0 | blind_moderate | 2.79 | 23.8 | 19 | 0 | 0 | -- | 15.7% | -- |
| neighborhood | 1 | wicer_iter1 | 2.94 | 20.0 | 16 | 19 | 1615 | 0% | 14.9% | +5.4% |
| news_stories | 0 | blind_moderate | 2.73 | 23.8 | 19 | 0 | 0 | -- | 15.9% | -- |
| news_stories | 1 | wicer_iter1 | 3.16 | 17.5 | 14 | 19 | 1649 | 0% | 13.2% | +16.1% |
| news_stories | 2 | wicer_iter2 | 2.95 | 20.0 | 16 | 33 | 2888 | 0% | 12.0% | -6.7% |
| regional_cuisine | 0 | blind_moderate | 2.95 | 27.5 | 22 | 0 | 0 | -- | 13.7% | -- |
| regional_cuisine | 1 | wicer_iter1 | 3.04 | 26.2 | 21 | 22 | 1761 | 82% | 13.5% | +3.0% |
| regional_folklore | 0 | blind_moderate | 2.38 | 38.8 | 31 | 0 | 0 | -- | 9.5% | -- |
| regional_folklore | 1 | wicer_iter1 | 2.96 | 25.0 | 20 | 31 | 2655 | 0% | 15.7% | +24.7% |
| regional_folklore | 2 | wicer_iter2 | 2.71 | 23.8 | 19 | 51 | 4512 | 0% | 11.9% | -8.4% |
| small_medium_ent | 0 | blind_moderate | 3.10 | 21.2 | 17 | 0 | 0 | -- | 16.0% | -- |
| small_medium_ent | 1 | wicer_iter1 | 3.14 | 15.0 | 12 | 17 | 1516 | 47% | 18.1% | +1.2% |


## WiCER held-out evaluation (questions the loop never saw)

| Topic | Questions | n | Blind (mean / score-1) | Best pinned WiCER | FC raw | Recovery | Iter |
|---|---|---|---|---|---|---|---|
| company_policies | answerable | 245 | 3.07 / 11.0% | 3.20 / 11.4% | 3.94 | 15% | 2 |
| company_policies | unanswerable | 75 | 2.59 / 32.0% | 2.93 / 21.3% | 2.60 | -- | 2 |
| cybersecurity_news | answerable | 246 | 3.29 / 15.4% | 3.12 / 17.1% | 4.15 | -19% | 1 |
| cybersecurity_news | unanswerable | 74 | 2.09 / 39.2% | 2.35 / 33.8% | 2.18 | -- | 1 |
| incident_report | answerable | 251 | 2.74 / 28.7% | 3.15 / 18.7% | 3.84 | 37% | 2 |
| incident_report | unanswerable | 69 | 1.84 / 53.6% | 1.59 / 65.2% | 1.88 | -- | 2 |
| local_arts_and_culture | answerable | 248 | 3.00 / 18.1% | 2.42 / 39.9% | 3.98 | -59% | 1 |
| local_arts_and_culture | unanswerable | 72 | 1.67 / 62.5% | 2.15 / 31.9% | 1.81 | -- | 1 |
| local_economy | answerable | 247 | 2.90 / 23.9% | 2.85 / 24.3% | 4.00 | -5% | 1 |
| local_economy | unanswerable | 73 | 2.37 / 31.5% | 2.29 / 38.4% | 2.10 | -- | 1 |
| local_education | answerable | 243 | 2.97 / 19.8% | 2.73 / 26.7% | 3.91 | -26% | 1 |
| local_education | unanswerable | 77 | 2.08 / 46.8% | 2.84 / 22.1% | 2.06 | -- | 1 |
| local_env_issues | answerable | 245 | 2.67 / 29.8% | 2.91 / 22.9% | 3.91 | 19% | 2 |
| local_env_issues | unanswerable | 75 | 2.31 / 34.7% | 2.67 / 33.3% | 2.31 | -- | 2 |
| local_health | answerable | 244 | 2.85 / 26.2% | 3.20 / 14.8% | 4.01 | 30% | 1 |
| local_health | unanswerable | 76 | 2.08 / 43.4% | 2.17 / 44.7% | 2.26 | -- | 1 |
| local_news | answerable | 264 | 2.92 / 22.0% | 2.55 / 32.6% | 3.91 | -38% | 1 |
| local_news | unanswerable | 56 | 2.36 / 42.9% | 2.95 / 17.9% | 2.09 | -- | 1 |
| local_politics | answerable | 248 | 3.13 / 18.1% | 2.47 / 35.5% | 3.98 | -78% | 1 |
| local_politics | unanswerable | 72 | 1.89 / 50.0% | 2.72 / 22.2% | 2.21 | -- | 1 |
| local_sports | answerable | 252 | 3.16 / 20.2% | 2.88 / 26.2% | 4.08 | -31% | 1 |
| local_sports | unanswerable | 68 | 2.59 / 36.8% | 3.60 / 4.4% | 2.53 | -- | 1 |
| local_tech | answerable | 246 | 3.20 / 17.5% | 3.43 / 13.4% | 4.21 | 23% | 2 |
| local_tech | unanswerable | 74 | 2.69 / 27.0% | 2.54 / 36.5% | 2.73 | -- | 2 |
| neighborhood | answerable | 248 | 3.15 / 15.7% | 3.14 / 16.1% | 3.98 | -1% | 1 |
| neighborhood | unanswerable | 72 | 1.85 / 59.7% | 1.82 / 58.3% | 2.12 | -- | 1 |
| news_stories | answerable | 266 | 2.90 / 22.6% | 2.88 / 25.2% | 4.04 | -1% | 1 |
| news_stories | unanswerable | 54 | 2.24 / 42.6% | 2.43 / 31.5% | 1.98 | -- | 1 |
| regional_cuisine | answerable | 257 | 2.98 / 23.3% | 2.35 / 40.5% | -- | -- | 1 |
| regional_cuisine | unanswerable | 63 | 2.19 / 39.7% | 2.76 / 19.0% | -- | -- | 1 |
| regional_folklore | answerable | 258 | 2.24 / 44.2% | 2.84 / 19.4% | -- | -- | 1 |
| regional_folklore | unanswerable | 62 | 2.05 / 46.8% | 1.73 / 54.8% | -- | -- | 1 |
| small_medium_ent | answerable | 255 | 3.24 / 18.8% | 3.05 / 19.6% | 4.01 | -26% | 1 |
| small_medium_ent | unanswerable | 65 | 1.98 / 46.2% | 2.74 / 16.9% | 2.06 | -- | 1 |


> Every probe is left out, so these are questions the loop never saw. RepLiQA's unanswerable questions (gold: "The answer is not found in the document.") are scored apart and given no recovery: they reward a reader that abstains, which a smaller context encourages, so a wiki can beat FC raw on them without having kept anything. With one compilation per arm, a held-out difference smaller than the gap between two compilations of the same prompt cannot be told apart from compiler noise.


## Ablation: diagnosed vs. random pinning, probe set (paper Table 5)

| Topic | Blind (WiCER arm) | WiCER (best-of) | WiCER (pinned only) | Blind (abl. arm) | Random pin (best-of) | Random pin (pinned only) |
|---|---|---|---|---|---|---|
| company_policies | 2.83 | 3.45 (+0.62) | 3.45 (+0.62) | 2.90 | 2.90 (+0.00) | 2.85 (-0.05) |
| cybersecurity_news | 3.10 | 3.30 (+0.20) | 3.30 (+0.20) | 2.89 | 2.89 (+0.00) | 2.86 (-0.03) |
| incident_report | 2.60 | 3.04 (+0.44) | 3.04 (+0.44) | 2.86 | 3.04 (+0.18) | 3.04 (+0.18) |
| local_arts_and_culture | 2.67 | 2.88 (+0.20) | 2.88 (+0.20) | 2.67 | 2.67 (+0.00) | 2.50 (-0.17) |
| local_economy | 2.62 | 3.26 (+0.64) | 3.26 (+0.64) | 3.31 | 3.31 (+0.00) | 2.80 (-0.51) |
| local_education | 2.91 | 3.35 (+0.44) | 3.35 (+0.44) | 3.06 | 3.06 (+0.00) | 2.70 (-0.36) |
| local_env_issues | 2.69 | 3.52 (+0.84) | 3.52 (+0.84) | 3.05 | 3.05 (+0.00) | 2.59 (-0.46) |
| local_health | 2.98 | 3.27 (+0.30) | 3.27 (+0.30) | 2.86 | 2.86 (+0.00) | 2.55 (-0.31) |
| local_news | 2.91 | 3.00 (+0.09) | 3.00 (+0.09) | 2.84 | 3.00 (+0.16) | 3.00 (+0.16) |
| local_politics | 2.96 | 2.99 (+0.02) | 2.99 (+0.02) | 2.94 | 2.94 (+0.00) | 2.54 (-0.40) |
| local_sports | 3.01 | 3.30 (+0.29) | 3.30 (+0.29) | 3.09 | 3.09 (+0.00) | 2.67 (-0.41) |
| local_tech | 2.89 | 3.40 (+0.51) | 3.40 (+0.51) | 2.84 | 2.84 (+0.00) | 2.80 (-0.04) |
| neighborhood | 2.79 | 2.94 (+0.15) | 2.94 (+0.15) | 2.77 | 2.77 (+0.00) | 2.42 (-0.35) |
| news_stories | 2.73 | 3.16 (+0.44) | 3.16 (+0.44) | 2.51 | 2.71 (+0.20) | 2.71 (+0.20) |
| regional_cuisine | 2.95 | 3.04 (+0.09) | 3.04 (+0.09) | 2.96 | 2.96 (+0.00) | 2.85 (-0.11) |
| regional_folklore | 2.38 | 2.96 (+0.59) | 2.96 (+0.59) | 2.95 | 2.95 (+0.00) | 2.71 (-0.24) |
| small_medium_ent | 3.10 | 3.14 (+0.04) | 3.14 (+0.04) | 2.92 | 2.92 (+0.00) | 2.88 (-0.05) |
| **Mean delta vs. own blind** | -- | +0.35 | +0.35 | -- | +0.03 | -0.17 |


> This is the load-bearing comparison (paper Section 7.5): it is the only thing separating 'diagnosis works' from 'pinning anything works'. The paper measures +0.16 for random pinning against +0.95 for WiCER.


> Each arm compiles its own blind wiki -- paper Section 7.5, 'each topic uses an independent blind compilation; blind baselines differ slightly from Table 4 due to LLM compiler non-determinism' -- so every delta here is measured against the blind in its own column, never the other arm's. 'Best-of' follows the paper's 'best iteration per topic' and so includes iteration 0, the unpinned blind: an arm whose pinning only ever hurt therefore reports its own baseline and a delta of exactly +0.00. That is not a null effect, it is a selection artifact, and it is visible in the paper too -- seven of its seventeen topics print Random Pin identical to Blind. The pinned-only columns drop that selection and report what pinning actually did, for both arms alike -- an ablation that measured its two arms differently is what produced the incoherent row this replaced.


> One construction difference from the paper: its Table 5 compares the Table 4 WiCER score against the ablation arm's independent blind, mixing two compilations in one delta. Here each arm is measured against its own blind, so our WiCER delta and the paper's +0.95 do not share a denominator even when the method agrees.


## Latency by condition (paper Table 13)

| Condition | Context tokens | TTFT median (s) | tok/s |
|---|---|---|---|
| fc_raw | 90,765 | 0.238 | 19.8 |
| wiki_light | 18,295 | 0.069 | 52.2 |
| wiki_moderate | 17,514 | 0.068 | 51.0 |
| wiki_aggressive | 16,592 | 0.065 | 52.6 |
| wicer_iter1 | 17,722 | 0.068 | 51.4 |
| wicer_iter2 | 18,414 | 0.069 | 50.4 |
| random_blind | 16,976 | 0.066 | 51.8 |
| random_pin | 16,120 | 0.065 | 52.8 |


## Success criteria (spec.md Section 8)

| Criterion | Target | Measured | Delta | Verdict | Coverage |
|---|---|---|---|---|---|
| FC raw mean | 3.46 | 3.62 | +0.16 | PASS | n=15 topics |
| RAG mean | 3.63 | -- | -- | NOT RUN | -- |
| Wiki-moderate (blind) mean | 2.25 | 2.80 | +0.55 | MISS | n=17 topics |
| Wiki-moderate score-1 rate % | 57.10 | 26.25 | -30.85 | MISS | n=17 topics |
| Best WiCER mean | 3.18 | 3.18 | -0.00 | PASS | n=17 topics |
| Best WiCER score-1 rate % | 24.80 | 17.28 | -7.52 | PASS | n=17 topics |


> These targets describe the full 17-topic RepLiQA sweep. A partial or reduced-scale run will miss them by construction -- read the verdict column only when the coverage column says 17 topics.


> Directional agreement is the real bar (spec.md Section 8): the FC/RAG crossover exists, and targeted diagnosis beats random pinning by a wide margin. Exact numbers will move -- the compiler is stochastic and TTFT changes outright on different hardware.
