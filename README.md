# ERA5 Weather Codec

Сжатие 28-канального состояния нижней атмосферы с ограниченным количеством
обучающих данных и вычислений.

Проект подготовлен для хакатона МИФИ по исследованию data efficiency
автокодировщиков на глобальных полях ERA5. Модель совместно кодирует
приземные и высотные переменные, формирует настоящий сериализованный
bitstream и оценивается в режимах сжатия 32x и 64x.

## Цель проекта

Основной исследовательский вопрос:

> Какое минимальное число уникальных глобальных погодных кадров необходимо,
> чтобы на одной GPU обучить компактное 28-канальное представление, не
> уступающее VAEformer по заранее зафиксированным статистическим критериям?

В качестве размера обучающей выборки используется число уникальных
шестичасовых состояний:

```text
N in {128, 256, 512, 1024, 2048, 4096, 8192}
```

Каждый следующий набор должен содержать все кадры предыдущего. Количество
случайно извлечённых patches не считается размером датасета.

## Данные

Рекомендуемый источник - ERA5 из
[WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html):

```text
gs://weatherbench2/datasets/era5/
1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr
```

Используются:

- 2014-2019 - обучающий пул;
- 2020 - validation и выбор checkpoint/рабочей точки;
- 2021 - финальный test.

Статистики нормализации и диапазоны PSNR вычисляются без использования
тестового года.

### Порядок 28 каналов

```text
t2m, mslp, u10, v10, tp6h, sst, tcwv, tcc,
T1000, T925, T850, T700,
U1000, U925, U850, U700,
V1000, V925, V850, V700,
Z1000, Z925, Z850, Z700,
Q1000, Q925, Q850, Q700
```

Восемь приземных каналов:

| Код | Переменная | Единицы |
|---|---|---:|
| `t2m` | температура на высоте 2 м | K |
| `mslp` | давление на среднем уровне моря | Pa |
| `u10` | зональный ветер на высоте 10 м | m/s |
| `v10` | меридиональный ветер на высоте 10 м | m/s |
| `tp6h` | осадки за предыдущие 6 часов | m |
| `sst` | температура поверхности моря | K |
| `tcwv` | водяной пар в атмосферном столбе | kg/m2 |
| `tcc` | общая облачность | 0-1 |

Остальные 20 каналов образованы переменными `T`, `U`, `V`, `Z`, `Q` на
уровнях 1000, 925, 850 и 700 hPa.

SST оценивается только по океанской маске. В качестве известных статических
признаков используются land-sea mask, орография и координатные поля.
Статические признаки не входят в числитель степени сжатия.

Текущий notebook реализует эксперимент на cell-centred сетке 0.5 градуса
размером `360 x 720`. Эксперимент 0.25 градуса должен выполняться отдельным
запуском на исходной сетке WeatherBench 2.

## Решение

Используется одна joint-модель для всех 28 погодных каналов:

```text
weather + static fields
        |
geo-aware encoder
        |
quantized latent symbols
        |
entropy coding -> .era5c bitstream
        |
geo-aware decoder
        |
28 reconstructed fields
```

Основные особенности модели:

- периодическое дополнение по долготе и отражающее дополнение по широте;
- три уровня пространственного сжатия;
- `pixel_unshuffle` в encoder для сохранения информации перед уменьшением
  разрешения;
- gated residual blocks с depthwise-свёртками 7x7 и channel attention;
- слабое ограниченное FiLM-conditioning по статическим полям;
- `pixel_shuffle` в decoder вместо фиксированной билинейной интерполяции;
- единый квантованный bottleneck для surface и pressure полей;
- straight-through estimator при обучении;
- отдельный gain для рабочих точек 32x и 64x;
- сериализация целочисленных символов и entropy coding через DEFLATE.

Архитектура не использует skip connection из входного погодного состояния в
decoder: восстановление должно выполняться из переданного латента и известных
статических полей.

## Фактическая степень сжатия

Степень сжатия определяется по размеру полного файла, а не по размерности
тензора bottleneck:

```text
CR = (32 * T * C * H * W) / (8 * B)
```

где:

- `T` - число кадров;
- `C = 28`;
- `H`, `W` - размеры сетки;
- `B` - полный размер bitstream в байтах с заголовком и side information.

На validation адаптивно выбирается максимальный gain, при котором минимальный
измеренный compression ratio удовлетворяет ограничению с safety margin:

```text
32x: CR_min >= 32 * 1.035
64x: CR_min >= 64 * 1.035
```

После выбора gain финальный test выполняется отдельно. Для каждого bitstream
проверяется точное совпадение квантованных символов до и после entropy
encode/decode.

## Функция потерь

Базовый reconstruction loss представляет собой latitude-weighted MSE с
равным вкладом surface и pressure групп:

```text
L_reconstruction = 0.5 * L_surface + 0.5 * L_pressure
```

Дополнительно используются:

- gradient loss для сохранения пространственных структур;
- усиленный штраф для ошибок осадков;
- мягкий occupancy floor, препятствующий полному схлопыванию латента.

Фактический bitrate управляется gain и измеряется по настоящему bitstream.

## Метрики

Для каждого канала вычисляется latitude-weighted RMSE:

```text
RMSE_f = sqrt(sum(cos(latitude) * (prediction - target)^2) /
              sum(cos(latitude)))

NRMSE_f = RMSE_f / train_std_f
```

Групповые показатели:

```text
S_surface  = mean(NRMSE по 8 приземным каналам)
S_pressure = mean(NRMSE по 20 высотным каналам)
S_all      = 0.5 * S_surface + 0.5 * S_pressure
```

Также публикуются:

- физический RMSE по каждому каналу;
- PSNR с диапазоном по 0.5-99.5 перцентилям train;
- сферический spectral error;
- метрики экстремальных осадков;
- фактический compression ratio;
- размер bitstream;
- результат exact latent roundtrip.

## Non-inferiority относительно VAEformer

Для каждого тестового времени сравнивается NRMSE нашей модели и референса:

```text
delta = 100 * (NRMSE_model / NRMSE_reference - 1)
```

95% доверительный интервал оценивается парным block bootstrap:

- блок - 7 суток;
- 2000 повторов;
- model и reference сравниваются на одних временных кадрах.

Критерии допуска:

| Критерий | 32x | 64x |
|---|---:|---:|
| Верхняя граница 95% CI общего NRMSE | <= +3% | <= +7% |
| Верхняя граница 95% CI surface/pressure | <= +5% | <= +8% |
| Верхняя граница отдельного критического поля | <= +7% | <= +10% |
| Нижняя граница разности PSNR | >= -0.25 dB | >= -0.5 dB |
| Ухудшение spectral error | <= 5% | <= 10% |
| Фактический CR и exact roundtrip | обязательны | обязательны |

Публичные агрегированные числа CRA5 недостаточны для этого теста. Для
официального `PASS/FAIL` необходимы совместимые per-frame reconstruction
outputs VAEformer на том же наборе времён. Если reference отсутствует,
notebook сохраняет собственные reconstruction scores, но выставляет
`NOT_EVALUABLE`, а не заявляет прохождение non-inferiority.

## Проверка полезности латента

После обучения codec полностью замораживается. Небольшая latent-модель:

- получает латент текущего состояния;
- предсказывает латент через 6 часов;
- содержит не более 2 млн параметров;
- обучается ровно на 1024 уникальных парах;
- выполняет не более 5000 optimizer steps.

Предсказанный латент декодируется обратно в 28 погодных полей. Итоговый
NRMSE сравнивается с persistence baseline.

## Ограничения

Решение соблюдает ограничения технического задания:

- одна GPU;
- не более 24 GiB peak VRAM;
- менее 20 млн обучаемых параметров;
- не более 50 000 optimizer steps;
- не более 48 GPU-часов;
- отсутствие внешнего погодного pretraining;
- одна совместная модель для всех 28 каналов.

## Запуск на Kaggle

1. Создайте Kaggle Notebook.
2. В `Settings -> Accelerator` выберите GPU P100 или T4.
3. Через `Add Input` подключите подготовленный набор данных.
4. Загрузите `nuclear-it-hackaton-solution.ipynb`.
5. Выберите режим запуска:

```python
run_mode = "AUTO"
```

Доступные режимы:

| Режим | Поведение |
|---|---|
| `AUTO` | загрузить совместимый checkpoint или обучить модель с нуля |
| `TRAIN` | всегда обучать с нуля |
| `INFERENCE` | требовать готовый checkpoint и выполнить только оценку |

6. Запустите `Run All`.

Для чистого обучения рекомендуется новая Kaggle-сессия без outputs старой
архитектуры.

### Ожидаемые входные файлы

```text
train.npy
val.npy
test.npy
train_pairs.npy
val_pairs.npy
static.npy
stats.npz
manifest.json
```

Для официального latent probe при необходимости подключаются:

```text
probe_train.npy
probe_val.npy
probe_test.npy
probe_train_pairs.npy
probe_val_pairs.npy
probe_test_pairs.npy
```

Для повторной оценки без обучения можно подключить:

```text
weather_codec_best.pt
```

или:

```text
weather_codec_final.pt
```

## Результаты запуска

Рабочая директория Kaggle:

```text
/kaggle/working/era5_tz_codec/
```

Главный архив для сдачи:

```text
/kaggle/working/era5_tz_codec/era5_tz_submission.zip
```

Основные артефакты:

```text
outputs/
├── weather_codec_best.pt
├── selected_rates.json
├── quality_bitrate_curve.csv
├── final_metrics.json
├── tz_test_score.json
├── tz_test_score.csv
├── noninferiority_vs_vaeformer.json
├── compliance.json
├── resources.json
├── codec_32x/
├── codec_64x/
└── probe/
```

Главная reconstruction-метрика - `primary_test_score_s_all`, записанная
отдельно для 32x и 64x в:

```text
outputs/tz_test_score.json
outputs/tz_test_score.csv
```

Чем меньше `S_all`, тем лучше восстановление.

## Воспроизводимость

Notebook сохраняет:

- seed и полную конфигурацию;
- точный порядок каналов и единицы;
- timestamps каждого split;
- число уникальных обучающих кадров;
- optimizer steps и число просмотренных примеров;
- checkpoint encoder/decoder;
- выбранные gain для 32x и 64x;
- bitstreams и их размеры;
- JSON/CSV с метриками;
- peak VRAM, время выполнения и GPU-hours;
- compliance report по ограничениям хакатона.

## Структура репозитория

```text
.
├── README.md
├── nuclear-it-hackaton-solution.ipynb
└── outputs/                         # результаты запуска, если публикуются
```

Большие `.npy`, checkpoints и bitstreams рекомендуется хранить в Kaggle
Datasets, Git LFS или объектном хранилище, а не в обычной истории Git.

## Ссылки

- [ERA5](https://doi.org/10.1002/qj.3803)
- [WeatherBench 2](https://doi.org/10.1029/2023MS004019)
- [WeatherBench 2 data guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
- [CRA5 / VAEformer](https://github.com/taohan10200/CRA5)
- [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436)

