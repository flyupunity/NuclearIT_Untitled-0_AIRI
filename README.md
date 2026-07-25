# ERA5 Weather Codec

Исследование сжатия 28-канального состояния нижней атмосферы ERA5 при
ограниченном количестве обучающих данных и вычислений.

Проект подготовлен для хакатона МИФИ. Основное решение обучает совместный
погодный codec с квантованным bottleneck, формирует настоящий bitstream,
проверяет точный roundtrip квантованных символов и оценивает две рабочие
точки: 32x и 64x.

В репозитории также представлена экспериментальная ветка transfer learning:
трёхканальный RGB hyperprior из CompressAI адаптируется к 28 погодным каналам
и дообучается на небольшой выборке ERA5.

## Ключевые результаты

Основной результат получен на сетке 0.5 градуса при `N=256` уникальных
обучающих кадров. Тестовая выборка текущего запуска содержит 16 кадров 2021
года.

| Метрика | 32x | 64x |
|---|---:|---:|
| Средний фактический CR | 33.167x | 66.332x |
| Минимальный CR по кадрам | 32.966x | 65.578x |
| Общий score, `S_all` | 0.160838 | 0.163668 |
| Surface score | 0.189091 | 0.191895 |
| Pressure score | 0.132586 | 0.135441 |
| Средний PSNR | 31.820 dB | 31.635 dB |
| Spectral error | 0.036391 | 0.036764 |
| Exact latent roundtrip | да | да |
| Codec gate | PASS | PASS |

Итоговая метрика:

```text
S_all = 0.5 * S_surface + 0.5 * S_pressure
```

Меньшее значение соответствует более точному восстановлению.

Официальный verdict относительно VAEformer для этого запуска:

```text
NOT_EVALUABLE
```

Причина не связана с качеством собственного codec: к запуску не были
приложены совместимые per-frame reconstruction outputs VAEformer, а 16
тестовых timestamps не образуют непрерывный семидневный блок для официального
paired block bootstrap.

## Постановка задачи

Цель исследования - определить минимальное число уникальных глобальных
шестичасовых кадров, достаточное для обучения полезного компактного
представления на одной GPU:

```text
N in {128, 256, 512, 1024, 2048, 4096, 8192}
```

Модель должна:

- совместно восстанавливать все 28 погодных каналов;
- обеспечивать фактическое сжатие не менее 32x и 64x;
- формировать сериализованный bitstream;
- точно восстанавливать квантованные символы после entropy decode;
- сохранять пространственные структуры и экстремальные осадки;
- давать латент, полезный для прогноза состояния на +6 часов;
- укладываться в ограничения одной GPU.

## Данные

Источник - ERA5 из
[WeatherBench 2](https://weatherbench2.readthedocs.io/en/latest/data-guide.html):

```text
gs://weatherbench2/datasets/era5/
1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr
```

Временные разбиения:

- 2014-2019 - обучающий пул;
- 2020 - validation и выбор checkpoint/рабочей точки;
- 2021 - финальный test.

Текущая реализация работает с cell-centred сеткой 0.5 градуса:

```text
28 x 360 x 720
```

Эксперимент 0.25 градуса должен выполняться отдельным запуском на
подготовленной полноразмерной сетке.

### Порядок каналов

```text
t2m, mslp, u10, v10, tp6h, sst, tcwv, tcc,
T1000, T925, T850, T700,
U1000, U925, U850, U700,
V1000, V925, V850, V700,
Z1000, Z925, Z850, Z700,
Q1000, Q925, Q850, Q700
```

Восемь приземных каналов:

| Код | Физический смысл | Единицы |
|---|---|---:|
| `t2m` | температура на высоте 2 м | K |
| `mslp` | давление на среднем уровне моря | Pa |
| `u10` | зональная компонента ветра на высоте 10 м | m/s |
| `v10` | меридиональная компонента ветра на высоте 10 м | m/s |
| `tp6h` | осадки за предыдущие 6 часов | m |
| `sst` | температура поверхности моря | K |
| `tcwv` | водяной пар в атмосферном столбе | kg/m2 |
| `tcc` | общая облачность | 0-1 |

Высотная часть содержит `T`, `U`, `V`, `Z`, `Q` на уровнях 1000, 925, 850
и 700 hPa. SST оценивается только по океанской маске.

Статические поля включают land-sea mask, орографию и координатные признаки.
Они известны encoder и decoder, не восстанавливаются и не учитываются в
числителе compression ratio.

## Основное решение

Главный notebook:

```text
nuclear-it-hackaton-solution (1).ipynb
```

### Архитектура

Используется один convolutional weather codec для всех 28 каналов:

```text
28 weather channels + 4 static channels
                    |
              geo-aware encoder
                    |
          192-channel latent, H/8 x W/8
                    |
         gain scaling + integer quantization
                    |
    row-delta + zigzag + DEFLATE bitstream
                    |
              geo-aware decoder
                    |
          reconstructed 28-channel state
```

Основные компоненты:

- `GeoConv2d` с circular padding по долготе и reflection padding по широте;
- три encoder-ступени со stride 2;
- каналы encoder: `32 -> 112 -> 144 -> 192`;
- residual blocks с GroupNorm, depthwise 5x5 и pointwise expansion;
- Squeeze-and-Excitation в bottleneck;
- ограниченный поканальный масштаб латента;
- integer quantization со straight-through estimator при обучении;
- слабое статическое conditioning в decoder без погодного bypass;
- три decoder-ступени с bilinear upsampling;
- отдельный reconstruction head и residual refinement.

Параметры основного codec:

```text
3,389,222
```

### Квантизация и bitstream

Непрерывный латент переводится в символы:

```text
symbols = latent / channel_scale * gain
integer_symbols = round(symbols)
```

Далее применяются:

1. пространственное delta-кодирование;
2. zigzag-преобразование знаковых целых чисел;
3. сериализация в `uint32`;
4. DEFLATE сжатие;
5. заголовок с формой, gain и SHA256 квантованных символов.

После decode проверяется SHA256 и точное совпадение integer symbols.
Реконструкция погодных полей остаётся lossy, но entropy roundtrip латента
является точным.

### Реальная степень сжатия

Степень сжатия считается по полному сериализованному потоку:

```text
CR = (32 * T * C * H * W) / (8 * B)
```

где `B` включает payload и заголовок. Статические карты исключены из
числителя.

На validation выполняется адаптивный поиск максимального gain, который ещё
удовлетворяет ограничению с safety margin `1.035`.

Выбранные значения в сохранённом запуске:

| Точка | Gain | Validation CR min |
|---|---:|---:|
| 32x | 170.8232 | 33.1315x |
| 64x | 16.6023 | 66.2439x |

### Обучение

Конфигурация основного эксперимента:

| Параметр | Значение |
|---|---:|
| Уникальные train frames | 256 |
| Patch size | 160x160 |
| Batch size на P100 | 2 |
| Optimizer steps | 12 000 |
| Просмотренные примеры | 24 000 |
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Warmup | 600 шагов |
| Reconstruction warmup | 2000 шагов |
| Gradient clipping | 0.75 |

Loss объединяет:

- latitude-weighted reconstruction MSE;
- равный вклад surface и pressure групп;
- gradient loss для пространственных структур;
- дополнительный precipitation loss;
- soft occupancy floor против схлопывания латента.

В приложенном выполненном notebook был повторно использован checkpoint того
же эксперимента с 12 000 шагами. Это не внешнее погодное предобучение:
training был пропущен только для повторной calibration и evaluation.

### Ресурсы

| Показатель | Значение |
|---|---:|
| GPU | Tesla P100 16 GB |
| Peak allocated VRAM | 13.805 GiB |
| Параметры codec | 3.389 млн |
| Параметры probe | 194 496 |
| Максимальные шаги codec | 12 000 |
| Шаги probe | 5000 |

Зафиксированные `0.423 GPU-hours` относятся к повторному inference-запуску с
готовым checkpoint и не должны интерпретироваться как полное время обучения.

## Проверка латентного представления

После обучения encoder и decoder замораживаются. Небольшая residual
latent-модель предсказывает состояние bottleneck через 6 часов.

Протокол:

- 1024 уникальные пары `t -> t+6h` для train;
- 32 пары 2020 года для выбора checkpoint;
- 32 пары 2021 года для финальной оценки;
- 194 496 параметров;
- 5000 optimizer steps;
- codec полностью заморожен;
- используется gain рабочей точки 64x.

Результаты на test 2021:

| Модель | `S_all` |
|---|---:|
| Persistence | 0.344875 |
| Frozen-latent probe | 0.282396 |

Относительное улучшение относительно persistence:

```text
18.12%
```

Сравнение с аналогичным probe поверх VAEformer осталось
`NOT_EVALUABLE`, поскольку reference features не были приложены.

## Метрики

Для каждого канала используется latitude-weighted RMSE:

```text
RMSE_f = sqrt(sum(cos(latitude) * (prediction - target)^2) /
              sum(cos(latitude)))

NRMSE_f = RMSE_f / train_std_f
```

Итоговые группы:

```text
S_surface  = mean(NRMSE по 8 приземным каналам)
S_pressure = mean(NRMSE по 20 высотным каналам)
S_all      = 0.5 * S_surface + 0.5 * S_pressure
```

Дополнительно рассчитываются:

- физический RMSE;
- PSNR с train-only диапазонами 0.5-99.5 перцентиля;
- spherical spectral error;
- RMSE и F1 экстремальных осадков;
- фактический CR;
- размер каждого bitstream;
- exact latent roundtrip.

## Non-inferiority относительно VAEformer

Для каждого тестового времени:

```text
delta = 100 * (NRMSE_model / NRMSE_reference - 1)
```

95% доверительный интервал должен оцениваться парным block bootstrap:

- семидневные блоки по 28 шестичасовых кадров;
- 2000 повторов;
- одинаковые timestamps для model и reference.

Критерии допуска:

| Критерий | 32x | 64x |
|---|---:|---:|
| Верхняя граница 95% CI общего NRMSE | <= +3% | <= +7% |
| Верхняя граница 95% CI surface/pressure | <= +5% | <= +8% |
| Верхняя граница отдельного критического поля | <= +7% | <= +10% |
| Нижняя граница разности PSNR | >= -0.25 dB | >= -0.5 dB |
| Ухудшение spectral error | <= 5% | <= 10% |
| Фактический CR и exact roundtrip | обязательны | обязательны |

Публичных агрегированных значений CRA5 недостаточно для этого протокола:
необходимы per-frame outputs VAEformer на общей 28-канальной выборке.

## Эксперименты с предобученными моделями

Экспериментальный notebook:

```text
era5-28ch-pretrained-hyperprior-kaggle (1).ipynb
```

Цель эксперимента - проверить, переносится ли prior, выученный стандартным
RGB image codec, на многоканальные погодные поля при небольшом `N`.

### RGB hyperprior -> 28 каналов

За основу взят предобученный
`bmshj2018_hyperprior(quality=6, metric="mse")` из CompressAI.

Исходная модель принимает и восстанавливает три RGB-канала. Для погоды:

- первая convolution заменена с 3 на 28 входных каналов;
- последняя transposed convolution заменена с 3 на 28 выходных каналов;
- веса новых слоёв инициализированы средними RGB-весами;
- масштаб первой convolution скорректирован множителем `3/28`;
- hyperprior, entropy bottleneck и Gaussian conditional сохранены;
- нормированные погодные поля отображаются в диапазон `[0, 1]`.

Размер адаптированной модели:

```text
12.056 млн параметров
```

Это меньше лимита 20 млн. Предобучение выполнено на обычных RGB-изображениях,
а не на внешнем погодном архиве.

### Поэтапное дообучение

Эксперимент проведён на:

| Параметр | Значение |
|---|---:|
| Разрешение | 0.5 градуса |
| Уникальные train frames | 128 |
| Patches на timestamp | 4 |
| Всего локальных patches | 512 |
| Patch size | 192x192 |
| Batch size | 2 |
| Gradient accumulation | 2 |
| Optimizer steps | 12 000 |
| Целевая точка | 32x |

Обучение разбито на три стадии:

1. шаги 1-500 - только новые входной и выходной adapters;
2. шаги 501-2500 - analysis/synthesis transforms;
3. шаги 2501-12000 - весь codec вместе с entropy model.

Новые слои обучаются с `lr=5e-4`, backbone - с `lr=5e-5`.
Rate-distortion loss использует latitude weights и отдельную океанскую маску
для SST.

Теоретический бюджет для 32x:

```text
28 * 32 / 32 = 28 bpp
```

Training target с запасом был установлен в `27 bpp`.

### Настоящий ANS-bitstream

Эксперимент использует entropy coding CompressAI:

- `y` кодируется через Gaussian conditional;
- `z` кодируется через entropy bottleneck;
- оба потока записываются вместе с формами и размерами;
- целостность файла проверяется CRC32;
- до и после entropy coding сравниваются целочисленные символы `y` и `z`.

Таким образом, `exact latent roundtrip=True` относится к обоим уровням
hyperprior.

### Результаты transfer-эксперимента

Глобальная validation выполнена на 64 кадрах 2020 года. Настоящий codec
bitstream измерялся на 8 равномерно распределённых кадрах; на остальных
использовался быстрый deterministic quantized forward.

| Метрика | Значение |
|---|---:|
| All NRMSE | 0.210867 |
| Surface NRMSE | 0.235985 |
| Pressure NRMSE | 0.185748 |
| Mean PSNR | 27.861 dB |
| Actual compression ratio | 207.633x |
| Actual bpp | 4.315 |
| Exact latent roundtrip | да |
| `tp6h` NRMSE | 0.663786 |
| `T850` NRMSE | 0.142292 |
| `Q850` NRMSE | 0.224938 |

### Что показал эксперимент

Положительный результат:

- RGB hyperprior удалось технически перенести с 3 на 28 каналов;
- обучение стабилизировалось;
- reconstruction оставалась конечной без NaN/Inf;
- настоящий ANS-bitstream декодировался с exact latent roundtrip;
- модель прошла внутренние sanity-пороги.

Главное ограничение:

- модель не попала в запланированную точку 32x;
- вместо примерно `28 bpp` получилось `4.315 bpp`;
- фактическое сжатие составило около 208x.

То есть этот запуск нельзя честно сравнивать с основным codec при 32x:
transfer-модель решала существенно более жёсткую rate-distortion задачу.
Её более высокий `S_all=0.210867` одновременно обусловлен меньшим `N=128` и
намного более сильным фактическим сжатием.

В ходе обучения адаптивный rate weight достиг нижнего ограничения
`1e-7`, тогда как оценочный bitrate последних шагов держался около 4 bpp.
Это показывает, что простой multiplicative rate-controller не способен
поднять bitrate до 27-28 bpp, когда архитектура и learned entropy model
предпочитают значительно более компактный режим.

Практический вывод из transfer-ветки:

> Предобученный RGB codec полезен как быстро запускаемый baseline и источник
> устойчивого entropy model, но замена только первой и последней свёртки не
> гарантирует требуемую рабочую точку и не учитывает геометрию погодных полей.

Именно поэтому итоговый pipeline использует собственный weather-aware codec,
отдельные gain для 32x/64x и calibration по реальному размеру bitstream.

## Ограничения текущих результатов

- Основная test-оценка выполнена на 16 кадрах, а не на всём 2021 году.
- В текущем test manifest нет непрерывного семидневного блока.
- VAEformer per-frame outputs не приложены.
- Поэтому reconstruction scores валидны для выбранных кадров, но официальный
  non-inferiority verdict пока не вычислен.
- Transfer-hyperprior проверен на validation 2020 и фактической точке около
  208x, а не на сопоставимой точке 32x.
- Эксперимент 0.25 градуса ещё требует отдельного запуска.

## Ограничения хакатона

Основной pipeline контролирует:

- не более одной GPU;
- не более 24 GiB peak VRAM;
- не более 20 млн обучаемых параметров;
- не более 50 000 optimizer steps;
- не более 48 GPU-hours;
- отсутствие внешнего погодного pretraining;
- один совместный codec для всех 28 каналов.

## Запуск на Kaggle

1. Создайте Kaggle Notebook.
2. В `Settings -> Accelerator` выберите GPU P100 или T4.
3. Через `Add Input` подключите подготовленный ERA5 dataset.
4. Загрузите основной notebook.
5. Выберите режим:

```python
run_mode = "AUTO"
```

| Режим | Поведение |
|---|---|
| `AUTO` | загрузить совместимый checkpoint или обучить с нуля |
| `TRAIN` | обязательно обучить с нуля |
| `INFERENCE` | требовать checkpoint и выполнить оценку |

6. Выполните `Run All`.

Для нового обучения рекомендуется чистая Kaggle-сессия без checkpoint старой
архитектуры.

### Входные файлы основного pipeline

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

Для официального latent probe:

```text
probe_train.npy
probe_val.npy
probe_test.npy
probe_train_pairs.npy
probe_val_pairs.npy
probe_test_pairs.npy
```

Для повторной оценки без обучения:

```text
weather_codec_best.pt
```

или:

```text
weather_codec_final.pt
```

## Артефакты

Рабочая директория:

```text
/kaggle/working/era5_tz_codec/
```

Основные результаты:

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

Архив для сдачи:

```text
/kaggle/working/era5_tz_codec/era5_tz_submission.zip
```

Transfer-эксперимент сохраняет checkpoint и отчёты в:

```text
/kaggle/working/era5_transfer_hyperprior/
```

## Рекомендуемая структура репозитория

```text
.
├── README.md
├── nuclear-it-hackaton-solution.ipynb
├── experiments/
│   └── era5-28ch-pretrained-hyperprior-kaggle.ipynb
└── outputs/                         # небольшие JSON/CSV и графики
```

Большие `.npy`, checkpoints и bitstreams рекомендуется публиковать через
Kaggle Datasets, Git LFS или объектное хранилище.

## Воспроизводимость

Notebook сохраняет:

- seed и полную конфигурацию;
- порядок каналов и физические единицы;
- timestamps train/validation/test;
- число уникальных обучающих кадров;
- optimizer steps и число просмотренных примеров;
- checkpoint encoder/decoder;
- gain для 32x и 64x;
- bitstreams и их размеры;
- JSON/CSV с метриками;
- peak VRAM, runtime и GPU-hours;
- compliance report.

## Ссылки

- [ERA5](https://doi.org/10.1002/qj.3803)
- [WeatherBench 2](https://doi.org/10.1029/2023MS004019)
- [WeatherBench 2 Data Guide](https://weatherbench2.readthedocs.io/en/latest/data-guide.html)
- [CRA5 / VAEformer](https://github.com/taohan10200/CRA5)
- [CompressAI](https://github.com/InterDigitalInc/CompressAI)
- [Scale Hyperprior](https://arxiv.org/abs/1802.01436)

