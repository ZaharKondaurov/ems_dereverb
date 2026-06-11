# FSPEN+

**FSPEN+** — легковесная нейросетевая модель для улучшения качества речевого сигнала в реальном времени: совместное **подавление шума и реверберации** для аудио с частототой дискретизации **48 kHz**. Архитектура развивает идею FSPEN (full-band + sub-band, dual-path RNN): исходный FSPEN ориентирован на шумоподавление. Здесь те же принципы перенесены на full-band обработку и задачу деревербации при сохранении умеренной вычислительной стоимости и размера модели.

Модель рассчитана на потоковую обработку и пакетную (файлы). В репозитории — обучение, инференс, локальные демо и веб-интерфейс с четырьмя вариантами конфигурации (baseline, overlap, SBLE, SBDC+overlap).

**Ключевые особенности:**

- обработка **full-band** моно-аудио;
- одновременное **denoise + dereverb** (обучение на синтетических смесях речи, шума и RIR);
- несколько пресетов под разный компромисс качество / RTF / задержка;
- демо с отображением **RTF** (real-time factor) и спектрограмм входа/выхода.

## Про FSPEN+

Бейзлайн был изначально предназначен только для задачи шумоподавления для **16 kHz** аудио, поэтому модель имеет некоторые слабые места. Чтобы от них избавиться, были предложены модификации модели.

Во-первых, из повышения частоты дискретизации входного сигнала следует повышение ```N_STFT``` с $512$ до $1024$. Значит, понялись входные размерности и изначальное разбиение на группы из FSPEN не подходит. Чтобы учесть расширение частотного диапазона, были было увеличено количество частотных групп в sub-band слоях.

Во-вторых, в бейзлайне свертки в sub-band энкодере обрабатывали группы независимо друг отдруга, хотя частоты физически связанны между собой. Чтобы избавиться от независимости групп, можно задавать их частотные диапазоны с пересечением. Кроме того, независимость групп может создавать искажения на границах. Чтобы избежать этого, границы частотных групп берутся с пересечением. Перечисленные пункты на схемах будут называться **Splitting into groups**.

В-третьих, одной из проблем архитектуры является малое рецептивное поле и менее информативные признаки. Чтобы это исправить, было решено увеличить количество слоёв в sub-band encoder/decoder. Также были подвергнуты изменениям sub-band свёрточные слои. Были заменены функции активации, а в sub-band decoder вместо линейных слоёв используются свёртки. Последние активации у full-band и sub-band decoder были заменены на ```nn.Sigmoid``` для масок для магнитуды, на ```nn.Tanh``` для масок для фазы.

Также в некоторых модификациях для full-band encoder было изменено представление входа с вещественной и мнимой части на "магинтуду-фазу"

Получились следующие модификации:

* FSPEN + 48 kHz – sub-band энкодер изменён таким образом, чтобы он мог работать со спектрограммой, полученной при помощи STFT с N_FFT = 1024.
* FSPEN + 48 kHz + overlap – границы частотных групп изменены так, чтобы они были с перекрытием.
* FSPEN + 48 kHz + sub-band layers extension (SBLE) – увеличение количества слоёв в sub-band энкодере и декодере, изменение слоёв энкодера с линейных на свёртки, изменение представления спектрограммы с вещественной и мнимой части на магнитуду и фазу.
* FSPEN + 48 kHz + sub-band decoder conv (SBDC) + overlap – пересечение частотных групп, изменение слоёв энкодера с линейных на свёртки, изменение представления спектрограммы с вещественной и мнимой части на магнитуду и фазу.

<div align="center" style="background-color:#ffffff;padding:16px;border-radius:4px;">
  <img src="images/fspen-48kHz.svg" alt="Схема FSPEN + 48 kHz + overlap" width="100%">
</div>
<p align="center">Схема FSPEN + 48 kHz + overlap</p>

<div align="center" style="background-color:#ffffff;padding:16px;border-radius:4px;">
  <img src="images/fspen-sble.svg" alt="Схема FSPEN + 48 kHz + SBLE" width="100%">
</div>
<p align="center">Схема FSPEN + 48 kHz + sub-band layers extension (SBLE)</p>

<div align="center" style="background-color:#ffffff;padding:16px;border-radius:4px;">
  <img src="images/fspen-sbdc.svg" alt="Схема FSPEN + 48 kHz + SBDC + overlap" width="100%">
</div>
<p align="center">Схема FSPEN + 48 kHz + sub-band decoder conv (SBDC) + overlap</p>

## Быстрый старт

Для проверки работоспособности моделей можно воспользовать Hugging Face Space https://huggingface.co/spaces/G1B-B0N/fspen_denoise_dereverb. Он содержит в себе 2 режима работы модели: Live, File.

В Live модель в реальном времени обрабатывает аудио, захаватываемый микрофоном, демонстрируя то, как выглядит спектрограмма изначально и после обработки. Кроме того, параллельно воспроизводится восстановленный сигнал, чтобы можно услышать результат работы модели (**рекомендуется использовать наушники**). Также предусмотрен мониторинг значения ```RTF```.

В разделе File можно обработать уже записанный файл. Для долгих записей рекомендуется использовать чанкирование(нажать на соответствующую кнопку)

Для каждого раздела можно выбрать одну из 4-х моделей и размер чанка.

Если Hugging Face Space не запускается, демо можно запустить локально. Перед запуском установите зависимости из `requirements.txt`.

### `web_app.py` — веб-демо (Live + File)

Запускает FastAPI-сервер с браузерным интерфейсом: потоковая обработка микрофона по WebSocket, загрузка файлов, выбор модели и отображение спектрограмм.

```
python web_app.py --preset fspen_48khz_overlap --chunk-ms 500 --host 0.0.0.0 --port 7860
```

| Аргумент | По умолчанию | Назначение |
| -------- | ------------ | ---------- |
| `--preset` | `fspen_48khz_overlap` | Идентификатор пресета модели. Доступные значения: `fspen_48khz`, `fspen_48khz_overlap`, `fspen_48khz_sble`, `fspen_48khz_sbdc_overlap`. Пресет задаёт конфиг, чекпоинт и класс модели. Полный список также отдаётся эндпоинтом `/api/catalog`. |
| `--device` | `cpu` | Устройство для инференса PyTorch: `cpu` или `cuda`. Если CUDA недоступна, автоматически используется CPU. |
| `--chunk-ms` | `512` | Длина обработки одного чанка в миллисекундах. Влияет на задержку и RTF: меньшее значение — ниже задержка, но выше накладные расходы; фактический размер чанка не меньше `4 × hop_length`. |
| `--history-sec` | `2.5` | Длина истории аудио (в секундах), которая накапливается для отрисовки спектрограмм в Live-режиме. |
| `--host` | `0.0.0.0` | Адрес, на котором слушает HTTP/WebSocket-сервер. `0.0.0.0` — доступ с других машин в сети. |
| `--port` | `7860` | TCP-порт сервера (тот же, что у Hugging Face Space). |

Те же параметры можно задать переменными окружения (удобно в Docker): `FSPEN_PRESET`, `FSPEN_DEVICE`, `FSPEN_CHUNK_MS`.

### `demo_mic.py` — CLI-демо (микрофон или файл)

Обрабатывает аудио без браузера: live-режим с микрофона (A/B: enhanced / bypass) или офлайн-обработка WAV-файла. В live-режиме клавиши: `E` — модель включена, `B` — сухой сигнал с той же задержкой, `Q` — выход.

```
python demo_mic.py --config TrainConfig_48kHz_overlap --checkpoint checkpoints/fspen_chkp/TrainConfig_48kHz_overlap.pt --chunk-ms 250
python demo_mic.py --list-devices
python demo_mic.py --config TrainConfig_48kHz_overlap --checkpoint checkpoints/fspen_chkp/TrainConfig_48kHz_overlap.pt --file noisy.wav --out enhanced.wav --chunked
```

| Аргумент | По умолчанию | Назначение |
| -------- | ------------ | ---------- |
| `--checkpoint` | `checkpoints/fspen_chkp/TrainConfig_48kHz_enc_ext_1986#0.pt` | Путь к файлу весов модели (`.pt`). Должен соответствовать `--config`. |
| `--config` | `TrainConfig_48kHz_enc_ext` | Имя класса конфигурации из `src/fspen_configs.py` (размер STFT, sub-band группы, архитектура). Примеры: `TrainConfig_48khz`, `TrainConfig_48kHz_overlap`, `TrainConfig_48kHz_enc_ext`, `TrainConfig_48kHz_enc_ext_lay_1_overlap`. |
| `--device` | `cpu` | Устройство PyTorch: `cpu` или `cuda`. |
| `--chunk-ms` | `512` | Длина чанка инференса в миллисекундах. Определяет, какими порциями модель обрабатывает поток; минимум — `4 × hop_length` сэмплов. |
| `--block-ms` | `10` | Размер блока аудио ввода-вывода (миллисекунды) для PortAudio/sounddevice. Меньше — ниже задержка callback'а, но выше нагрузка на CPU; минимум 64 сэмпла. |
| `--input-device` | системный по умолчанию | Индекс входного аудиоустройства (микрофон). Список индексов — `--list-devices`. |
| `--output-device` | системный по умолчанию | Индекс выходного аудиоустройства (наушники/колонки). |
| `--list-devices` | выкл. | Вывести список PortAudio-устройств и завершить работу. |
| `--file` | — | Путь к входному WAV. Если указан, скрипт работает в офлайн-режиме (без микрофона). |
| `--out` | `<имя_файла>_enhanced.wav` | Путь для сохранения результата в офлайн-режиме. |
| `--chunked` | выкл. | В офлайн-режиме: обрабатывать длинный файл по чанкам (как в live), а не целиком одним STFT-проходом. Полезно для длинных записей и оценки потокового поведения. |
| `--enhanced` | вкл. | Стартовать в режиме с включённой моделью (по умолчанию). |
| `--bypass` | выкл. | Стартовать в режиме bypass: сухой сигнал с той же алгоритмической задержкой, без обработки моделью. Перекрывает `--enhanced`. |

### `train_fspen.py` — обучение моделей

Скрипт обучения моделей с валидацией STOI/NISQA, сохранением чекпоинтов и графиков. В качестве лосса берётся Multi-resolution loss.

**Два способа задать параметры** (как в большинстве ML-проектов):

1. **YAML-файл** — рецепт эксперимента в `configs/train/`, удобно версионировать и воспроизводить.
2. **CLI-флаги** — точечные переопределения; **имеют приоритет над YAML**.

```
python train_fspen.py --list-configs

# YAML (рекомендуется для полного эксперимента)
python train_fspen.py --train-config configs/train/fspen_48khz_overlap.yaml

# YAML + переопределение отдельных полей
python train_fspen.py -c configs/train/fspen_48khz_overlap.yaml --epochs 10 --device cpu

# Посмотреть итоговый конфиг после слияния YAML + CLI
python train_fspen.py -c configs/train/fspen_48khz_overlap.yaml --dump-config

# Только CLI
python train_fspen.py \
  --config TrainConfig_48kHz_overlap \
  --model FullSubPathExtension \
  --model-name TrainConfig_48kHz_overlap \
  --epochs 50 --batch-size 32 --device cuda
```

Готовые YAML-рецепты обучения:

| Модель | YAML |
| ------ | ---- |
| FSPEN baseline (16 kHz) | `configs/train/fspen_baseline.yaml` |
| FSPEN + 48 kHz | `configs/train/fspen_48khz.yaml` |
| FSPEN + 48 kHz + overlap | `configs/train/fspen_48khz_overlap.yaml` |
| FSPEN + 48 kHz + SBLE | `configs/train/fspen_48khz_enc_ext.yaml` |
| FSPEN + 48 kHz + SBDC + overlap | `configs/train/fspen_48khz_enc_ext_lay_1_overlap.yaml` |

| Аргумент | По умолчанию | Назначение |
| -------- | ------------ | ---------- |
| `-c`, `--train-config` | — | Путь к YAML с гиперпараметрами обучения. CLI-флаги переопределяют значения из файла. |
| `--dump-config` | выкл. | Вывести итоговый конфиг (YAML + CLI) и выйти. |
| `--config` | — | **Обязателен** (в CLI или YAML). Класс конфигурации из `src/fspen_configs.py`: `TrainConfig`, `TrainConfig_baseline`, `TrainConfig_48khz`, `TrainConfig_48kHz_overlap`, `TrainConfig_48kHz_enc_ext`, `TrainConfig_48kHz_enc_ext_lay_1_overlap`. |
| `--model` | из `--config` | Класс модели: `FullSubPathExtension` или `FullSubPathExtension_ext`. |
| `--eval-fn` | из `--config` | Функция прямого прохода: `model_eval_old` (real/imag) или `model_eval` (mag/phase). |
| `--model-name` | `{config}_{seed}` | Базовое имя чекпоинта; при коллизии добавляется суффикс `#N`. |
| `--resume` | — | Путь к `.pt` для продолжения обучения (веса, optimizer, scheduler, plots). |
| `--list-configs` | выкл. | Показать доступные конфиги и пары model/eval, затем выйти. |
| `--epochs` | `50` | Число эпох обучения. |
| `--batch-size` | `32` | Batch size на одно GPU; при нескольких GPU умножается на их число. |
| `--lr` | `5e-4` | Learning rate Adam. |
| `--scheduler-step` | `4` | StepLR: период снижения LR (эпохи). |
| `--scheduler-gamma` | `0.98` | StepLR: множитель LR на каждом шаге. |
| `--val-every` | `1` | Запускать валидацию каждые N эпох. |
| `--plot-every` | `1` | Сохранять PNG с метриками в `{chkp-dir}/plots/` каждые N эпох. |
| `--num-workers` | `4` | `num_workers` DataLoader. |
| `--seed` | `42` | Seed для random/numpy/torch. |
| `--device` | `cuda` | `cpu` или `cuda`. |
| `--chkp-dir` | `checkpoints/fspen_chkp` | Папка для чекпоинтов и графиков. |
| `--data-dir` | `data/DS_10283_2791/clean_trainset_56spk_wav` | Чистая речь (train). |
| `--val-data-dir` | `data/DS_10283_2791/clean_trainset_28spk_wav` | Чистая речь (val). |
| `--noise-dir-train` | `data/demand_train` | Шумы для train. |
| `--noise-dir-val` | `data/demand_val` | Шумы для val. |
| `--rir-dirs-train` | 4 папки `rirs48_*_3_train` | RIR для train (через запятую). |
| `--rir-dirs-val` | 4 папки `rirs48_*_3_val` | RIR для val (через запятую). |
| `--snr` | `0,5,10,15` | SNR в дБ для смешивания (через запятую). |
| `--noise-proba` | `0.85` | Вероятность добавления шума. |
| `--rir-proba` | `0.85` | Вероятность добавления реверберации. |
| `--max-seq-len-sec` | `4.0` | Длина обучающего сегмента в секундах. |
| `--val-partition` | `5000` | Лимит val-файлов; `0` — использовать все. |
| `--nisqa-config` | `NISQA_s/config/nisqa_s.yaml` | Конфиг NISQA для метрик валидации. |
| `--no-nisqa` | выкл. | Отключить NISQA (быстрее, без зависимости NISQA_s). |
| `--warmup-dataset-epoch` | выкл. | Вызвать `set_epoch(1)` на датасетах до цикла обучения (совпадение с ноутбуками). |
| `--deterministic` | вкл. (YAML) | Детерминированное обучение: `cudnn` deterministic, TF32 off, `num_workers` принудительно `0`. Отключить: `deterministic: false` в YAML. |

`eval_fn` выбирается автоматически по `--config` (`model_eval` для ext-конфигов, иначе `model_eval_old`).

Чекпоинты сохраняются каждую эпоху; лучший по val NISQA-MOS — в `{model-name}_best_mos.pt`.

### `test_fspen.py` — оценка чекпоинта

На синтетическом тестовом наборе даннаых (речь + шум + RIR) считает метрики PESQ / NISQA / STOI / SI-SDR / SRMR / DNSMOS, опционально можно проверить производительность модели и получить RTF (полный файл и потоковые чанки).

**Два способа задать параметры:** YAML в `configs/test/` + CLI-переопределения (CLI имеет приоритет).

```
python test_fspen.py -c configs/test/fspen_48khz_overlap.yaml

python test_fspen.py -c configs/test/fspen_48khz_overlap.yaml \
  --checkpoint checkpoints/fspen_chkp/TrainConfig_48kHz_overlap.pt \
  --output-csv metrics/overlap_test.csv

python test_fspen.py --checkpoint checkpoints/fspen_chkp/model.pt \
  --config TrainConfig_48kHz_overlap \
  --max-samples 100 --device cpu

python test_fspen.py -c configs/test/fspen_48khz_overlap.yaml --dump-config
```

Готовые YAML-рецепты тестирования:

| Модель | YAML |
| ------ | ---- |
| FSPEN + 48 kHz | `configs/test/fspen_48khz.yaml` |
| FSPEN + 48 kHz + overlap | `configs/test/fspen_48khz_overlap.yaml` |
| FSPEN + 48 kHz + SBLE | `configs/test/fspen_48khz_enc_ext.yaml` |
| FSPEN + 48 kHz + SBDC + overlap | `configs/test/fspen_48khz_enc_ext_lay_1_overlap.yaml` |

| Аргумент | По умолчанию | Назначение |
| -------- | ------------ | ---------- |
| `-c`, `--test-config` | — | Путь к YAML с параметрами оценки. |
| `--dump-config` | выкл. | Вывести итоговый конфиг и выйти. |
| `--checkpoint` | — | **Обязателен** (в CLI или YAML). Путь к `.pt` чекпоинту. |
| `--config` | из чекпоинта | Класс конфигурации: `TrainConfig_48kHz_overlap`, `TrainConfig_48kHz_enc_ext`, и др. |
| `--model` | из `--config` | `FullSubPathExtension` или `FullSubPathExtension_ext`. |
| `--test-dir` | `data/DS_10283_2791/clean_testset_wav` | Чистая речь для теста. |
| `--noise-dir` | `data/demand_test` | Шумы для теста. |
| `--rir-dirs` | 4 папки `rirs48_*_3_test` | Папки с RIR (через пробел или запятую). |
| `--snr` | `0 5 10 15` | SNR в дБ (через пробел или запятую). |
| `--noise-proba` | `0.85` | Вероятность добавления шума. |
| `--rir-proba` | `0.85` | Вероятность добавления реверберации. |
| `--dataset-epoch` | `99` | Эпоха датасета (`set_epoch`) — фиксирует выбор RIR/шума. |
| `--seed` | `1984` | Seed датасета (`base_seed`). |
| `--num-workers` | `0` | `num_workers` DataLoader. |
| `--max-samples` | все | Ограничить число оцениваемых файлов. |
| `--device` | `cuda` | Устройство для метрик: `cpu` или `cuda`. |
| `--normalize-output` | вкл. | Нормализовать выход по пику входного сигнала. |
| `--no-normalize-output` | выкл. | Отключить нормализацию выхода. |
| `--nisqa-config` | `NISQA_s/config/nisqa_s.yaml` | Конфиг NISQA. |
| `--no-nisqa` | выкл. | Отключить NISQA. |
| `--output-csv` | — | Путь для сохранения усреднённых метрик (CSV). |
| `--benchmark` | вкл. (YAML) | Замерить RTF (полный файл + потоковые чанки). |
| `--no-benchmark` | выкл. | Пропустить бенчмарк производительности. |
| `--benchmark-device` | как `--device` | Устройство для RTF; в YAML по умолчанию `cpu`. |
| `--benchmark-chunk-size` | `n_fft × 5` | Размер чанка для потокового RTF. |
| `--benchmark-max-samples` | все | Ограничить число файлов для бенчмарка. |

`eval_fn` выбирается автоматически по `--config` (как при обучении).

### `generate_data.py` — генерация данных

Разбиение DEMAND-шума и симуляция RIR через pyroomacoustics с последующим split на train / val / test.

**Задачи:**

- `split_demand` — `data/demand` → `demand_train` / `demand_val` / `demand_test` (доли 70% / 15% / 15%);
- `generate_rirs` — для каждого пресета комнаты: генерация wav + `meta.csv`, затем split в `*_train`, `*_val`, `*_test`.

```
python generate_data.py -c configs/data/default.yaml

python generate_data.py --tasks split_demand

python generate_data.py --tasks generate_rirs --rir-presets small

python generate_data.py --tasks generate_rirs --rir-presets small medium --seed 1984

python generate_data.py -c configs/data/default.yaml --dump-config
```

**Пресеты RIR** (`--rir-presets`):

| preset | папка | count | rt60 |
| ------ | ----- | ----- | ---- |
| `small` | `data/rirs48_small_3` | 200 | 0.4–0.5 |
| `medium` | `data/rirs48_medium_3` | 400 | 0.5–0.7 |
| `large` | `data/rirs48_large_3` | 160 | 0.70–0.85 |
| `super_large` | `data/rirs48_super_large_3` | 80 | 0.85–1.0 |

| Аргумент | По умолчанию | Назначение |
| -------- | ------------ | ---------- |
| `-c`, `--data-config` | — | Путь к YAML (`configs/data/default.yaml`). |
| `--dump-config` | выкл. | Вывести итоговый конфиг и выйти. |
| `--tasks` | `split_demand generate_rirs` | Задачи (через пробел): `split_demand`, `generate_rirs`. |
| `--seed` | `1984` | Seed для random/numpy/torch и split. |
| `--sample-rate` | `48000` | Частота дискретизации RIR. |
| `--stimulus` | `data/.../p226_001.wav` | Стимул для симуляции комнаты (pyroomacoustics). |
| `--rir-presets` | все 4 пресета | Какие группы комнат генерировать. |
| `--train-fraction` | `0.7` | Доля train при первом split (demand и RIR). |
| `--val-fraction` | `0.5` | Доля val из оставшейся части (итого ~15% val, ~15% test). |
| `--demand-src-dir` | `data/demand` | Исходная папка DEMAND. |
| `--demand-train-dir` | `data/demand_train` | Выход train для шума. |
| `--demand-val-test-dir` | `data/demand_val_test` | Промежуточная папка (30%). |
| `--demand-val-dir` | `data/demand_val` | Выход val для шума. |
| `--demand-test-dir` | `data/demand_test` | Выход test для шума. |

Полная генерация всех 4 пресетов RIR занимает ~10+ минут на CPU; для проверки начните с `--rir-presets small`.

## Датасет

RIR и split DEMAND можно пересоздать скриптом `generate_data.py` (см. выше). Исходные данные:

Модель была обучена на датасетах:
* [VoiceBank+Demand](https://datashare.ed.ac.uk/handle/10283/2791?show=full) — аудио с речью и шумы 
* [TAU Urban Acoustic Scenes 2019](https://zenodo.org/records/2589280) — дополнительные шумы

Файлы с RIR были сгенерированы при помощи Python библиотеки pyroomacoustics. RIR файлы были разделены на 4 группы с разными параметрами. Общие параметры каждой группы: высота комнаты $\text{uniform}(2.5, 3.0)$ метров, отношение сторон комнаты $[1 : 3, 2 : 3, 1 : 1]$ (вероятность выбора $0.5$, $0.25$, $0.25$ соответственно). Каждая группа была разбита на train, validation, test следующим образом: $0.7$, $0.15$, $0.15$. Уникальные параметры групп следующие:

* Первая группа представляет из себя комнату малого размера (прямоугольный параллелепипед) с площадью комнаты из $\text{uniform}(15, 30) m^2$ , $\text{rt60} = \text{uniform}(0.4, 0.5)$, 200 файлов.
* Вторая группа – комната среднего размера: площадь $= \text{uniform}(30, 80) m^2$, $\text{rt60} = \text{uniform}(0.5, 0.7)$, 400 файлов.
* Третья группа – комната большого размера: площадь $= \text{uniform}(80, 120) m^2$, $\text{rt60} = \text{uniform}(0.7, 0.85)$, 160 файлов.
* Четвёртая группа – комната большого размера с длинной реверберацией: площадь $= \text{uniform}(80, 120) m^2$ , $\text{rt60} = \text{uniform}(0.85, 1.)$, 80 файлов.

Шум тоже был разбит на 3 выборки таким образом (train,
validation, test): 1318, 309, 180. Всего 34 вида шума, каждая часть данных содержит все виды шума. ```SNR``` для смешивания шума с сигналом выбирался случайно из списка $[0, 5, 10, 15]$ (значения равновероятны)


## Метрики

Результаты показывают, что ключевым фактором повышения качества является переход FSPEN на обработку аудио с частотой 48 кГц и способность подавлять реверберацию. Предлагаемые модели достигают существенного роста всех основных метрик относительно исходного FSPEN при сохранении крайне малой вычислительной сложности. По сравнению с бейзлайном, количество параметров выросло не более чем в 2 раза, а количество MACs не более чем в 3.  Наилучший наилучшие показатели подавления шума и реверберации и баланс между качеством и качеством показывает  FSPEN + 48 kHz + Overlap. Несмотря на некоторое отставание от DeepFilterNet по абсолютному качеству, предложенные модели требуют на порядок меньше параметров и вычислений, что делает их привлекательными для применения на ресурсно-ограниченных устройствах и в системах реального времени. Более того, обученные модели очень близки к образцовому аудио по метрикам шумоподавления.

| Model    | Params. | MACs | PESQ | NISQA-MOS | NISQA-NOISE | SRMR | STOI | RTF (@ 1 s) |
| -------   | -------- | --| -- | -- | -- | -- | -- | -- |
| Noisy and Reverb | — | — | $1.68$  | $2.73$ | $2.40$ | $6.64$ | $0.82$ | — |
| Ground Truth | — | — | — | $4.07$ | $4.12$ | $8.94$ | $1.00$ | — |
| FSPEN | $79\text{k}$ | $2.6\text{M}$ | $2.22 \pm 0.18 $ | $3.55 \pm 0.06$ | $3.83 \pm 0.08$ | $8.87 \pm 0.60$ | $0.77 \pm 0.00$ | $0.08 \pm 0.01$  |
| FSPEN + 48 kHz (prop.) | $93\text{k}$ | $5.6\text{M}$ | $2.39 \pm 0.19$ | $3.74 \pm 0.13$ | $3.99 \pm 0.12$ | $9.47 \pm 0.65$ | $0.88 \pm 0.02$ | $0.11 \pm 0.01$ | 
| FSPEN + 48 kHz + SBLE (prop.) | $137\text{k}$ | $6.2\text{M}$ | $2.32 \pm 0.05$ | $3.67 \pm 0.13$ | $3.87 \pm 0.03$ | $7.8 \pm 0.09$ | $0.80 \pm 0.00$ | 0.13 \pm 0.03 |
| FSPEN + 48 kHz + SBDC + Overlap (prop.) | $111\text{k}$ | $6.0\text{M}$ | $2.33 \pm 0.08$ | $3.64 \pm 0.16$ | $3.88 \pm 0.08$ | $7.8 \pm 0.21$ | $0.80 \pm 0.01$ | $0.13 \pm 0.01$ |
| FSPEN + 48 kHz + Overlap (prop.)| $95\text{k}$ | $5.6\text{M}$ | $2.35 \pm 0.28$ | $3.73 \pm 0.13$ | $4.00 \pm 0.12$ | $9.6 \pm 0.45$ | $0.88 \pm 0.01$ | $0.11 \pm 0.01$ |
| GTCRN | $48\text{k}$ | $8.1\text{M}$ | $2.10$ | — | — | $8.22$ | $0.86$ | $0.03$ |


