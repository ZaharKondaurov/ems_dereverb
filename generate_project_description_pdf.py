#!/usr/bin/env python3
"""Generate a 3–5 page project description PDF for portfolio / review."""

from fpdf import FPDF

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"
OUTPUT = "/home/zakhar/ems_dereverb/opisanie_proekta.pdf"


class ProjectDescriptionPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", FONT)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.add_font("DejaVu", "I", FONT_ITALIC)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(
            0,
            8,
            "Описание проекта — FSPEN+ (Кондауров З.Д.)",
            align="R",
            new_x="LMARGIN",
            new_y="NEXT",
        )
        self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font("DejaVu", "", 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Страница {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(20, 60, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def subsection_title(self, title: str):
        self.set_font("DejaVu", "B", 10.5)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(0.5)

    def body_text(self, text: str):
        self.set_font("DejaVu", "", 9.5)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def bullet(self, text: str):
        self.set_font("DejaVu", "", 9.5)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(5, 5, "•")
        self.multi_cell(0, 5, text)
        self.set_x(x)
        self.ln(0.3)


def build_pdf() -> tuple[str, int]:
    pdf = ProjectDescriptionPDF()
    pdf.alias_nb_pages()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    # --- Title ---
    pdf.ln(6)
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(20, 60, 120)
    pdf.multi_cell(
        0,
        9,
        "FSPEN+ — легковесная модель шумоподавления\nи дереверберации для full-band аудио",
        align="C",
    )
    pdf.ln(4)
    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        0,
        6,
        "Описание проекта\n"
        "Кондауров Захар Дмитриевич\n"
        "ВКР НИУ ВШЭ, ОП «Прикладной анализ данных и искусственный интеллект»\n"
        "Соруководитель: ООО «Единое видео»",
        align="C",
    )
    pdf.ln(8)

    pdf.section_title("О проекте")
    pdf.body_text(
        "Проект решает задачу улучшения качества речевого сигнала (Speech Enhancement): "
        "совместное подавление шума и реверберации для моно-аудио 48 кГц в режиме реального "
        "времени. На основе легковесной архитектуры FSPEN (79 тыс. параметров, PESQ 2.97 на "
        "VoiceBank+DEMAND) разработаны четыре модификации, адаптированные к full-band обработке "
        "и совместной задаче denoise + dereverb. Лучшая конфигурация — FSPEN + 48 kHz + Overlap — "
        "содержит 95 тыс. параметров, достигает PESQ 2.35, NISQA-MOS 3.73, SRMR 9.6, STOI 0.88 "
        "при RTF 0.11."
    )
    pdf.body_text(
        "Репозиторий github.com/ZaharKondaurov/ems_dereverb объединяет исследовательскую часть "
        "(обучение, эксперименты, метрики) и продуктовую (веб-демо, потоковый инференс, Docker-деплой "
        "на Hugging Face Spaces). Проект выполнен в рамках ВКР и промышленного соруководства "
        "со стороны команды звуковых технологий."
    )

    # --- 1. Engineering ---
    pdf.section_title("1. Разработка и инженерия")

    pdf.subsection_title("Технологическая сложность и архитектура")
    pdf.body_text(
        "Решение сочетает DSP и глубокое обучение: модель работает в STFT-домене, предсказывает "
        "маски магнитуды и фазы, использует dual-path RNN (DPRNN) в bottleneck и параллельные "
        "full-band / sub-band энкодеры. Переход с 16 кГц на 48 кГц потребовал пересчёта N_FFT "
        "(512 → 1024), расширения sub-band групп (5 → 8), проектирования пересекающихся частотных "
        "диапазонов (Overlap) и расширения рецептивного поля (SBLE, SBDC). Конфигурации вынесены "
        "в типизированные пресеты (src/fspen_configs.py, src/web_models.py), что позволяет "
        "переключать четыре варианта модели без изменения кода приложения."
    )

    pdf.subsection_title("Инструменты разработки и DevOps")
    pdf.body_text(
        "Git (feature-ветки, GitHub + HF Spaces), Docker (python:3.10-slim, FastAPI, порт 7860), "
        "MLOps (чекпоинты, пресеты, RTF/MACs, requirements.txt). Потоковый инференс — "
        "StreamingEnhancer с hidden state, AGC, live/offline путями. Веб-стек: FastAPI + WebSocket "
        "+ спектрограммы + RTF; CLI: demo_mic.py, web_app.py."
    )

    pdf.subsection_title("Качество кода и проектирование")
    pdf.body_text(
        "Слои: models/ (FSPEN), src/ (датасет, loss, стриминг, конфиги, веб), notebooks/ (43 шт.). "
        "Pydantic API, threading.Lock, asyncio.to_thread. Переиспользованы FSPEN, pyroomacoustics, "
        "NISQA, бенчмарки DeepFilterNet/GTCRN; onnx/ptflops для экспорта и подсчёта MACs."
    )

    # --- 2. Data Science ---
    pdf.add_page()
    pdf.section_title("2. Data Science")

    pdf.subsection_title("Понимание специфики данных")
    pdf.body_text(
        "Речевой сигнал моделируется как x = s * r + n, где s — чистая речь, r — RIR, n — шум. "
        "Для обучения использованы VoiceBank+DEMAND (речь и 34 типа шумов, 48 кГц) и TAU Urban "
        "Acoustic Scenes 2019 (дополнительные шумы). RIR сгенерированы pyroomacoustics в четырёх "
        "группах комнат (площадь 15–120 м², RT60 0.4–1.0 с) с train/val/test разбиением 70/15/15. "
        "SNR выбирается из [0, 5, 10, 15] дБ; вероятность добавления шума и реверберации — 0.85. "
        "Класс SignalDataset реализует on-the-fly смешивание с ресэмплингом, нормализацией и "
        "контролем длины сегментов."
    )

    pdf.subsection_title("Предобработка и выбор модели")
    pdf.body_text(
        "Предобработка: STFT с vorbis-окном, представление спектрограммы в виде real/imag или "
        "magnitude/phase в зависимости от модификации. Функция потерь — STFT Multi-resolution loss "
        "(MR loss) с несколькими разрешениями FFT; экспериментировались дополнительные компоненты "
        "(low bands loss, clipping penalty). Выбор бейзлайна обоснован сравнительным анализом "
        "37 источников: FSPEN — лучший компромисс PESQ/параметры среди моделей < 1M; "
        "модификации проектировались итеративно (baseline → overlap → SBLE → SBDC+overlap)."
    )

    pdf.subsection_title("Эксперименты, метрики и валидация")
    pdf.body_text(
        "Метрики: PESQ, NISQA-MOS/NOISE, STOI, SRMR (качество) + MACs, RTF (производительность). "
        "Валидация на синтетической выборке (шум+reverb) и VoiceBank+DEMAND (denoise), сравнение "
        "с DeepFilterNet, GTCRN, FSPEN; тесты на реальных данных (real_data_test, EARS-Reverb). "
        "Метрики отслеживаются в train/val на каждой эпохе. Вывод: 48 кГц + Overlap дают "
        "максимальный прирост; PESQ ~17% ниже DFN, но параметров в 21× меньше."
    )

    # --- 3. AI application ---
    pdf.section_title("3. Применение ИИ")

    pdf.subsection_title("ИИ как предмет и результат проекта")
    pdf.body_text(
        "Ядро проекта — нейросетевая модель глубокого обучения для speech enhancement. "
        "Архитектура FSPEN+ использует свёрточные и рекуррентные блоки, предсказывает "
        "спектральные маски и обучается end-to-end на синтетических смесях. Для оценки качества "
        "применяется нейросетевая метрика NISQA (без необходимости ground truth на инференсе), "
        "что позволяет оценивать реальные записи."
    )

    pdf.subsection_title("ИИ-инструменты в процессе разработки")
    pdf.body_text(
        "В ходе работы над проектом использовались AI-инструменты для ускорения исследования "
        "и разработки:"
    )
    for item in [
        "LLM-ассистенты (Cursor Agent) — рефакторинг StreamingEnhancer, прототип web_app.py, "
        "генерация README и PDF-описаний, отладка fspen_configs;",
        "AI-агенты — исследование архитектурных модификаций, анализ ноутбуков метрик, "
        "подготовка Docker-деплоя для Hugging Face;",
        "Готовые AI-компоненты: NISQA (нейрометрика), DeepFilterNet/GTCRN (бенчмарки), "
        "Hugging Face Spaces (публичный AI-сервис).",
    ]:
        pdf.bullet(item)

    pdf.subsection_title("Практическая отдача от AI-подхода")
    pdf.body_text(
        "Ускорение цикла «гипотеза → эксперимент → метрики», быстрый MVP (live mic + file upload), "
        "публичный деплой на Hugging Face для обратной связи, потенциал интеграции в ASR и VoIP."
    )

    # --- 4. Product thinking ---
    pdf.section_title("4. Продуктовое мышление")

    pdf.subsection_title("Проблема и целевая аудитория")
    pdf.body_text(
        "Проблема: современные SE-модели (DeepFilterNet, MP-SENet) дают высокое качество, "
        "но требуют значительных вычислительных ресурсов; легковесные модели (FSPEN, GTCRN) "
        "ориентированы на 16 кГц и только шумоподавление. Целевая аудитория — системы "
        "с ограниченными ресурсами: наушники, смартфоны, умные колонки, VoIP-звонки, "
        "edge-предобработка для ASR. Пользователю нужна локальная обработка с минимальной "
        "задержкой (RTF < 1) без потери разборчивости речи в зашумлённых и реверберантных условиях."
    )

    pdf.subsection_title("Анализ конкурентов и продуктовые гипотезы")
    pdf.body_text(
        "Обзор 37 источников: тяжёлые (MP-SENet, DeepFilterNet) vs легковесные (GTCRN, FSPEN). "
        "Гипотезы: (H1) 48 кГц сохранит легковесность; (H2) overlap устранит артефакты; "
        "(H3) joint denoise+dereverb не ухудшит VBD; (H4) RTF < 0.15 на CPU. "
        "H1, H2, H4 подтверждены; H3 частично (PESQ −12% на VBD, NISQA-NOISE ≈ GT)."
    )

    pdf.subsection_title("MVP и оценка импакта")
    pdf.body_text(
        "MVP: Hugging Face Space (Live + File, 4 модели, спектрограммы, RTF), локальный "
        "web_app.py + Docker, CLI demo_mic.py (A/B bypass). Импакт: PESQ 1.68→2.35 (+40%), "
        "NISQA-MOS 2.73→3.73, SRMR 6.64→9.6, STOI 0.82→0.88; RTF 0.11 (9× быстрее реального "
        "времени); параметров в 21× меньше DeepFilterNet."
    )

    pdf.subsection_title("Обратная связь и развитие")
    pdf.body_text(
        "Публичное демо на Hugging Face обеспечивает тестирование без установки; в UI — "
        "рекомендация наушников (учёт акустической обратной связи). Соруководство "
        "ООО «Единое видео» задало прикладные требования к real-time. Направления развития:"
    )
    for item in [
        "перебалансировка MR loss для высоких частот;",
        "обучение на реальных RIR и более разнообразных корпусах;",
        "квантизация и перенос sub-band модулей из Python-циклов;",
        "устранение артефакта дублирования гармоник.",
    ]:
        pdf.bullet(item)

    pdf.ln(4)
    pdf.set_font("DejaVu", "I", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(
        0,
        5,
        "Репозиторий: https://github.com/ZaharKondaurov/ems_dereverb\n"
        "Веб-демо: https://huggingface.co/spaces/G1B-B0N/fspen_denoise_dereverb\n"
        "ВКР: vkr.pdf (38 стр., 37 источников)",
    )

    if pdf.page_no() < 3:
        raise RuntimeError(f"PDF too short: {pdf.page_no()} pages (expected 3–5)")
    if pdf.page_no() > 5:
        raise RuntimeError(f"PDF exceeds 5 pages: {pdf.page_no()} pages")

    pdf.output(OUTPUT)
    return OUTPUT, pdf.page_no()


if __name__ == "__main__":
    path, pages = build_pdf()
    print(f"Created: {path} ({pages} pages)")
