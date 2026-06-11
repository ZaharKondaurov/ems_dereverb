#!/usr/bin/env python3
"""Generate a brief 3-page PDF summary of the thesis."""

from fpdf import FPDF

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUTPUT = "/home/zakhar/ems_dereverb/vkr_kratkoe_opisanie.pdf"


class ThesisSummaryPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font("DejaVu", "", FONT)
        self.add_font("DejaVu", "B", FONT_BOLD)
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "B", 9)
        self.set_text_color(100, 100, 100)
        self.cell(
            0,
            8,
            "Краткое описание ВКР — Кондауров З.Д.",
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

    def section_title(self, title):
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(20, 60, 120)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("DejaVu", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("DejaVu", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(5, 5.5, "•")
        self.multi_cell(0, 5.5, text)
        self.set_x(x)
        self.ln(1)


def build_pdf():
    pdf = ThesisSummaryPDF()
    pdf.alias_nb_pages()
    pdf.set_margins(20, 20, 20)
    pdf.add_page()

    # Title page block
    pdf.ln(8)
    pdf.set_font("DejaVu", "B", 16)
    pdf.set_text_color(20, 60, 120)
    pdf.multi_cell(
        0,
        9,
        "Легковесная модель шумоподавления\nи дереверберации с малыми\nвычислительными затратами",
        align="C",
    )
    pdf.ln(6)

    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(
        0,
        6,
        "Краткое описание выпускной квалификационной работы\n"
        "Кондаурова Захара Дмитриевича\n"
        "НИУ ВШЭ, направление 01.03.02 «Прикладная математика и информатика»\n"
        "Образовательная программа «Прикладной анализ данных и искусственный интеллект»",
        align="C",
    )
    pdf.ln(10)

    pdf.section_title("Аннотация")
    pdf.body_text(
        "Работа посвящена задаче улучшения качества речевого сигнала (Speech Enhancement) — "
        "совместному подавлению шума и реверберации в аудио с частотой дискретизации 48 кГц "
        "в режиме, пригодном для работы в реальном времени. Современные нейросетевые решения "
        "достигают высокого качества, но часто требуют значительных вычислительных ресурсов "
        "и ориентированы на аудио 16 кГц и задачу только шумоподавления. "
        "Цель работы — разработать и обучить легковесную модель (не более 1 млн параметров), "
        "способную одновременно подавлять шум и реверберацию при сохранении приемлемого "
        "качества и скорости обработки."
    )

    pdf.section_title("Постановка задачи")
    pdf.body_text(
        "Искажённый речевой сигнал моделируется как свёртка чистой речи с импульсной "
        "характеристикой помещения (RIR) с добавлением шума: x = s * r + n. "
        "Модель должна восстанавливать чистый сигнал, работая со спектрограммой, "
        "полученной через STFT. Ключевые требования к итоговой системе:"
    )
    for item in [
        "обработка full-band аудио (48 кГц);",
        "совместное шумоподавление и дереверберация;",
        "не более 1 млн параметров;",
        "RTF < 1 (обработка быстрее длительности сигнала).",
    ]:
        pdf.bullet(item)

    pdf.section_title("Подход и архитектура")
    pdf.body_text(
        "В качестве базовой архитектуры выбрана модель FSPEN (79 тыс. параметров, PESQ 2.97 "
        "на VoiceBank+DEMAND) — одна из лучших легковесных моделей шумоподавления. "
        "FSPEN сочетает full-band и sub-band энкодеры, рекуррентный bottleneck на основе "
        "DPRNN (Dual Path Enhancer) и симметричные декодеры, предсказывающие маски "
        "для спектрограммы. Обучение выполняется с функцией потерь STFT Multi-resolution loss."
    )
    pdf.body_text(
        "Для адаптации FSPEN к задаче совместного шумоподавления и дереверберации "
        "на 48 кГц предложены четыре модификации:"
    )
    for item in [
        "FSPEN + 48 kHz — адаптация STFT (N_FFT = 1024) и sub-band групп (5 → 8);",
        "FSPEN + 48 kHz + Overlap — пересекающиеся частотные группы в sub-band энкодере;",
        "FSPEN + 48 kHz + SBLE — расширение sub-band слоёв, свёртки вместо линейных, "
        "вход full-band энкодера в виде магнитуды и фазы;",
        "FSPEN + 48 kHz + SBDC + Overlap — пересечение групп, свёртки в sub-band декодере, "
        "магнитуда-фаза на входе.",
    ]:
        pdf.bullet(item)

    pdf.add_page()
    pdf.section_title("Данные и обучение")
    pdf.body_text(
        "Модели обучались на синтетических смесях речи, шума и реверберации. "
        "Использовались корпуса VoiceBank+DEMAND (речь и шумы) и TAU Urban Acoustic Scenes 2019 "
        "(дополнительные шумы). RIR генерировались библиотекой pyroomacoustics в четырёх группах "
        "комнат разного размера (площадь 15–120 м², RT60 0.4–1.0 с). "
        "SNR при смешивании выбирался из [0, 5, 10, 15] дБ. "
        "Вероятность добавления шума и реверберации — по 0.85. "
        "Обучение: оптимизатор Adam, batch size 32, 50 эпох, learning rate × 0.98 каждые 4 эпохи."
    )

    pdf.section_title("Результаты")
    pdf.body_text(
        "На синтетической тестовой выборке (шум + реверберация) лучшим вариантом "
        "оказалась модификация FSPEN + 48 kHz + Overlap (95 тыс. параметров, 5.6M MACs):"
    )
    for item in [
        "PESQ: 2.35 (против 1.68 у искажённого сигнала и 1.56 у исходного FSPEN);",
        "NISQA-MOS: 3.73, NISQA-NOISE: 4.00 (близко к ground truth: 4.07 / 4.12);",
        "STOI: 0.88, SRMR: 9.6 (эффективное подавление реверберации);",
        "RTF: 0.11 (работа в реальном времени).",
    ]:
        pdf.bullet(item)

    pdf.body_text(
        "По сравнению с DeepFilterNet (2M параметров, 85.6M MACs) предложенная модель "
        "уступает по абсолютным метрикам качества (PESQ ниже на ~17%, NISQA-MOS на ~11%), "
        "но в 21 раз меньше по числу параметров и в 15 раз — по MACs. "
        "На бенчмарке VoiceBank+DEMAND (только шумоподавление) модификации сохраняют "
        "способность к denoise: PESQ 2.64–2.68 против 2.97 у исходного FSPEN "
        "(снижение ~12% из-за более сложной задачи обучения)."
    )

    pdf.section_title("Выводы")
    pdf.body_text(
        "Цель работы достигнута: разработаны легковесные модификации FSPEN для совместного "
        "шумоподавления и дереверберации full-band аудио в реальном времени. "
        "Ключевым фактором повышения качества стали переход на 48 кГц и пересекающиеся "
        "частотные группы (Overlap). Модели применимы на ресурсно-ограниченных устройствах "
        "(наушники, смартфоны, умные колонки), в аудиозвонках и как предобработка для ASR."
    )
    pdf.body_text(
        "Направления дальнейших исследований: перебалансировка MR loss для высоких частот, "
        "обучение на более разнообразных корпусах с реальными RIR, оптимизация реализации "
        "(перенос sub-band модулей из Python-циклов, квантизация), устранение артефакта "
        "дублирования гармоник."
    )

    pdf.ln(4)
    pdf.set_font("DejaVu", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(
        0,
        5,
        "Исходный код и демо: https://github.com/ZaharKondaurov/ems_dereverb\n"
        "Веб-демо: https://huggingface.co/spaces/G1B-B0N/fspen_denoise_dereverb",
    )

    if pdf.page_no() > 3:
        raise RuntimeError(f"PDF exceeds 3 pages: {pdf.page_no()} pages")

    pdf.output(OUTPUT)
    return OUTPUT, pdf.page_no()


if __name__ == "__main__":
    path, pages = build_pdf()
    print(f"Created: {path} ({pages} pages)")
