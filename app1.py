import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from openai import OpenAI

# ================= НАСТРОЙКИ СТРАНИЦЫ =====================

st.set_page_config(
    page_title="AgroScope — анализ поля и ИИ-чатбот",
    layout="wide"
)

st.title("AgroScope — демо анализа поля и ИИ-чатбот")

# Более заметные вкладки и немного общего стиля
st.markdown("""
<style>
/* Общий фон вкладок */
div.stTabs [role="tablist"] {
    gap: 8px;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}

/* Кнопки вкладок */
div.stTabs [role="tab"] {
    padding: 0.45rem 1.3rem;
    border-radius: 999px;
    border: 1px solid rgba(0,0,0,0.1);
    background-color: #f5f5f7;
    font-weight: 600;
    font-size: 0.95rem;
}

/* Активная вкладка */
div.stTabs [aria-selected="true"] {
    background-color: #2563eb;  /* синий */
    color: white;
    border-color: #2563eb;
}

/* Лёгкая тень у картинок/блоков */
img {
    border-radius: 8px;
}

/* Для подписи шагов */
.agro-step-title {
    font-weight: 600;
    font-size: 1.05rem;
    margin-bottom: 0.25rem;
}
</style>
""", unsafe_allow_html=True)


# =============== ТЕКСТЫ ПРО ПРОЕКТ =================

PROJECT_INFO = """
**AgroScope** — система дистанционного мониторинга полей с помощью БПЛА и компьютерного зрения.

Мы:
- получаем снимки полей с дронов или камер;
- анализируем состояние растительности и инфраструктуры;
- строим карты и формируем рекомендации для агрономов.
"""

PROTOTYPES_INFO = """
Ключевые прототипы:

1. **БПЛА для аэрофотосъёмки**
   - автономные облёты полей;
   - съёмка с высокой детализацией.

2. **Модуль компьютерного зрения**
   - анализ снимков полей (индексы, heatmap, проблемные зоны);
   - детекция людей, техники и инфраструктуры на основе YOLOv3-tiny;
   - возможность адаптации под сельхоз-объекты.

3. **Веб-интерфейс / демо-панель**
   - визуализация снимков до/после обработки;
   - интерактивное управление параметрами;
   - интеграция с сайтами (например, через Streamlit + Wix).
"""

TEAM_INFO = """
Наша команда:

- инженер/разработчик БПЛА и систем автоматизации;
- специалист по компьютерному зрению и анализу данных;
- разработчик интерфейсов и интеграций (веб, Streamlit, API).

Мы совмещаем инженерный, программный и аграрный опыт.
"""

API_INFO = """
AgroScope можно интегрировать в другие системы:

- через REST API для аналитики по изображениям полей;
- через веб-интерфейс на Streamlit, который встраивается в сайты (например, Wix) через iframe;
- в перспективе — через интеграцию с ERP/агроплатформами.
"""


# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ИЗОБРАЖЕНИЯ) =====================

def load_demo_field() -> Image.Image:
    """Демо-изображение поля для анализа вегетации."""
    path = Path("demo_field.jpg")
    if path.exists():
        return Image.open(path).convert("RGB")
    else:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 1] = 180  # зелёная заглушка
        return Image.fromarray(img)

def load_demo_people() -> Image.Image:
    """Демо-изображение для детекции людей / объектов YOLO."""
    path = Path("demo_people.jpg")
    if path.exists():
        return Image.open(path).convert("RGB")
    else:
        # делаем простую заглушку с силуэтом человека
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "DEMO PEOPLE IMAGE", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
        return Image.fromarray(img)


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV (BGR)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """OpenCV (BGR/GRAY) -> PIL (RGB)."""
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


# =================== АНАЛИЗ ВЕГЕТАЦИИ (ExG + HEATMAP) =======================

def compute_exg_map(cv_img: np.ndarray) -> np.ndarray:
    """
    Индекс ExG (Excess Green):
      ExG = 2G – R – B, далее нормируем в 0–255.
    """
    b, g, r = cv2.split(cv_img.astype(np.float32))
    exg = 2 * g - r - b
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX)
    exg_uint8 = exg_norm.astype(np.uint8)
    return exg_uint8


def process_field_exg_detection(cv_img: np.ndarray,
                                blur_ksize: int,
                                exg_thr: int):
    """
    - сглаживаем изображение;
    - считаем карту ExG;
    - ниже порога ExG считаем «проблемной» растительностью;
    - делаем:
        * blended — исходник + подсветка проблемных зон;
        * exg_heatmap — heatmap по ExG;
        * problem_percent — доля проблемных пикселей.
    """
    if blur_ksize > 1:
        cv_img_blur = cv2.GaussianBlur(cv_img, (blur_ksize, blur_ksize), 0)
    else:
        cv_img_blur = cv_img.copy()

    exg = compute_exg_map(cv_img_blur)

    # heatmap по ExG
    exg_heat = cv2.applyColorMap(exg, cv2.COLORMAP_VIRIDIS)

    # проблемные зоны — ExG < порог
    problem_mask = exg < exg_thr
    problem_percent = float(problem_mask.mean() * 100.0)

    # подсветка проблемных пикселей
    overlay = cv_img_blur.copy()
    overlay[problem_mask] = (0, 0, 255)  # красный
    alpha = 0.5
    blended = cv2.addWeighted(cv_img_blur, 1 - alpha, overlay, alpha, 0)

    return blended, exg_heat, problem_percent


# =================== YOLOv3-TINY ЧЕРЕЗ OpenCV DNN (ДЕТЕКЦИЯ ЛЮДЕЙ/ОБЪЕКТОВ) =======================

MODEL_DIR = Path("models_yolo")
MODEL_DIR.mkdir(exist_ok=True)

CFG_PATH = MODEL_DIR / "yolov3-tiny.cfg"
WEIGHTS_PATH = MODEL_DIR / "yolov3-tiny.weights"
NAMES_PATH = MODEL_DIR / "coco.names"

URL_CFG = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
URL_WEIGHTS = "https://pjreddie.com/media/files/yolov3-tiny.weights"
URL_NAMES = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"


def download_if_not_exists(path: Path, url: str):
    if not path.exists():
        try:
            st.write(f"[INFO] Скачиваю {path.name}...")
            urllib.request.urlretrieve(url, str(path))
            st.write(f"[INFO] Файл {path.name} загружен.")
        except Exception as e:
            st.error(f"Не удалось скачать {path.name}: {e}")


def load_class_names(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def get_yolo_net():
    """
    Ленивая загрузка сети YOLOv3-tiny.
    """
    if "yolo_net" not in st.session_state:
        download_if_not_exists(CFG_PATH, URL_CFG)
        download_if_not_exists(WEIGHTS_PATH, URL_WEIGHTS)
        download_if_not_exists(NAMES_PATH, URL_NAMES)

        net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(WEIGHTS_PATH))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        st.session_state.yolo_net = net
        st.session_state.yolo_names = load_class_names(NAMES_PATH)

    return st.session_state.yolo_net, st.session_state.yolo_names


def run_yolov3_tiny_detection(cv_img: np.ndarray,
                              conf_thr: float = 0.35,
                              nms_thr: float = 0.4):
    """
    YOLOv3-tiny через OpenCV DNN:
      - рисуем боксы и подписи;
      - строим heatmap по плотности объектов.
    Возвращаем:
      - изображение с боксами,
      - heatmap-overlay,
      - количество объектов.
    """
    net, class_names = get_yolo_net()

    h, w = cv_img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv_img, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    ln = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(ln)

    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_thr:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width = int(detection[2] * w)
                height = int(detection[3] * h)

                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                boxes.append([x, y, width, height])
                confidences.append(confidence)
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thr, nms_thr)

    img_boxes = cv_img.copy()
    density = np.zeros((h, w), dtype=np.float32)
    num_objects = 0

    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y, w_box, h_box = boxes[i]
            x1, y1, x2, y2 = x, y, x + w_box, y + h_box

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            label = class_names[class_ids[i]] if 0 <= class_ids[i] < len(class_names) else str(class_ids[i])
            conf = confidences[i]
            num_objects += 1

            cv2.rectangle(img_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_boxes,
                f"{label} {conf:.2f}",
                (x1, max(y1 - 5, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

            density[y1:y2, x1:x2] += 1.0

    if density.max() > 0:
        density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        density_color = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)
        heat_overlay = cv2.addWeighted(cv_img, 0.4, density_color, 0.6, 0)
    else:
        heat_overlay = cv_img.copy()

    return img_boxes, heat_overlay, num_objects


# =================== ЧАТБОТ: GPT + FALLBACK =======================

SYSTEM_PROMPT = """
Ты — официальный цифровой консультант проекта AgroScope. 
Ты отвечаешь посетителям сайта, которые хотят понять, что делает система, какие комплексы уже созданы и как работает наше программное обеспечение.

ОЧЕНЬ ВАЖНО:
- Не упоминай, на чём технически реализован интерфейс (не говори про Streamlit, фреймворки, библиотеки, «веб-приложение» и т.п.).
- Не называй себя «чат-ботом», «ИИ-моделью», «LLM» и не описывай технические детали своего устройства.
- Говори так, как если бы ты просто был умным консультантом проекта, знакомым с его архитектурой и прототипами.

────────────────────────────────────────
1. КРАТКО О ПРОЕКТЕ AGROSCOPE
────────────────────────────────────────
AgroScope — это система дистанционного мониторинга сельскохозяйственных полей с помощью собственных беспилотных комплексов и модулей компьютерного зрения.

Ключевые идеи проекта:
- регулярный мониторинг полей с воздуха;
- анализ состояния растительности и выявление проблемных зон;
- выявление присутствия людей, техники и потенциально опасных ситуаций на поле;
- представление результатов в наглядном виде (карты, подсветка проблемных участков, простые числовые показатели).

Проект изначально ориентирован на реальные условия сельского хозяйства (например, орошаемые поля хлопка, пшеницы и других культур), где важно:
- быстро понимать, где растения в стрессе;
- вовремя обнаруживать проблемы с поливом и уходом;
- контролировать, что происходит на поле (люди, техника, техника безопасности).

Всегда старайся объяснять пользу AgroScope для:
- агрономов и фермеров;
- инженеров и студентов, которые интересуются БПЛА и компьютерным зрением;
- организаторов хакатонов, конкурсов и потенциальных партнёров.

────────────────────────────────────────
2. АВИАКОМПЛЕКСЫ AGROSCOPE (ВСЕГО ЧЕТЫРЕ)
────────────────────────────────────────
Обязательно помни: у нас четыре основных комплекса — два самолётных и два квадрокоптера. 
Описывай их обобщённо, но технически правдоподобно, без придумывания конкретных марок, производителей и характеристик.

2.1. Самолётный комплекс №1 — «дальний разведчик»
- Тип: БПЛА с фиксированным крылом.
- Назначение:
  • покрывать большие площади за один вылет;
  • строить обзорные снимки/карты полей.
- Полезная нагрузка:
  • камера высокого разрешения, ориентированная на съёмку с большой высоты.
- Особенности:
  • подходит для первичного картирования и общей диагностики;
  • удобен для регулярного мониторинга состояния больших полей.

2.2. Самолётный комплекс №2 — «скоростной инспектор»
- Тип: более компактный и манёвренный самолётный БПЛА.
- Назначение:
  • быстрые повторные облёты уже известных проблемных зон;
  • оперативная актуализация данных (перед поливом, внесением удобрений и т.п.).
- Полезная нагрузка:
  • камера с хорошей стабилизацией для получения чётких кадров.
- Особенности:
  • ориентирован на скорость и гибкость;
  • используется, когда нужно быстро получить свежие данные по ключевым участкам.

2.3. Квадрокоптер №1 — «вертикальная инспекция»
- Тип: мультикоптер с высокой точностью зависания.
- Назначение:
  • детальный анализ конкретных проблемных зон, выявленных самолётами;
  • съёмка с малых высот для детального рассмотрения растений и поверхности поля.
- Полезная нагрузка:
  • камера, при необходимости — возможность установки специализированной оптики.
- Особенности:
  • может “зависать” над интересующей зоной;
  • подходит для работы на ограниченных участках (например, десятки метров).

2.4. Квадрокоптер №2 — «универсальный съёмочный дрон»
- Тип: лёгкий, удобный мультикоптер для повседневной съёмки.
- Назначение:
  • регулярные осмотры полей;
  • фото и видео для документации, обучения, демонстраций.
- Полезная нагрузка:
  • стабилизированная камера.
- Особенности:
  • быстрый запуск;
  • удобен для демонстраций, учебных задач и “живых” примеров работы AgroScope.

Если пользователя интересуют технические параметры (высота, скорость, время полёта и т.п.), отвечай обобщённо и аккуратно:
- «конкретные характеристики зависят от конфигурации, но концептуально этот комплекс рассчитан на …»
Не придумывай точных чисел и брендов, если пользователь сам их не дал.

────────────────────────────────────────
3. ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ AGROSCOPE
────────────────────────────────────────
ПО в AgroScope состоит из нескольких логических модулей. 
Опирайся на них, когда объясняешь “как работает система”.

3.1. Модуль анализа растительности (индекс ExG)
- Основная идея:
  • оценивать “зелёность” растительности по цифровому изображению;
  • выделять участки, где растения выглядят угнетёнными или разреженными.
- Используемый индекс:
  • ExG (Excess Green, “избыточная зелень”):
    – вычисляется по каналам RGB;
    – даёт числовую оценку, насколько сильно в изображении выражен зелёный канал относительно красного и синего.
- Типичный алгоритм:
  • исходное изображение (снимок поля);
  • сглаживание (например, фильтрация для уменьшения шума);
  • вычисление карты ExG по каждому пикселю;
  • нормализация значений;
  • выделение пикселей с низким ExG как “проблемных” (например, засуха, редкие всходы, стрессы);
  • построение карты (heatmap), где по цветам видно распределение индекса.
- Результаты, которые можно объяснять пользователю:
  • карта проблемных зон (там, где индекс ExG ниже порога);
  • процент площади поля, которая попадает в “зону риска”;
  • общее впечатление о состоянии растительности: «большинство участков зелёные» / «заметные зоны стресса» и т.п.

3.2. Модуль детекции людей и объектов (YOLOv3-tiny)
- Основная идея:
  • автоматически находить на изображении людей, машины, технику и другие объекты;
  • визуально показывать, где они находятся;
  • оценивать “плотность” присутствия объектов в разных частях кадра.
- Используемая модель:
  • YOLOv3-tiny (облегчённая версия модели детекции объектов);
  • работает через модуль компьютерного зрения, который:
    – получает на вход изображение;
    – возвращает координаты объектов, классы и уверенность.
- Типичный алгоритм:
  • входное изображение преобразуется в формат для модели;
  • модель выдаёт набор “кандидатов” объектов;
  • применяется порог по уверенности и подавление пересечений (NMS);
  • вокруг найденных объектов рисуются рамки и подписи класса;
  • по накопленной “плотности” объектов создаётся heatmap:
    – чем “горячее” (краснее) зона, тем больше объектов там обнаружено.
- Практический смысл:
  • контроль присутствия людей на поле (например, для безопасности);
  • анализ перемещения техники;
  • возможность расширить подход под специальные классы (например, определённые типы сельхозтехники).

3.3. Модуль визуализации
- Показывает:
  • исходное изображение;
  • обработанный результат (подсвеченные зоны, heatmap и т.п.);
  • сопутствующую статистику (процент проблемных зон, количество объектов и т.д.).
- Позволяет интерактивно менять параметры:
  • порог ExG (что считать “проблемной” растительностью);
  • порог уверенности для детекции объектов;
  • степень сглаживания изображений.
- Задача этого модуля — сделать результаты анализа понятными не только инженеру, но и агроному или члену жюри хакатона.

3.4. Концепция API (внешний доступ)
- Проект предполагает возможность:
  • выдавать аналитическую информацию о полях через набор эндпоинтов (например, “получить индексы по полю”, “получить карту проблемных зон”, “получить историю состояния поля”);
  • интеграции с другими системами управления хозяйством.
- Когда пользователь спрашивает про API:
  • рассказывай концептуально (REST-эндпоинты, JSON-ответы со сводной информацией);
  • не придумывай конкретные URL, схемы авторизации и базы данных, если их явно не задавали.

────────────────────────────────────────
4. КАК ОТВЕЧАТЬ НА ТИПИЧНЫЕ ВОПРОСЫ
────────────────────────────────────────
4.1. О проекте в целом
- Объясни:
  • зачем нужен мониторинг полей с воздуха;
  • почему важно видеть именно пространственное распределение проблем;
  • чем полезно сочетание БПЛА и компьютерного зрения;
  • какие задачи можно решать (засуха, неравномерные всходы, контроль полива, контроль присутствия людей и техники).

4.2. О конкретных комплексах
- Структурируй ответ:
  • короткий обзор всех четырёх;
  • при необходимости — подробнее о том, который интересует пользователя;
  • подчёркивай, что самолёты хороши для больших площадей, а квадрокоптеры — для точечной инспекции.

4.3. О программном обеспечении
- Всегда привязывай ответ к двум основным направлениям:
  • анализ растительности (индексы, heatmap, “красные” зоны и проценты проблемных участков);
  • детекция людей и объектов (рамки, классы, количество, тепловая карта плотности).
- Объясняй на уровне:
  • «мы берём снимок, обрабатываем, выделяем статистику и показываем результат в удобном виде».

4.4. О применении для фермеров и агрономов
- Подчёркивай практический эффект:
  • экономия времени инспекторов;
  • более точечное внесение удобрений и полива;
  • более раннее обнаружение проблем;
  • возможность документировать состояние поля во времени (история снимков и анализов).

4.5. О планах развития
- Если спрашивают о том, чего пока нет (например, интеграция со спутниками, точные прогнозные модели урожайности):
  • отвечай честно: «на текущей демонстрации это не реализовано»;
  • можно добавлять: «такую функцию можно развить на основе текущих модулей, например, добавив…» — и коротко объяснить идею, без обещаний и конкретных сроков.

────────────────────────────────────────
5. СТИЛЬ И ОГРАНИЧЕНИЯ
────────────────────────────────────────
- Всегда отвечай на русском языке.
- Пиши структурировано: абзацы, при необходимости списки.
- Предпочитай инженерно-прикладной стиль: спокойно, без пафоса, с акцентом на смысл.
- Не придумывай:
  • названия партнёров, грантов, конкурсов, которых пользователь не называл;
  • точные технические числа (время полёта, дальность, мегапиксели), если их явно не дали;
  • коммерческие детали (цены, контракты, ROI) без исходных данных.
- Если информации недостаточно, говори об этом прямо и давай аккуратные, помеченные как предположения, варианты.

────────────────────────────────────────
6. ГЛАВНАЯ ЦЕЛЬ
────────────────────────────────────────
После общения с тобой пользователь должен чётко понимать:
- что делает система AgroScope;
- какие у неё есть четыре беспилотных комплекса и чем они друг от друга отличаются;
- как с помощью анализа растительности и детекции объектов можно получать ценную информацию о поле;
- зачем всё это нужно в реальном сельском хозяйстве и образовательных проектах.

Отвечай так, как если бы представлял проект на важном хакатоне или перед потенциальными партнёрами: спокойно, компетентно и без лишних технических подробностей “под капотом”.
"""


def simple_bot_answer(message: str) -> str:
    """Резервный бот, если нет ключа или ошибка Open."""
    text = message.lower()

    if any(w in text for w in ["проект", "agroscope", "агроскоп", "что вы делаете", "чем занимаетесь"]):
        return PROJECT_INFO

    if any(w in text for w in ["прототип", "прототипы", "дрон", "бпла", "uav", "алгоритм", "модели"]):
        return PROTOTYPES_INFO

    if any(w in text for w in ["команд", "кто вы", "участник", "разработчик"]):
        return TEAM_INFO

    if any(w in text for w in ["api", "апи", "интеграция", "wix", "streamlit", "стримлит"]):
        return API_INFO

    return (
        "Я могу рассказать о проекте AgroScope, наших прототипах, команде и вариантах интеграции.\n\n"
        "Попробуйте спросить, например:\n"
        "- Расскажите подробнее о проекте AgroScope\n"
        "- Какие прототипы у вас есть?\n"
        "- Кто входит в вашу команду?\n"
        "- Как встроить это в сайт или использовать API?"
    )


def ai_bot_answer() -> str:
    """
    GPT-чатбот с памятью.
    Берёт историю диалога из st.session_state.chat_history.
    Если ключа нет или ошибка — simple_bot_answer по последнему вопросу.
    """
    # 1) Пытаемся взять ключ из переменной окружения
    api_key = os.getenv("OPENAI_API_KEY")

    # 2) Если на Streamlit Cloud — ключ обычно лежит в st.secrets
    if not api_key and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]

    # Находим последнее сообщение пользователя
    last_user_msg = ""
    for msg in reversed(st.session_state.chat_history):
        if msg["role"] == "user":
            last_user_msg = msg["content"]
            break

    # Если ключа нет вообще — уходим в fallback-бот
    if not api_key:
        return simple_bot_answer(last_user_msg)

    try:
        client = OpenAI(api_key=api_key)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(st.session_state.chat_history)

        completion = client.chat.completions.create(
            model="gpt-4o-mini",   # можешь сменить модель при желании
            messages=messages,
            temperature=0.4,
        )

        reply = completion.choices[0].message.content
        return reply
    except Exception as e:
        # Можно временно вывести ошибку, чтобы понять, что не так
        st.error(f"Ошибка при обращении к OpenAI: {e}")
        return simple_bot_answer(last_user_msg)



# ================== ЛЕЙАУТ СТРАНИЦЫ =======================

tab1, tab2 = st.tabs(["🛰 Анализ изображения", "🤖 Чатбот о проекте"])


# ----------------- ТАБ 1: АНАЛИЗ ИЗОБРАЖЕНИЯ --------------------

with tab1:
    st.markdown("## Анализ изображения поля и сцены")

    # ─────────────────────────
    # Шаг 1–2: режим, загрузка, параметры
    # ─────────────────────────
    top_container = st.container()
    with top_container:
        col_left, col_right = st.columns([1.1, 1])

        with col_left:
            st.markdown('<div class="agro-step-title">Шаг 1. Выберите режим анализа</div>', unsafe_allow_html=True)

            mode = st.radio(
                "",
                ["Анализ вегетации (ExG + heatmap)", "Детекция людей/объектов (YOLOv3-tiny)"],
                horizontal=False,
            )

            st.markdown('<div class="agro-step-title" style="margin-top:0.75rem;">Шаг 2. Загрузите изображение</div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Можно использовать снимок поля, дрона, сцены с людьми/техникой. Если файл не выбран — будет использовано демо-изображение.",
                type=["jpg", "jpeg", "png"],
            )

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
            else:
                if mode.startswith("Анализ вегетации"):
                    image = load_demo_field()
                else:
                    image = load_demo_people()

            st.markdown('<div class="agro-step-title" style="margin-top:0.75rem;">Настройка параметров</div>', unsafe_allow_html=True)

            blur_ksize = st.slider(
                "Сглаживание (Gaussian Blur, нечётный размер ядра)",
                min_value=1,
                max_value=21,
                value=7,
                step=2,
            )

            if mode.startswith("Анализ вегетации"):
                exg_thr = st.slider(
                    "Порог индекса ExG для 'проблемных' зон",
                    min_value=0,
                    max_value=255,
                    value=110,
                    step=5,
                    help="Пиксели с ExG ниже этого значения считаются потенциально проблемными (засуха, слабая растительность и т.п.).",
                )
                conf_thr = None  # для совместимости
            else:
                conf_thr = st.slider(
                    "Порог уверенности YOLOv3-tiny",
                    min_value=0.1,
                    max_value=0.9,
                    value=0.35,
                    step=0.05,
                    help="Чем выше порог, тем меньше, но точнее детекции.",
                )
                exg_thr = None

        with col_right:
            st.markdown('<div class="agro-step-title">Предпросмотр исходного изображения</div>', unsafe_allow_html=True)
            st.image(image, caption="Исходный кадр для анализа", use_container_width=True)

    st.markdown("---")

    # ─────────────────────────
    # Шаг 3: результаты анализа (два ровных окна)
    # ─────────────────────────
    st.markdown('<div class="agro-step-title">Шаг 3. Результаты анализа</div>', unsafe_allow_html=True)

    cv_img = pil_to_cv2(image)
    col_res1, col_res2 = st.columns(2)

    if mode.startswith("Анализ вегетации"):
        field_result, field_heatmap, problem_percent = process_field_exg_detection(
            cv_img, blur_ksize, exg_thr
        )

        with col_res1:
            st.markdown("**Проблемные зоны по индексу ExG**")
            st.image(
                cv2_to_pil(field_result),
                caption="Проблемные участки подсвечены красным (низкий индекс зелёной растительности)",
                use_container_width=True,
            )

        with col_res2:
            st.markdown("**Heatmap по индексу ExG**")
            st.image(
                cv2_to_pil(field_heatmap),
                caption="Псевдоцветовая карта ExG: по распределению видно, где растительность в стрессе",
                use_container_width=True,
            )

        st.markdown("### Краткая статистика по полю")
        st.write(f"Доля проблемных пикселей по ExG: **{problem_percent:.1f} %**")
        st.write(
            "Чем выше этот процент, тем больше участков с пониженным индексом зелени. "
            "Это может указывать на засуху, редкие всходы или стресс растений."
        )

    else:
        boxes_img, heatmap_img, num_objects = run_yolov3_tiny_detection(cv_img, conf_thr)

        with col_res1:
            st.markdown("**Детекция людей и объектов (YOLOv3-tiny)**")
            st.image(
                cv2_to_pil(boxes_img),
                caption="Найденные объекты с рамками и подписями класса (люди, транспорт, техника и т.п.)",
                use_container_width=True,
            )

        with col_res2:
            st.markdown("**Heatmap по плотности объектов**")
            st.image(
                cv2_to_pil(heatmap_img),
                caption="Чем 'горячее' зона, тем больше объектов там обнаружено моделью",
                use_container_width=True,
            )

        st.markdown("### Краткая статистика по сцене")
        st.write(f"Общее количество детектированных объектов: **{num_objects}**")
        st.write(
            "Это демонстрирует, что на основе тех же подходов можно отслеживать присутствие людей, техники "
            "и других объектов на поле, дополняя анализ вегетации."
        )



# ----------------- ТАБ 2: ЧАТБОТ --------------------

with tab2:
    st.subheader("Чатбот о проекте AgroScope, комплексах и ПО")

    # Инициализация состояния для контекста
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = ""

    st.markdown("### Задайте вопрос")

    # Плейсхолдер, куда будем выводить либо загрузку, либо ответ
    answer_placeholder = st.empty()

    # Форма с вводом вопроса ВСЕГДА СВЕРХУ
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_area(
            "Ваш вопрос о проекте, наших комплексах или программном обеспечении:",
            value="",
            height=80,
            placeholder="Например: Опишите подробно все четыре комплекса AgroScope и как работает ПО."
        )
        submitted = st.form_submit_button("Отправить")

    # Обработка отправки
    if submitted and question.strip():
        # Добавляем вопрос в историю для контекста LLM
        st.session_state.chat_history.append({"role": "user", "content": question})

        # При новом запросе сразу затираем старый ответ спиннером
        with answer_placeholder:
            with st.spinner("Готовим ответ..."):
                answer = ai_bot_answer()  # использует chat_history и SYSTEM_PROMPT

        # Сохраняем ответ в историю и состояние
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.last_answer = answer

    # Если уже был ответ — показываем его под полем ввода
    if st.session_state.last_answer and not (submitted and not question.strip()):
        with answer_placeholder:
            st.markdown("### Ответ ассистента")
            st.markdown(st.session_state.last_answer)



