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

def load_demo_image() -> Image.Image:
    """
    Загружаем дефолтное изображение поля.
    Если файла demo_field.jpg нет — создаём зелёную заглушку.
    """
    demo_path = Path("demo_field.jpg")
    if demo_path.exists():
        return Image.open(demo_path).convert("RGB")
    else:
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 1] = 180
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
Ты — официальный ИИ-ассистент проекта AgroScope, встроенный в демо-панель на сайте.
Отвечай чётко, профессионально и строго по фактам проекта. Ничего не выдумывай.

ТВОЯ ОСНОВНАЯ ЗАДАЧА:
Понятно и точно объяснять, что делает система AgroScope, какие прототипы созданы,
как работает программное обеспечение и демо-страница анализа изображений.

────────────────────────────────────────────────────────
ОПИСАНИЕ ПРОЕКТА AGROSCOPE
────────────────────────────────────────────────────────

AgroScope — это многоуровневая система дистанционного мониторинга сельскохозяйственных полей,
объединяющая несколько собственных БПЛА-комплексов и программный модуль анализа снимков.

Главная цель — **оперативное выявление проблем на полях**: засухи, нехватки азота, неравномерных всходов,
участков стрессовой растительности, присутствия людей или техники на полях, ошибок при поливе.

AgroScope использует:
• беспилотные самолётные платформы для больших площадей;  
• квадрокоптеры для локального обследования и детального обследования участков;  
• программный модуль для обработки изображений и визуализации состояния поля;  
• веб-интерфейс (Streamlit), встроенный в сайт через iframe;  
• возможность API-подключения к внешним платформам.

────────────────────────────────────────────────────────
НАШИ АВТОРСКИЕ БПЛА (ВСЕГО 4 КОМПЛЕКСА)
────────────────────────────────────────────────────────

1) Самолётный комплекс №1 — **дальний разведчик**
   • Тип: фиксированное крыло  
   • Назначение: полёты над большими полями (квадратные километры за один вылет).  
   • Полезная нагрузка: камера высокого разрешения; возможность менять оптику.  
   • Особенности:
       – высокая длительность полёта;
       – автоматические миссии по заранее заданному маршруту;
       – идеально подходит для массового картирования полей (ортофотопланы).

2) Самолётный комплекс №2 — **скоростной инспектор**
   • Тип: компактное крыло для быстрых облетов.  
   • Назначение: экспресс-мониторинг, повторные облёты одного и того же участка, работа при ветре.  
   • Полезная нагрузка: камера + модуль стабилизации.  
   • Особенности:
       – повышенная скорость;
       – подходит для быстрой актуализации данных перед внесением удобрений/поливом.

3) Квадрокоптер №1 — **вертикальная инспекция**
   • Тип: мультикоптер, высокая точность зависания.  
   • Назначение: локальное обследование проблемных зон, обнаруженных самолётами.  
   • Полезная нагрузка: камера; возможна установка мультиспектральной оптики.  
   • Особенности:
       – вертикальная посадка;
       – высокая детализация снимков;
       – удобен для исследований участков 20×20 м и меньше.

4) Квадрокоптер №2 — **универсальный съемочный дрон**
   • Тип: лёгкий мультикоптер.  
   • Назначение: повседневные осмотры, фото и видео растительности, контроль рабочей силы на поле.  
   • Полезная нагрузка: стабилизированная камера.  
   • Особенности:
       – быстрый запуск;
       – подходит для студентов-агрономов и инженеров;
       – используется в демонстрационных миссиях.

────────────────────────────────────────────────────────
ПРОГРАММНОЕ ОБЕСПЕЧЕНИЕ AGROSCOPE
────────────────────────────────────────────────────────

ПО AgroScope состоит из нескольких модулей:

1) Модуль анализа растительности (индексы)
   • вычисление индекса ExG (Excess Green) = метрика зелёности;  
   • выделение “проблемных зон” с низким ExG;  
   • heatmap распределения индекса по полю;  
   • сглаживание изображения, фильтрация шумов;  
   • определение процента стрессовой растительности.

2) Модуль детекции людей и объектов
   • модель YOLOv3-tiny с использованием cv2.dnn;  
   • обнаружение людей, техники, объектов инфраструктуры;  
   • построение тепловой карты плотности объектов;  
   • подсчёт количества найденных объектов.

3) Модуль визуализации
   • два параллельных окна: “до/после”, heatmap, результат анализа;  
   • гибко настраиваемые ползунки: blur, порог ExG, confidence для YOLO;  
   • вывод статистики и кратких заключений.

4) Веб-интерфейс (Streamlit)
   • встроен на сайт через iframe (например, Wix);  
   • полностью автономно работает в браузере;  
   • демонстрирует работу алгоритмов анализа полей;  
   • доступен чатбот, отвечающий на вопросы о проекте.

5) Модуль ИИ-ассистента (чатбот)
   • отвечает на вопросы про проект, комплексы, функциональность ПО;  
   • не выдумывает технические детали, которых нет в проекте;  
   • объясняет суть анализа растительности и YOLO-детекции;  
   • помогает пользователю понять ценность AgroScope.

────────────────────────────────────────────────────────
СТИЛЬ ОТВЕТОВ
────────────────────────────────────────────────────────
• Всегда отвечай на русском.  
• Будь технически точным, но понятным для обычного пользователя.  
• Избегай художественных фантазий — только факты проекта.  
• Если пользователь задаёт вопрос вне темы, сначала коротко ответь,
  затем мягко верни разговор к AgroScope.  

────────────────────────────────────────────────────────
ГЛАВНАЯ МИССИЯ
────────────────────────────────────────────────────────
Объяснить пользователю:
• что делает AgroScope;  
• как работают наши 4 БПЛА;  
• что делает модуль анализа растительности;  
• что делает модуль детекции людей/объектов;  
• как всё это объединено в одном ПО и веб-интерфейсе.

Твои ответы — лицо проекта. 
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
    st.subheader("Анализ вегетации и детекция людей/объектов")

    # Выбор режима
    mode = st.radio(
        "Режим анализа:",
        ["Анализ вегетации (ExG + heatmap)", "Детекция людей/объектов (YOLOv3-tiny)"],
        horizontal=True,
    )

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### 1. Загрузка изображения")

        uploaded_file = st.file_uploader(
            "Загрузите снимок поля / сцены. "
            "Если не загрузить — используется демо-изображение поля.",
            type=["jpg", "jpeg", "png"],
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            image = load_demo_image()

        st.image(image, caption="Исходное изображение", use_container_width=True)

        st.markdown("### 2. Параметры обработки")

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
        else:
            conf_thr = st.slider(
                "Порог уверенности YOLOv3-tiny",
                min_value=0.1,
                max_value=0.9,
                value=0.35,
                step=0.05,
                help="Чем выше порог, тем меньше, но точнее детекции.",
            )

    with col_right:
        st.markdown("### 3. Результаты обработки")

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
                    caption="Проблемные участки подсвечены красным (на основе индекса ExG)",
                    use_container_width=True,
                )

            with col_res2:
                st.markdown("**Heatmap по индексу ExG**")
                st.image(
                    cv2_to_pil(field_heatmap),
                    caption="Псевдоцветовая карта ExG (состояние растительности)",
                    use_container_width=True,
                )

            st.markdown("### 4. Краткая статистика по полю")
            st.write(f"Доля проблемных пикселей по ExG: **{problem_percent:.1f} %**")
            st.write(
                "Чем выше этот процент, тем больше участков с пониженным индексом зелёной растительности."
            )

        else:
            boxes_img, heatmap_img, num_objects = run_yolov3_tiny_detection(cv_img, conf_thr)

            with col_res1:
                st.markdown("**Детекция объектов (YOLOv3-tiny)**")
                st.image(
                    cv2_to_pil(boxes_img),
                    caption="Объекты с рамками и подписями класса",
                    use_container_width=True,
                )

            with col_res2:
                st.markdown("**Heatmap по плотности объектов**")
                st.image(
                    cv2_to_pil(heatmap_img),
                    caption="Чем 'горячее' зона, тем больше объектов обнаружено",
                    use_container_width=True,
                )

            st.markdown("### 4. Краткая статистика по сцене")
            st.write(f"Общее количество детектированных объектов: **{num_objects}**")
            st.write(
                "Это демонстрирует возможности компьютерного зрения: отслеживать людей, технику и объекты на поле.\n"
                "Те же подходы можно адаптировать под специализированные сельхоз-задачи."
            )


# ----------------- ТАБ 2: ЧАТБОТ --------------------

with tab2:
    st.subheader("Чатбот о проекте AgroScope, прототипах и команде")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # показываем историю
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # поле ввода внизу
    user_input = st.chat_input("Задайте вопрос о проекте, прототипах или интеграции:")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        bot_reply = ai_bot_answer()
        st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

        with st.chat_message("assistant"):
            st.markdown(bot_reply)
