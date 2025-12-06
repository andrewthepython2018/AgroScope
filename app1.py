import io
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
import streamlit as st
import matplotlib.pyplot as plt

# ================= НАСТРОЙКИ СТРАНИЦЫ =====================

st.set_page_config(
    page_title="AgroScope — анализ поля и чатбот",
    layout="wide"
)

st.title("AgroScope — демо анализа полей и ИИ-чатбот")


# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =====================

def load_demo_image() -> Image.Image:
    """
    Загружаем дефолтное изображение поля.
    Положи файл demo_field.jpg в ту же папку, где app.py.
    Если не найдётся — создадим зелёный прямоугольник-заглушку.
    """
    demo_path = Path("demo_field.jpg")
    if demo_path.exists():
        return Image.open(demo_path).convert("RGB")
    else:
        # Заглушка: просто зелёный "газон"
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[:, :, 1] = 180  # зелёный канал
        return Image.fromarray(img)


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    """PIL -> OpenCV (BGR)."""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
    """OpenCV (BGR или GRAY) -> PIL (RGB)."""
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def process_image_for_detection(cv_img: np.ndarray,
                                blur_ksize: int,
                                green_thr: int) -> np.ndarray:
    """
    Очень упрощённый "алгоритм распознавания":
    - блюр для сглаживания шума
    - выделяем "зелёные" пиксели по каналу G
    - всё, что ниже порога G, красим в красный как "проблемные зоны".
    """
    # Блюр
    if blur_ksize > 1:
        cv_img_blur = cv2.GaussianBlur(cv_img, (blur_ksize, blur_ksize), 0)
    else:
        cv_img_blur = cv_img.copy()

    # Разделяем каналы
    b, g, r = cv2.split(cv_img_blur)

    # Маска "проблемных зон" — там, где зелёный ниже порога
    problem_mask = g < green_thr

    # Создаём копию и подсвечиваем проблемные пиксели красным
    result = cv_img_blur.copy()
    result[problem_mask] = (0, 0, 255)  # BGR: красный

    return result


def create_heatmap(cv_img: np.ndarray,
                   blur_ksize: int,
                   green_thr: int) -> np.ndarray:
    """
    "Heatmap" по зелёному каналу:
    - берём зелёный канал
    - опционально блюрим
    - применяем цветовую карту
    - инвертируем, чтобы "хуже" выглядело краснее.
    """
    b, g, r = cv2.split(cv_img)

    if blur_ksize > 1:
        g_blur = cv2.GaussianBlur(g, (blur_ksize, blur_ksize), 0)
    else:
        g_blur = g

    # Нормируем к диапазону 0–255
    g_norm = cv2.normalize(g_blur, None, 0, 255, cv2.NORM_MINMAX)

    # Инверсия: меньше зелёного -> более "горячая" зона
    g_inv = 255 - g_norm

    heatmap = cv2.applyColorMap(g_inv, cv2.COLORMAP_JET)

    return heatmap


# ====== ПРОСТОЙ ЧАТБОТ БЕЗ ВНЕШНЕГО API (МОЖНО ПОТОМ ЗАМЕНИТЬ НА GPT) =======

PROJECT_INFO = """
Наш проект — система дистанционного мониторинга полей AgroScope.
Мы используем БПЛА и компьютерное зрение для анализа состояния растений,
поиска засушливых зон и проблемных участков. На основе данных формируем
рекомендации для агрономов.
"""

PROTOTYPES_INFO = """
У нас есть несколько прототипов:
1) БПЛА для аэрофотосъёмки полей с камерой высокого разрешения.
2) Прототип программного обеспечения для анализа снимков (индексы растительности,
   выделение проблемных зон, тепловые карты).
3) Веб-интерфейс/дашборд, где агроном может просматривать поля, отчёты и рекомендации.
"""

TEAM_INFO = """
Наша команда:
- Руководитель проекта / инженер по БПЛА.
- Специалист по компьютерному зрению и анализу данных.
- Разработчик интерфейса и интеграции (веб, Streamlit, API).
Команда объединяет опыт в области автоматизации, программирования и агротехнологий.
"""


def simple_bot_answer(message: str) -> str:
    """
    Очень простой rule-based бот, реагирующий на ключевые слова.
    Можно заменить на вызов OpenAI / другого LLM.
    """
    text = message.lower()

    if any(word in text for word in ["проект", "agroscope", "агроскоп", "что вы делаете", "чем занимаетесь"]):
        return PROJECT_INFO

    if any(word in text for word in ["прототип", "дрон", "uav", "бпла", "модель"]):
        return PROTOTYPES_INFO

    if any(word in text for word in ["команд", "кто вы", "участник", "разработчик"]):
        return TEAM_INFO

    if any(word in text for word in ["api", "интеграция", "встраивание", "wix", "стримлит", "streamlit"]):
        return (
            "Мы предоставляем API и веб-интерфейс для интеграции:\n"
            "- REST API для получения аналитики по полям;\n"
            "- веб-страницу на Streamlit, которую можно встроить в сайты (например, на Wix) через iframe."
        )

    # Ответ по умолчанию
    return (
        "Я могу рассказать о нашем проекте, прототипах и команде.\n"
        "Попробуйте спросить, например:\n"
        "- Расскажите о проекте\n"
        "- Какие у вас прототипы?\n"
        "- Кто входит в команду?\n"
        "- Как интегрировать это в сайт?"
    )


# ================== ЛЕЙАУТ СТРАНИЦЫ =======================

tab1, tab2 = st.tabs(["🛰 Анализ поля", "🤖 Чатбот о проекте"])

# ----------------- ТАБ 1: АНАЛИЗ ПОЛЯ --------------------

with tab1:
    st.subheader("Визуализация алгоритма обработки снимка поля")

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("### 1. Загрузка изображения")

        uploaded_file = st.file_uploader(
            "Загрузите снимок поля (jpg/png). Если не загрузить, будет использовано демо-изображение.",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
        else:
            image = load_demo_image()

        st.image(image, caption="Исходное изображение", use_container_width=True)

        st.markdown("### 2. Параметры обработки")

        blur_ksize = st.slider(
            "Сглаживание (размер ядра Gaussian Blur)",
            min_value=1, max_value=21, value=7, step=2,
            help="Чем больше ядро, тем сильнее сглаживание шума."
        )

        green_thr = st.slider(
            "Порог зелёного канала для 'проблемных зон'",
            min_value=0, max_value=255, value=100, step=5,
            help="Пиксели с зелёным значением ниже порога считаются потенциально проблемными."
        )

    with col_right:
        st.markdown("### 3. Результаты обработки")

        cv_img = pil_to_cv2(image)

        # "Распознавание" (подсвеченные проблемные зоны)
        detected_img = process_image_for_detection(cv_img, blur_ksize, green_thr)
        heatmap_img = create_heatmap(cv_img, blur_ksize, green_thr)

        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.markdown("**Распознанные проблемные зоны**")
            st.image(
                cv2_to_pil(detected_img),
                caption="Проблемные участки подсвечены красным",
                use_container_width=True
            )

        with col_res2:
            st.markdown("**Heatmap по состоянию поля**")
            st.image(
                cv2_to_pil(heatmap_img),
                caption="Псевдоцветовая карта (чем краснее — тем 'хуже')",
                use_container_width=True
            )

        # Немного статистики по кадру
        b, g, r = cv2.split(cv_img)
        mean_green = float(np.mean(g))
        st.markdown("### 4. Краткая статистика по изображению")
        st.write(f"Среднее значение зелёного канала: **{mean_green:.1f}**")
        st.write(f"Выбранный порог зелёного: **{green_thr}**")

# ----------------- ТАБ 2: ЧАТБОТ --------------------

with tab2:
    st.subheader("Чатбот о проекте, прототипах и команде")

    # Инициализируем историю сообщений
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Отображаем историю
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    # Поле ввода
    user_input = st.chat_input("Задайте вопрос о нашем проекте, прототипах или команде:")

    if user_input:
        # Добавляем сообщение пользователя
        st.session_state.chat_history.append(("user", user_input))

        # Получаем ответ бота
        bot_reply = simple_bot_answer(user_input)
        st.session_state.chat_history.append(("assistant", bot_reply))

        # Отображаем последние два сообщения
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
