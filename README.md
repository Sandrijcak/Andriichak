# UAV Fortification Detection System

**Метод виявлення фортифікаційних споруд на знімках з БПЛА засобами глибокого навчання**

Репозиторій програмного забезпечення до кваліфікаційної роботи **Олександра Андрійчак** (Хмельницький національний університет, 2025)

---

## 1. Огляд проєкту

У сучасних умовах оперативного аналізу аерофотознімків із безпілотних літальних апаратів критично важливо автоматизувати виявлення фортифікаційних споруд. Метою цього проєкту є створення швидкої та точної моделі на базі YOLOv8, здатної в реальному часі локалізувати окопи, траншеї та бліндажі на великих знімках місцевості. Розроблений метод донавчено на спеціалізованому датасеті та забезпечує баланс між продуктивністю й точністю.

## 2. Архітектура системи

Програмна реалізація побудована за принципами модульного програмування, що гарантує гнучкість, масштабованість і легкість обслуговування. Основні компоненти:

1. **app.py** – центральний модуль-орchestrator, що відповідає за ініціалізацію системи, передачу зображень на обробку і об’єднання результатів.
2. **Streamlit UI** – веб-інтерфейс для завантаження знімків та відображення анотацій.
3. **YOLOv8 Module** – завантажує попередньо натреновані ваги, виконує інференс і повертає координати та класи виявлених об’єктів.
4. **Results Processor** – формує фінальне анотоване зображення та структурує метадані (bounding boxes, confidence).
5. **Data Handlers (Pillow, NumPy)** – обробка і трансформація зображень у тензори для подальшої передачі в модель.
   Програмна реалізація спроєктована за принципами модульного програмування, що забезпечує:

* **Гнучкість**: легке додавання нових функцій та моделей;
* **Масштабованість**: підтримка обробки як одиничних знімків, так і потокових даних;
* **Супровід**: чітке розділення коду на компоненти з низькою залежністю.

### Основні компоненти:

1. **Головний застосунок (app.py)**

   * Оркеструє процес: ініціалізація, передача на обробку, збирання результатів.
2. **Інтерфейс користувача (Streamlit)**

   * Забезпечує зручний веб-інтерфейс для завантаження зображень та візуалізації результатів.
3. **Модуль нейронної мережі (YOLOv8)**

   * Завантажує ваги моделі, виконує інференс, повертає bounding boxes.
4. **Модуль обробки результатів**

   * Інкапсулює дані про виявлені об’єкти, формує анотоване зображення.
5. **Модулі роботи з даними (Pillow, NumPy)**

   * Обробляють графічні файли, конвертацію у тензори, масштабування, нормалізацію.

## 3. Встановлення та запуск

Оптимальним середовищем для роботи є **Anaconda**. Нижче наведено кроки для налаштування:

```bash
# 1. Створити та активувати віртуальне середовище
conda create --name webapp python=3.10
conda activate webapp

# 2. Перейти у директорію з проєктом
cd шлях/до/папки/

# 3. Встановити залежності
pip install -r requirements.txt

# 4. Запустити веб-застосунок
streamlit run app.py
```

> **Примітка:** після першого налаштування достатньо повторювати лише активацію середовища та запуск `streamlit run app.py`.

## 4. Використання застосунку

1. **Завантаження знімка**: через кнопку **Browse files** оберіть аерофотознімок з БПЛА.
2. **Автоматичний аналіз**: завантажене зображення відразу обробляється YOLOv8.
3. **Результат**: на екрані відображається анотоване зображення з прямокутниками навколо фортифікаційних споруд та підписами.



## 5. Особливості реалізації

* **Локальна обробка**: повністю автономний режим без звернень до зовнішніх сервісів.
* **Модульність**: кожен компонент протестовано окремо, що спрощує розширення.
* **Продуктивність**: модель оптимізована для обробки знімків високої роздільності з кадровою частотою до 10 FPS.
* **Конфіденційність**: всі дані зберігаються локально, що критично важливо для військових застосувань.

## 6. Приклади використання

```bash
conda activate webapp
cd шлях/до/папки/UAV 
streamlit run app.py
```

1. **Крок 1**: Завантажте зображення місцевості з БПЛА.
2. **Крок 2**: Система відобразить анотації фортифікаційних об’єктів.

---

**UAV Fortification Detection System** є результатом дослідження і реалізації сучасних методів глибокого навчання для задачі виявлення оборонних споруд, що може бути адаптовано під різні сценарії розвідувальних та моніторингових операцій.
