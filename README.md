# SpotifySongAnalisis - программа по анализу песен из Spotify

Данный репозиторий является курсовым проектом по дисциплине [Наука о данных и аналитика больших объемов информации](https://github.com/Tka4uk-Andrei/semesters_description/blob/master/semester_1.md#наука-о-данных-и-аналитика-больших-объемов-информации) в рамках программы магистратуры `09.04.04 Программная инженерия` (профиль `09.04.04_02 Основы анализа и разработки приложений с большими объемами распределенных данных`) в [Санкт-Петербургском политехническом университете](https://www.spbstu.ru).

Над кодом для данного проекта работали:
- Илья Буров([@IlyaBurov](https://github.com/Ilya-Burov))
- Усов Михаил([@UsovMichail](https://github.com/mishokU))
---
## Инструкции по настройке окружения

1. При работе использовать `python 3.9`
2. Скачать и распаковать [репозиторий](https://github.com/mishokU/SpotifySongAnalisis)
3Установить зависимости выполнив
   ```bash
   pip install -r requirements.txt
   ```

---

## Получение `Client id`, `Client secret' для Spotify API

1. Залогиниться в spotify for developers на этой [странице](https://developer.spotify.com/dashboard/).
2. Создать новый проект по этой [ссылке](https://developer.spotify.com/dashboard/applications) указываем название и описание проекта
3. Во вкладке входа в наше приложение видим `Client id` ,`Client secret'

---

## Инструкции по работе с проектом

2. Перед запуском скриптов в файле `auth.py` задайте нужные значения для следующих переменных:
    - `client_id` - строка, содержащая `client_id` для взаимодействия с Spotify API;
    - `client_secret` - секретный ключ для Spotify API конкретного проекта;
3. Также стоит заменить путь к файлам dataPath = '/Users/m.usov/PycharmProjects/SpotifySongAnalisis/data/' на ваш
4. Для запуска приложения требуется выполнить следующую комманду:
  ```bash
    python3 code/program.py
  ```