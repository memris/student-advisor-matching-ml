# Интеллектуальная система подбора научных руководителей

Этот репозиторий содержит ядро machine learning движка для системы, предназначенной для автоматического подбора научных руководителей студентам на основе тем их дипломных работ. Проект был разработан как часть более крупного full-stack приложения.

## Постановка задачи

Процесс подбора научных руководителей в университете часто является ручным, трудоемким и субъективным. Цель этого проекта — автоматизировать этот процесс, используя методы обработки естественного языка (NLP) и машинного обучения для предоставления объективных, основанных на данных рекомендаций.

## Моя роль и вклад

Я отвечала за проектирование и реализацию всего machine learning пайплайна для этого проекта. Это включало:
- Предобработку данных и создание признаков (feature engineering) на основе исторических данных.
- Разработку основной рекомендательной модели.
- Создание решения в виде скрипта, готового к интеграции в веб-сервис.

## Технический подход и методология

Система функционирует как рекомендательная, предлагая ранжированный список руководителей для заданной темы студента. Ядром системы является модель **RandomForestClassifier**.

Методологию можно разделить на четыре ключевых этапа:

1.  **Предобработка данных:** Темы студентов и области экспертизы руководителей (полученные из их публикаций и интересов) очищаются, токенизируются и нормализуются. Этот этап включает удаление стоп-слов (с использованием кастомного словаря для предметной области) и применение лемматизации с помощью `pymorphy2`.

2.  **Создание признаков (Feature Engineering):** Основным признаком для модели является **косинусное сходство (Cosine Similarity)** между TF-IDF векторами темы студента и областей экспертизы руководителя. Это создает мощную метрику текстовой релевантности.

3.  **Обучение модели:** Вместо сложной многоклассовой задачи, я сформулировала ее как задачу **бинарной классификации**: "Является ли пара `(студент, руководитель)` удачной (1) или нет (0)?".
    - **Положительные примеры (метка 1)** были сгенерированы из исторических данных об успешных парах студент-руководитель.
    - **Отрицательные примеры (метка 0)** были созданы с помощью техники **negative sampling**, путем подбора студентам случайных руководителей, с которыми они не работали. Такой подход позволяет создать надежный обучающий набор данных.

4.  **Предсказание и ранжирование:** Когда поступает новая тема студента, система вычисляет признак (косинусное сходство) для каждого потенциального руководителя. Обученная модель RandomForest затем предсказывает *вероятность* успешного совпадения. Итоговые рекомендации ранжируются на основе этой вероятностной оценки.

## Скрипты в репозитории

В настоящее время репозиторий содержит основные Python-скрипты:

-   `main_script.py`: Основной исполняемый скрипт. Он принимает тему студента и путь к файлу с данными в качестве аргументов командной строки. Скрипт выполняет полный цикл загрузки данных, обучения, оценки и, наконец, выводит ранжированный список рекомендуемых руководителей в формате JSON.
-   `prepare_supervisors_data.py`: Вспомогательный скрипт, отвечающий за загрузку и подготовку исходных данных о руководителях из различных источников в Excel-файле.

# Intelligent System for Matching Students and Academic Advisors

This repository contains the core machine learning engine for a system designed to automatically recommend suitable academic advisors to students based on their thesis topics. This project was developed as part of a larger full-stack application.

## Problem Statement

The process of matching students with academic advisors in a university is often manual, time-consuming, and subjective. This project aims to automate this process by leveraging Natural Language Processing (NLP) and Machine Learning to provide objective, data-driven recommendations.

## My Role & Contribution

I was responsible for designing and implementing the entire machine learning pipeline for this project. This included:
- Data preprocessing and feature engineering from historical data.
- Developing the core recommendation model.
- Creating a script-based solution ready for integration into a web service.

## Technical Approach & Methodology

The system functions as a recommender, suggesting a ranked list of supervisors for a given student's topic. The core of the system is a **RandomForestClassifier** model.

The methodology can be broken down into four key stages:

1.  **Data Preprocessing:** Student topics and supervisor expertise areas (derived from their publications and interests) are cleaned, tokenized, and normalized. This involves removing stop-words (using a custom domain-specific dictionary) and applying lemmatization with `pymorphy2`.

2.  **Feature Engineering:** The primary feature for the model is the **Cosine Similarity** between the TF-IDF vectors of a student's topic and a supervisor's areas of expertise. This creates a powerful measure of textual relevance.

3.  **Model Training:** Instead of a complex multi-class problem, I framed this as a **binary classification task**: "Is this `(student, supervisor)` pair a good match (1) or not (0)?".
    - **Positive examples (label 1)** were generated from historical data of successful student-advisor pairings.
    - **Negative examples (label 0)** were generated using a **negative sampling** technique, pairing students with random supervisors they did not work with. This approach creates a robust training dataset.

4.  **Prediction & Ranking:** When a new student topic is provided, the system calculates the feature (cosine similarity) for every potential supervisor. The trained RandomForest model then predicts the *probability* of a successful match. The final recommendations are ranked based on this probability score.

## Scripts in this Repository

This repository currently contains the core Python scripts:

-   `main_script.py`: The main executable script. It takes a student's topic and a path to a data file as command-line arguments. It performs the full cycle of data loading, training, evaluation, and finally outputs a ranked list of recommended supervisors in JSON format.
-   `prepare_supervisors_data.py`: A helper script responsible for loading and preparing the initial supervisor data from various sources within the Excel file.

---
---
