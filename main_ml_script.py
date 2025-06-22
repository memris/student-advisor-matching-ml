import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import re
import sys
import json

morph_analyzer = None 
try:
    import pymorphy2
    morph_analyzer = pymorphy2.MorphAnalyzer()
    print("ML Script: pymorphy2 успешно импортирован и инициализирован.", file=sys.stderr)
except Exception as e: 
    print(f"ML Script Warning: Не удалось инициализировать pymorphy2 (ошибка: {e}). Лемматизация не будет выполнена.", file=sys.stderr)
    morph_analyzer = None

from prepare_supervisors_data import load_and_prepare_supervisor_data 

def preprocess_text(text, lemmatizer=morph_analyzer):
    if not isinstance(text, str):
        return ""
    
    text_lower = text.lower()
    text_cleaned = re.sub(r'[^\w\s-]', '', text_lower)
    text_cleaned = re.sub(r'\d+', '', text_cleaned)
    
    stop_words = {
        "и", "в", "с", "на", "по", "о", "к", "из", "от", "до", "об", "за", "под", "над", "при", "без", "через",
        "для", "или", "но", "а", "то", "так", "как", "бы", "же", "только", "еще", "уже", "вот", "там", "тут",
        "не", "ни", "тоже", "также", "что", "кто", "где", "когда", "почему", "зачем", "какой", "который",
        "мой", "твой", "свой", "наш", "ваш", "их", "его", "ее", "он", "она", "оно", "они", "я", "ты", "мы", "вы",
        "себя", "этот", "тот", "другой", "каждый", "весь", "сам", "самый",
        "быть", "являться", "стать", "делать", "сделать", "иметь", "мочь", "хотеть", "сказать", "говорить",
        "идти", "дать", "взять", "новый", "старый", "хороший", "плохой", "большой", "маленький",
        "один", "два", "три", "несколько", "много", "мало", "очень", "поэтому", "потому", "затем", "далее",
        "основной", "главный", "некоторый", "определенный", "различный", "следующий", "данный",
        "разработка", "проектирование", "исследование", "анализ", "системы", "система", "методы", "метод", "подходы", "подход",
        "приложения", "приложение", "программы", "программа", "платформе", "платформа", "средствами", "средство", "технологии", "технология",
        "компании", "компания", "предприятия", "предприятие", "учреждения", "учреждение", "организации", "организация", "отдела", "отдел",
        "использование", "внедрение", "совершенствование", "оптимизация", "управление", "поддержка", "реализация", "повышение", "улучшение",
        "автоматизация", "создание", "моделирование", "обеспечения", "обеспечение", "учета", "учет", "обработка",
        "информационной", "информационный", "информационных", "бизнес",
        "данных", "данные", "процессов", "процесс", "деятельности", "деятельность", "работы", "работа", "проекта", "проект",
        "цель", "задачи", "задача", "основе", "сфере", "рамках", "случая", "случае", "примере", "вопроса", "вопросы", "аспект", "аспекты", "вкр",
        "ооо", "зао", "пао", "мэи", "барс", "т.д", "т.п", "др", "например", "целях", "помощью", "путем", "пути",
        "современных", "актуальность", "значимость", "решение", "проблемы", "проблема", "возможности", "возможность",
        "условиях", "требования", "требований", "особенностей", "особенности", "функционирования", "построение",
        "на примере", "с использованием", "в условиях", "для компании", "на предприятии", "в сфере", "с целью",
        "студента", "руководителя", "научного", "тема", "дипломной", "квалификационной", "выпускной"
    }

    words = text_cleaned.split()
    processed_words = []
    
    if lemmatizer:
        try:
            for word_token in words:
                if not word_token: continue
                parsed_word = lemmatizer.parse(word_token)[0]
                normal_form = parsed_word.normal_form
                if normal_form not in stop_words and len(normal_form) > 2: # > 2 для общего случая
                     processed_words.append(normal_form)
        except Exception as e_lemma:
            print(f"ML Script Warning (preprocess_text): Ошибка при лемматизации: {e_lemma}. Используется обработка без лемматизации.", file=sys.stderr)
            processed_words = [word for word in words if word not in stop_words and len(word) > 2]
    else: 
        processed_words = [word for word in words if word not in stop_words and len(word) > 2]
            
    return " ".join(processed_words).strip()


if __name__ == '__main__':
    print("Python script started", file=sys.stderr)
    
    if len(sys.argv) < 3:
        error_message = {"error": "Недостаточно аргументов. Ожидается: <путь_к_excel_файлу> <тема_студента>."}
        print(json.dumps(error_message), file=sys.stderr)
        print(json.dumps([])) 
        sys.exit(1)

    MAIN_EXCEL_FILE_PATH = sys.argv[1]
    student_topic_from_arg = sys.argv[2]
    print(f"ML Script: Используется Excel файл: {MAIN_EXCEL_FILE_PATH}", file=sys.stderr)
    print(f"ML Script: Тема студента для анализа: '{student_topic_from_arg}'", file=sys.stderr)

    SHEET_SUPERVISORS_DATA = 'Преподаватели'
    SHEET_NAMES_STUDENTS_THESES = ['Декабрь 2023', 'Февраль 2024 (наши + ИДДО)', 'Июнь 2024 (наши + ИДДО)']
    MAIN_TRAINING_SHEET_NAME = 'Декабрь 2023'

    print("ML Script: Загрузка данных...", file=sys.stderr)
    list_of_theses_dfs = []
    for sn in SHEET_NAMES_STUDENTS_THESES:
        try: 
            df_s = pd.read_excel(MAIN_EXCEL_FILE_PATH, sheet_name=sn, header=0, engine='openpyxl')
            if not df_s.empty: list_of_theses_dfs.append(df_s)
            else: print(f"  ML Script Warning: Лист '{sn}' пуст.", file=sys.stderr)
        except Exception as e: print(f"  ML Script Warning: Не удалось загрузить лист {sn}: {e}", file=sys.stderr)
    
    s_exp_loaded, df_s_loaded = load_and_prepare_supervisor_data(MAIN_EXCEL_FILE_PATH, SHEET_SUPERVISORS_DATA, list_of_theses_dfs)
    if df_s_loaded.empty: 
        print(json.dumps({"error":"Критические данные о руководителях не загружены"}), file=sys.stderr); print(json.dumps([])); sys.exit(1)
    
    try: 
        df_s_hist = pd.read_excel(MAIN_EXCEL_FILE_PATH, sheet_name=MAIN_TRAINING_SHEET_NAME, header=0, engine='openpyxl')
        if df_s_hist.empty: raise ValueError(f"Лист '{MAIN_TRAINING_SHEET_NAME}' для обучения пуст.")
    except Exception as e: 
        print(json.dumps({"error":f"Не удалось загрузить основной лист для обучения '{MAIN_TRAINING_SHEET_NAME}': {e}"}), file=sys.stderr); print(json.dumps([])); sys.exit(1)
    
    supervisors_expertise_loaded, df_supervisors_loaded, df_students_history = s_exp_loaded, df_s_loaded, df_s_hist
    print(f"ML Script: Загружено {len(df_supervisors_loaded)} рук., история {len(df_students_history)} студ.", file=sys.stderr)

    print("\nML Script: Предобработка текста и векторизация...", file=sys.stderr)
    theme_col_student = None
    possible_theme_cols = ['Тема (окончательную формулировку см. в БАРСе)', 'Тематика ВКР', 'Тема ВКР', 'Тема']
    for col_cand in possible_theme_cols:
        if col_cand in df_students_history.columns and not df_students_history[col_cand].isnull().all():
            theme_col_student = col_cand
            break
    if not theme_col_student: 
        print(json.dumps({"error":"Колонка темы студента не найдена в обучающей выборке"}), file=sys.stderr); print(json.dumps([])); sys.exit(1)

    df_students_history['processed_topic_student'] = df_students_history[theme_col_student].apply(
        lambda x: preprocess_text(x, lemmatizer=morph_analyzer)
    )

    supervisor_tags_col_for_tfidf = 'Теги_строкой'
    if supervisor_tags_col_for_tfidf not in df_supervisors_loaded.columns or df_supervisors_loaded[supervisor_tags_col_for_tfidf].isnull().all():
        print(f"ML Script Warning: Колонка '{supervisor_tags_col_for_tfidf}' не найдена или пуста в данных руководителей. Попытка создать из 'Теги'.", file=sys.stderr)
        if 'Теги' in df_supervisors_loaded.columns:
            df_supervisors_loaded[supervisor_tags_col_for_tfidf] = df_supervisors_loaded['Теги'].apply(
                lambda tags_list: preprocess_text(" ".join(tags_list if isinstance(tags_list, list) else []), lemmatizer=morph_analyzer)
            )
        else:
            print(json.dumps({"error": "Невозможно создать 'Теги_строкой', так как колонка 'Теги' отсутствует."}), file=sys.stderr); print(json.dumps([])); sys.exit(1)

    all_texts_corpus = list(df_students_history['processed_topic_student'].dropna()) + \
                       list(df_supervisors_loaded[supervisor_tags_col_for_tfidf].dropna())
    all_texts_corpus = [text for text in all_texts_corpus if text]
    if not all_texts_corpus: 
        print(json.dumps({"error":"Корпус текстов для TF-IDF пуст."}), file=sys.stderr); print(json.dumps([])); sys.exit(1)

    tfidf_vectorizer = TfidfVectorizer(max_features=1500, min_df=2, ngram_range=(1,2))
    tfidf_vectorizer.fit(all_texts_corpus)

    student_topic_vectors = tfidf_vectorizer.transform(df_students_history['processed_topic_student'])
    supervisor_tag_vectors = tfidf_vectorizer.transform(df_supervisors_loaded[supervisor_tags_col_for_tfidf])
    supervisor_vectors_map = {name: vec for name, vec in zip(df_supervisors_loaded['Руководитель'], supervisor_tag_vectors)}

    print("ML Script: Формирование датасета для обучения...", file=sys.stderr)
    features_list = []
    labels_list = []

    if 'Руководитель' not in df_students_history.columns: 
        print(json.dumps({"error":"Колонка 'Руководитель' отсутствует в обучающей выборке"}), file=sys.stderr); print(json.dumps([])); sys.exit(1)

    for index, row_student in df_students_history.iterrows():
        student_vec = student_topic_vectors[index]
        supervisor_name_actual = row_student['Руководитель']
        if pd.isna(supervisor_name_actual) or not str(supervisor_name_actual).strip(): continue

        if supervisor_name_actual in supervisor_vectors_map:
            supervisor_vec_actual = supervisor_vectors_map[supervisor_name_actual]
            similarity = cosine_similarity(student_vec, supervisor_vec_actual)[0][0]
            
            # ФОРМИРОВАНИЕ ПРИЗНАКОВ
            current_features = [similarity]
            features_list.append(current_features)
            labels_list.append(1)

    num_negative_per_positive = 2 
    all_supervisor_names_loaded = df_supervisors_loaded['Руководитель'].dropna().unique().tolist() 
    for index, row_student in df_students_history.iterrows():
        student_vec = student_topic_vectors[index]; actual_supervisor = row_student['Руководитель']
        if pd.isna(actual_supervisor) or not str(actual_supervisor).strip(): continue
        count_neg = 0; attempts_neg = 0; max_attempts_neg = len(all_supervisor_names_loaded) * 3
        while count_neg < num_negative_per_positive and attempts_neg < max_attempts_neg :
            attempts_neg += 1; 
            if not all_supervisor_names_loaded: break 
            random_supervisor_name = np.random.choice(all_supervisor_names_loaded)
            if random_supervisor_name != actual_supervisor and random_supervisor_name in supervisor_vectors_map:
                supervisor_vec_random = supervisor_vectors_map[random_supervisor_name]
                similarity = cosine_similarity(student_vec, supervisor_vec_random)[0][0]
                
                # ФОРМИРОВАНИЕ ПРИЗНАКОВ ДЛЯ НЕГАТИВНЫХ ПРИМЕРОВ
                current_features_neg = [similarity]
                features_list.append(current_features_neg)
                labels_list.append(0)
                count_neg += 1
    
    if not features_list: 
        print(json.dumps({"error":"Список признаков для обучения пуст."}), file=sys.stderr); print(json.dumps([])); sys.exit(1)

    X = np.array(features_list)
    y = np.array(labels_list)

    if X.shape[0] == 0 or X.shape[1] == 0: 
        print(json.dumps({"error":f"Массив признаков X пуст или имеет неверную форму: {X.shape}."}), file=sys.stderr); print(json.dumps([])); sys.exit(1)

    print("ML Script: Обучение модели RandomForestClassifier...", file=sys.stderr)
    model = None 
    if len(np.unique(y)) < 2 or X.shape[0] < 10 : 
        print("ML Script Warning: Недостаточно данных или классов для обучения/тестирования модели.", file=sys.stderr)
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            if len(X_train) < 5 or len(X_test) < 5 or len(np.unique(y_train)) < 2 :
                print("ML Script Warning: После разделения выборки стали слишком малы. Модель не будет обучена.", file=sys.stderr)
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample', n_jobs=-1, max_depth=10, min_samples_leaf=5)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                accuracy_sklearn = model.score(X_test, y_test)

                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()

                print(f"ML Script: Debug - Матрица ошибок (TN, FP, FN, TP): ({tn}, {fp}, {fn}, {tp})", file=sys.stderr)

                # Вычисление метрик по формулам
                total_population = tp + tn + fp + fn
                # Знаменатель для Precision: все предсказанные как позитивные (TP + FP)
                predicted_positives_for_precision = tp + fp
                # Знаменатель для Recall: все реально позитивные (TP + FN)
                actual_positives_for_recall = tp + fn
                
                # Accuracy = (TP+TN)/(TP+TN+FP+FN)
                accuracy_formula = (tp + tn) / total_population if total_population > 0 else 0.0
                
                # Precision = TP / (TP + FP)
                precision_formula = tp / predicted_positives_for_precision if predicted_positives_for_precision > 0 else 0.0
                
                # Recall = TP / (TP + FN)
                recall_formula = tp / actual_positives_for_recall if actual_positives_for_recall > 0 else 0.0
                
                # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
                if (precision_formula + recall_formula) > 0:
                    f1_score_formula = (2 * precision_formula * recall_formula) / (precision_formula + recall_formula)
                else:
                    f1_score_formula = 0.0

                print("\nML Script: Критерии оценки качества (рассчитанные по формулам, класс 1 - позитивный):", file=sys.stderr)
                print(f"  Accuracy = (TP+TN)/(TP+TN+FP+FN) = ({tp}+{tn})/({tp}+{tn}+{fp}+{fn}) = {accuracy_formula:.4f}", file=sys.stderr)
                print(f"  Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision_formula:.4f}", file=sys.stderr)
                print(f"  Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall_formula:.4f}", file=sys.stderr)
                print(f"  F1-score = 2 * (Precision * Recall) / (Precision + Recall) = {f1_score_formula:.4f}", file=sys.stderr)
                
                print(f"\nML Script: Точность модели (Accuracy) на тестовой выборке (из model.score): {accuracy_sklearn:.3f}", file=sys.stderr)
                
                target_names_report = ['Класс 0 (неверный рук.)', 'Класс 1 (верный рук.)']
                report_str = classification_report(y_test, y_pred, target_names=target_names_report, zero_division=0)
                print(f"ML Script: Отчет по классификации (sklearn.metrics):\n{report_str}", file=sys.stderr)

        except Exception as e:
            print(f"ML Script Error: Ошибка во время обучения или разделения данных: {e}", file=sys.stderr)

    def get_recommendations(student_topic_text_arg, supervisors_df_full_arg, vectorizer_fitted_arg, ml_model_trained_arg, supervisor_vecs_map_fitted_arg):
        if not ml_model_trained_arg: 
            print("ML Script Warning: Модель не обучена, рекомендации невозможны.", file=sys.stderr)
            return []
            
        processed_student_topic = preprocess_text(student_topic_text_arg, lemmatizer=morph_analyzer)
        if not processed_student_topic: 
            print("ML Script Warning: Тема студента пуста после предобработки.", file=sys.stderr)
            return []
            
        try: 
            student_vec_pred = vectorizer_fitted_arg.transform([processed_student_topic])
        except Exception as e: 
            print(f"ML Script Error: Ошибка при векторизации темы студента для предсказания: {e}", file=sys.stderr)
            return []

        recommendations = []
        for idx, sup_row in supervisors_df_full_arg.iterrows():
            supervisor_name = sup_row['Руководитель']
            if pd.isna(supervisor_name) or not str(supervisor_name).strip(): continue
            if supervisor_name not in supervisor_vecs_map_fitted_arg: continue 

            supervisor_vec_pred = supervisor_vecs_map_fitted_arg[supervisor_name]
            similarity_pred = cosine_similarity(student_vec_pred, supervisor_vec_pred)[0][0]
            
            features_for_prediction_np = np.array([similarity_pred]).reshape(1, -1)

            try:
                probability = ml_model_trained_arg.predict_proba(features_for_prediction_np)[0][1] 
                recommendations.append({
                    'supervisor': supervisor_name, 
                    'score_model': float(probability), 
                    'score_similarity_tfidf': float(similarity_pred),
                    'tags': sup_row.get('Теги', []) 
                })
            except Exception as e_pred:
                print(f"ML Script Warning: Ошибка при предсказании для {supervisor_name}: {e_pred}", file=sys.stderr)
        
        recommendations.sort(key=lambda x: x['score_model'], reverse=True)
        return recommendations

    output_recommendations = [] 
    if model:
        print(f"\nML Script: Формирование рекомендаций для темы: '{student_topic_from_arg}'", file=sys.stderr)
        output_recommendations = get_recommendations(
            student_topic_from_arg, 
            df_supervisors_loaded,       
            tfidf_vectorizer,            
            model,                       
            supervisor_vectors_map       
        )
        print(f"ML Script: Сформировано {len(output_recommendations)} рекомендаций.", file=sys.stderr)
    else:
        print("ML Script: Модель не была обучена, рекомендации не будут сформированы.", file=sys.stderr)

    try:
        print(json.dumps(output_recommendations, ensure_ascii=False, indent=None))
    except Exception as e_json_dump:
        print(json.dumps({"error": f"Ошибка при формировании JSON для вывода: {e_json_dump}"}), file=sys.stderr)
        print(json.dumps([]))
        sys.exit(1)

    print("ML Script: Работа завершена (JSON выведен в stdout).", file=sys.stderr)