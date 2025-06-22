import pandas as pd
import re
from collections import Counter
import sys

# Вспомогательные функции
def extract_quota(note_text):
    if pd.isna(note_text): return None 
    note_text_lower = str(note_text).lower()
    if "уволен" in note_text_lower or "декрет" in note_text_lower: return 0
    match = re.search(r'(?:до\s*)?(\d+)\s*(?:чел(?:овек)?|магистр(?:ов)?|дипломник(?:ов)?|студент(?:ов)?)', note_text_lower)
    if match: return int(match.group(1))
    return None

def preprocess_text_for_tags_no_lemma(text):
    if pd.isna(text) or not str(text).strip(): return []
    text_lower = str(text).lower()
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
    words = [word for word in text_cleaned.split() if word not in stop_words and len(word) > 3]
    return words

# --- Основная функция ---
def load_and_prepare_supervisor_data(
    excel_file_path, 
    sheet_name_supervisors,
    list_of_student_theses_dfs=None
):
    try:
        df_supervisors_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name_supervisors, header=0, engine='openpyxl')
        expected_cols_for_supervisors = ['ФИО', 'Должн.', 'Ставка', 'Примечание', 'Направление']
        for col in expected_cols_for_supervisors:
            if col not in df_supervisors_raw.columns:
                print(f"Prepare Supervisors Warning: Ожидаемая колонка '{col}' не найдена на листе '{sheet_name_supervisors}'. Она будет создана пустой.", file=sys.stderr)
                df_supervisors_raw[col] = pd.NA
    except Exception as e:
        print(f"Prepare Supervisors Error: ошибка при чтении данных руководителей из '{sheet_name_supervisors}' файла '{excel_file_path}': {e}", file=sys.stderr)
        return {}, pd.DataFrame(columns=['Руководитель', 'Должность', 'Ставка', 'Теги', 'Квота', 'Примечание_исходное', 'Теги_строкой'])

    auto_extracted_tags_by_supervisor = {}
    if list_of_student_theses_dfs is not None and isinstance(list_of_student_theses_dfs, list):
        print(f"Prepare Supervisors Info: Извлечение тегов из {len(list_of_student_theses_dfs)} источников тем ВКР студентов...", file=sys.stderr)
        all_theme_words_per_supervisor_across_sources = {}

        for i, df_theses in enumerate(list_of_student_theses_dfs):
            if df_theses is None or df_theses.empty:
                print(f"Prepare Supervisors Warning: Источник тем ВКР №{i+1} пуст или не предоставлен, пропускаем.", file=sys.stderr)
                continue
            
            print(f"  Prepare Supervisors Info: Обработка источника тем ВКР №{i+1}...", file=sys.stderr)
            theme_col_name = None
            possible_theme_cols = ['Тема (окончательную формулировку см. в БАРСе)', 'Тематика ВКР', 'Тема ВКР', 'Тема']
            for col_candidate in possible_theme_cols:
                if col_candidate in df_theses.columns and not df_theses[col_candidate].isnull().all():
                    theme_col_name = col_candidate
                    break
            supervisor_col_name = 'Руководитель'
            if supervisor_col_name in df_theses.columns and theme_col_name:
                for supervisor_fio_from_theses, group in df_theses.groupby(supervisor_col_name):
                    if pd.isna(supervisor_fio_from_theses): continue
                    supervisor_fio_key = str(supervisor_fio_from_theses).strip()
                    if supervisor_fio_key not in all_theme_words_per_supervisor_across_sources:
                        all_theme_words_per_supervisor_across_sources[supervisor_fio_key] = []
                    for theme_text in group[theme_col_name]:
                        all_theme_words_per_supervisor_across_sources[supervisor_fio_key].extend(
                            preprocess_text_for_tags_no_lemma(theme_text)
                        )
            else:
                print(f"Prepare Supervisors Warning: В источнике тем ВКР №{i+1} отсутствуют колонки '{supervisor_col_name}' или подходящая для '{possible_theme_cols}'. Пропускаем.", file=sys.stderr)
        
        for supervisor_fio, all_words in all_theme_words_per_supervisor_across_sources.items():
            if all_words:
                word_counts = Counter(all_words)
                min_occurrences_auto = 1 
                top_n_auto = 10
                potential_tags = [word for word, count in word_counts.items() if count >= min_occurrences_auto]
                if len(potential_tags) > top_n_auto:
                     extracted_tags = [word for word, count in word_counts.most_common(top_n_auto)]
                elif potential_tags:
                     extracted_tags = potential_tags
                else:
                     extracted_tags = [word for word, count in word_counts.most_common(top_n_auto)]
                if extracted_tags:
                    auto_extracted_tags_by_supervisor[supervisor_fio] = extracted_tags
    
    manual_tags_updates = {
        'Е А.В.': ['бизнес-процессов', 'оптимизация'],
        'Ж О.В.': ['информационные системы', 'разработка информационных', 'базы данных', 'баз данных'],
        'Б А.П.': ['информационные системы', 'разработка информационных', 'базы данных', 'онлайн'],
        'Б К.С.': ['ии', 'искусственный интеллект', 'машинного обучения'],
        'Б Ю.С.': ['ии', 'искусственный интеллект', 'машинного обучения'],
        'Т О.Л.': ['crm-системы', 'бизнес-процессов', 'импортозамещения']
    }

    tag_expansion_rules = {
        "ии": ["машинного обучения", "нейронные сети", "глубокого обучения"],
        "искусственный интеллект": ["машинного обучения", "нейронные сети", "глубокого обучения"],
        "машинного обучения": ["нейронные сети", "глубокого обучения", "анализ данных"], 
        "веб": ["веб-приложения", "веб-сервисы", "frontend", "backend", "javascript", "html", "css"],
        "базы данных": ["sql", "nosql", "проектирование баз", "субд"],
        "бизнес-процессов": ["моделирование бизнес-процессов", "оптимизация бизнес-процессов", "анализ бизнес-процессов", "bpm", "erp", "crm"]
    }

    supervisors_expertise_dict = {}
    supervisors_data_list = []
    fio_col_in_df = 'ФИО'

    for index, row in df_supervisors_raw.iterrows():
        fio_original = row.get(fio_col_in_df)
        if pd.isna(fio_original) or not str(fio_original).strip(): continue
        fio = str(fio_original).strip()

        current_tags = []
        direction_text = row.get('Направление')
        if pd.notna(direction_text) and str(direction_text).strip():
            processed_direction_words = preprocess_text_for_tags_no_lemma(str(direction_text))
            current_tags.extend(processed_direction_words)

        for manual_fio_key, manual_tag_list_to_add in manual_tags_updates.items():
            if manual_fio_key.lower() in fio.lower(): 
                current_tags.extend([tag.lower() for tag in manual_tag_list_to_add])
        
        if fio in auto_extracted_tags_by_supervisor:
            current_tags.extend(auto_extracted_tags_by_supervisor[fio])
        else: 
            fio_parts_supervisor = fio.split()
            if len(fio_parts_supervisor) >= 2: 
                short_fio_variants = [f"{fio_parts_supervisor[0]} {fio_parts_supervisor[1][0]}."]
                if len(fio_parts_supervisor) >= 3:
                    short_fio_variants.append(f"{fio_parts_supervisor[0]} {fio_parts_supervisor[1][0]}.{fio_parts_supervisor[2][0]}.")
                for auto_fio_key, auto_tags_list in auto_extracted_tags_by_supervisor.items():
                    auto_fio_key_lower = auto_fio_key.lower()
                    found_match = False
                    for variant in short_fio_variants:
                        if variant.lower() in auto_fio_key_lower:
                            current_tags.extend(auto_tags_list); found_match = True; break
                    if found_match: break
        
        expanded_tags = list(current_tags) 
        for tag in current_tags:
            for rule_key, rule_values in tag_expansion_rules.items():
                if rule_key in tag: 
                    expanded_tags.extend(rule_values)
        
        unique_tags = sorted(list(set(filter(None, expanded_tags)))) 
        supervisors_expertise_dict[fio] = unique_tags
        
        quota = extract_quota(row.get('Примечание'))
        default_quota = 3
        is_active_supervisor = "уволен" not in str(row.get('Примечание','')).lower() and \
                               "декрет" not in str(row.get('Примечание','')).lower()
        
        if quota is None:
            quota = default_quota if is_active_supervisor else 0
            
        supervisors_data_list.append({
            'Руководитель': fio,
            'Должность': row.get('Должность'), 
            'Ставка': row.get('Ставка'),       
            'Теги': unique_tags,
            'Квота': quota,
            'Примечание_исходное': row.get('Примечание')
        })

    df_supervisors_final = pd.DataFrame(supervisors_data_list)
    if not df_supervisors_final.empty:
        df_supervisors_final = df_supervisors_final.drop_duplicates(subset=['Руководитель'], keep='first')
        df_supervisors_final['Теги_строкой'] = df_supervisors_final['Теги'].apply(
            lambda x: ' '.join([str(tag).replace(' ', '_') for tag in x if str(tag).strip()])
        )
    else:
        print("Prepare Supervisors Warning: Финальный DataFrame руководителей пуст.", file=sys.stderr)
        df_supervisors_final = pd.DataFrame(columns=['Руководитель', 'Должность', 'Ставка', 'Теги', 'Квота', 'Примечание_исходное', 'Теги_строкой'])
    
    return supervisors_expertise_dict, df_supervisors_final


if __name__ == '__main__':
    TEST_EXCEL_FILE_PATH = '23-24.xlsx'
    TEST_SHEET_NAME_SUPERVISORS = 'Преподаватели'
    TEST_SHEET_NAMES_STUDENT_THESES_LIST = ['Декабрь 2023', 'Февраль 2024 (наши + ИДДО)', 'Июнь 2024 (наши + ИДДО)']

    print_err = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

    print_err(f"Запуск тестирования функции load_and_prepare_supervisor_data...")
    print_err(f"Чтение руководителей из файла: {TEST_EXCEL_FILE_PATH}, лист: {TEST_SHEET_NAME_SUPERVISORS}")
    print_err(f"Чтение тем ВКР из листов: {TEST_SHEET_NAMES_STUDENT_THESES_LIST}")

    list_of_theses_dfs_for_test = []
    for sheet_name in TEST_SHEET_NAMES_STUDENT_THESES_LIST:
        try:
            df_single_theses_source = pd.read_excel(TEST_EXCEL_FILE_PATH, sheet_name=sheet_name, header=0, engine='openpyxl')
            list_of_theses_dfs_for_test.append(df_single_theses_source)
            print_err(f"  Успешно загружен лист тем: {sheet_name}, строк: {len(df_single_theses_source)}")
        except Exception as e:
            print_err(f"  Предупреждение: Не удалось загрузить лист тем '{sheet_name}': {e}")
            list_of_theses_dfs_for_test.append(None)

    expertise_dict, df_results = load_and_prepare_supervisor_data(
        TEST_EXCEL_FILE_PATH, 
        TEST_SHEET_NAME_SUPERVISORS,
        list_of_student_theses_dfs=list_of_theses_dfs_for_test
    )

    if not df_results.empty:
        print_err("\n--- Словарь supervisors_expertise (примеры с тегами) ---")
        count_printed = 0
        for k, v in expertise_dict.items():
            if v: 
                print_err(f"'{k}': {v}")
                count_printed+=1
            if count_printed >= 10 : break 
        if not any(expertise_dict.values()): print_err("Словарь компетенций пуст или у всех руководителей пустые списки тегов.")

        print_err("\n--- DataFrame df_supervisors (примеры с непустыми тегами) ---")
        df_with_tags = df_results[df_results['Теги'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
        print_err(df_with_tags[['Руководитель', 'Теги', 'Квота', 'Теги_строкой']].head(10).to_string()) # to_string для лучшего вывода DataFrame
        
        print_err(f"\nВсего обработано уникальных руководителей: {len(df_results)}")
        print_err(f"Руководителей с извлеченными тегами: {len(df_with_tags)}")
    else:
        print_err("\nТестирование завершено: не удалось загрузить или обработать данные о руководителях.")