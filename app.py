import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import re

from eda import (open_data,
                 table_na_duble,
                 data_processing,
                 corr_matrix,
                 show_continuous_grafics,
                 pie_categorical_grafics,
                 show_categorical_grafics,
                 target_grafics)


def process_main_page():

    '''
    :return: None
    '''
    # show_main_page()
    datasets, df_dict, datasets_names = open_data()
    # сайдбары для настроек отображения графиков
    st.sidebar.markdown("*Настройки отображения графиков числовых признаков*")

    width_cont = st.sidebar.slider("plot cont width", 1, 10, 5)
    height_cont = st.sidebar.slider("plot cont height", 1, 10, 2)

    st.sidebar.markdown("---")

    st.sidebar.markdown("*Настройки отображения графиков круговых диаграмм*")

    width_pie = st.sidebar.slider("plot pie width", 1, 10, 5)
    height_pie = st.sidebar.slider("plot pie height", 1, 10, 4)

    st.sidebar.markdown("---")

    st.sidebar.markdown("*Настройки отображения графиков категориальных признаков*")

    width_cat = st.sidebar.slider("plot cat width", 1, 10, 6)
    height_cat = st.sidebar.slider("plot cat height", 1, 10, 2)


    # выводим информацию об изначальных датасетах
    st.markdown("**Краткая информация об исследуемых датасетах**")

    df_origin_table = table_na_duble(datasets)
    st.table(df_origin_table)

    if st.button('Раскрыть расширенную информацию о каждом датасете'):
        for i in range(len(datasets)):
            name = re.sub(r'D_', '', datasets_names[i].split('.')[0])
            st.markdown(f'Датасет: {name}\n')
            df_ = df_dict[name].sample(2) if len(df_dict[name]) <= 2 else df_dict[name].sample(3)
            st.table(df_)
            st.markdown(f'Уникальные значения по каждому столбцу')
            st.table(datasets[i].nunique())
            st.text(f'\nРазмер датасета: {len(df_dict[name])}')
            str = ('Есть пропуски!') if df_dict[name].isna().any().any() else ('Пропусков нет')
            st.text(str)
            st.markdown("___")

    st.markdown("___")

    # собираем данные в единый датасет, формируем отчёты об числовых и категориальных признаках
    all_data, continuous_features, categorical_features, cat_feats = data_processing(df_dict)

    st.markdown("# Объединённый и очищенный датасет")
    st.dataframe(all_data.sample(3))

    st.text(('Есть пропуски!') if all_data.isna().any().any() else ('Пропусков нет'))
    st.text(('Есть дубли!') if all_data.duplicated().any() else ('Дубликатов нет'))
    st.text(f'Размер датасета {all_data.shape}')

    st.markdown("## Описание числовых характеристик")
    st.table(all_data[continuous_features].describe())

    st.markdown("## Описание категориальных характеристик")
    st.table(all_data.describe(include='category'))

    st.markdown("___")
    # отрисовываем матрицу корреляции
    st.pyplot(corr_matrix(all_data), use_container_width=True)
    st.write("Линейной корреляции таргета с признаками не наблюдается")
    st.write("Из корреляций между признаками можно отметить: закрытые кредиты и количество кредитов, "
             "cумма первого платежа и суммой последнего кредита, возраст и количеством рабочих лет"
             )

    st.markdown("___")

    # раздел для формирования графики
    st.markdown("# Графики")
    st.markdown("## Графики числовых признаков")
    st.write('По условию требовалось построить графики распределений и зависимостей целевой переменной и признаков.\n'
             'Решил отобразить всё на одном графике. Также решил не ограничивать количество графиков.\n'
             'Но для отрисовки графика необходимо выбрать признак. А затем нажать кнопку: отобразить график.')

    st.write('*Размер графика можно поменять в панели слева, но потребуется заново отрисовать график.*')

    st.markdown("**Выберем график какой числовой переменной хотим посмотреть**")
    cont_feat = st.selectbox(
        'Выберете числовую переменную',
        ('CREDIT',
         'FST_PAYMENT',
         'AGE',
         'CHILD_TOTAL',
         'DEPENDANTS',
         'OWN_AUTO',
         'PERSONAL_INCOME',
         'WORK_TIME_IN_YEARS',
         'CLOSED_LOANS',
         'LOAN_AMOUNT'))

    if cont_feat == 'CREDIT':
        analysis_cont = 'У нас не нормальное распределение. Явного влияния на таргет нет.'
    elif cont_feat == 'FST_PAYMENT':
        analysis_cont = 'У нас не нормальное распределение. Явного влияния на таргет нет. 1 значение сильно выделяется,' \
                        'при этом оно реалистичное'
    elif cont_feat == 'AGE':
        analysis_cont = 'У нас не нормальное распределение. Явного влияния на таргет нет.'
    elif cont_feat == 'CHILD_TOTAL':
        analysis_cont = 'Явного влияния на таргет нет. Людей с большим количеством детей представлено очень мало.'
    elif cont_feat == 'DEPENDANTS':
        analysis_cont = 'Явного влияния на таргет нет. Людей с большим количеством иждивенцев представлено очень мало.'
    elif cont_feat == 'OWN_AUTO':
        analysis_cont = 'Явного влияния на таргет нет. Людей с двумя машинами практически нет.'
    elif cont_feat == 'PERSONAL_INCOME':
        analysis_cont = 'У нас не нормальное распределение. Явного влияния на таргет нет. Здесь изначально были неверные или сомнительные значения. Они были удалены.'
    elif cont_feat == 'WORK_TIME_IN_YEARS':
        analysis_cont = 'У нас не нормальное распределение. Явного влияния на таргет нет. Здесь изначально были неверные или сомнительные значения. Они были удалены.' \
                        'Признак ввёден взамен изначального признака времени работы в месяцах.'
    elif cont_feat == 'CLOSED_LOANS':
        analysis_cont = 'Явного влияния на таргет нет.'
    elif cont_feat == 'LOAN_AMOUNT':
        analysis_cont = 'Явного влияния на таргет нет.'

    if st.button('Отобразить график выбранного числового признака'):
        st.write('You selected:', cont_feat)
        st.text(analysis_cont)
        st.pyplot(show_continuous_grafics(all_data, cont_feat, width_cont, height_cont), use_container_width=False)

    st.markdown("## Графики категориальных признаков")

    cat_feat = st.selectbox(
        'Выберете категориальную переменную',
        ('GENDER', 'EDUCATION', 'MARITAL_STATUS', 'SOCSTATUS_WORK_FL',
         'SOCSTATUS_PENS_FL', 'FL_PRESENCE_FL', 'GEN_INDUSTRY', 'GEN_TITLE',
         'JOB_DIR', 'FAMILY_INCOME', 'TERM'))

    # labels = [], title=''

    if cat_feat == 'LOAN_NOW':
        labels = ['есть кредит', 'нет кредита']
        title = "Диаграмма наличия кредита"

        analysis_cat = 'Явного распределения по признаку наличия кредита по целевой переменной нет'
    elif cat_feat == 'GENDER':
        labels = ['мужчины', 'женщины']
        title = "Диаграмма пола клиентов"

        analysis_cat = 'Явного распределения по признаку гендера по целевой переменной нет'
    elif cat_feat == 'MARITAL_STATUS':
        labels = ['Состою в браке', 'Не состоял в браке', 'Разведён(-а)',
                  'Вдовец/вдова', 'Гражданский брак']
        title = 'Диаграмма семейного положения'

        analysis_cat = 'Явного распределения по признаку семейного положения по целевой переменной нет'
    elif cat_feat == 'EDUCATION':
        labels = ['Среднее специальное', 'Среднее', 'Высшее',
                  'Неоконченное высшее', 'Неполное среднее', 'два и более высших', 'учёная степень']
        title = 'Диаграмма наличия образования'

        analysis_cat = 'Люди с более высокой степенью образования реже откликаются на промоакции'
    elif cat_feat == 'SOCSTATUS_WORK_FL':
        labels = ['Работает', 'Не работает']
        title = 'Диаграмма наличия работы'

        analysis_cat = 'Не работающие практически не представлены, поэтому сложно судить о наличии влияния этого признака'
    elif cat_feat == 'SOCSTATUS_PENS_FL':
        labels = ['Не пенсионер', 'Пенсионер']
        title = 'Диаграмма наличия пенсии'

        analysis_cat = 'Вероятно у пенсионеров меньше отклика на промоакции, но слишком мало представлено людей в этой категории'
    elif cat_feat == 'FL_PRESENCE_FL':
        labels = ['Есть', 'Нет']
        title = 'Диаграмма наличия квартиры'

        analysis_cat = 'Явного распределения по признаку наличия квартиры по целевой переменной нет'
    elif cat_feat == 'FAMILY_INCOME':
        labels = ['от 10000 до 20000 руб.',
                  'от 20000 до 50000 руб.',
                  'от 5000 до 10000 руб.',
                  'свыше 50000 руб.',
                  'до 5000 руб.']
        title = 'Диаграмма заработков'

        analysis_cat = 'Явного распределения в признаке наличия квартиры по целевой переменной нет'
    elif cat_feat == 'GEN_INDUSTRY':

        analysis_cat = 'Сложно судить о наличии влияния данного признака на целевую переменную. Мало объектов представлено' \
                   'в неосновных категориях'
    elif cat_feat == 'GEN_TITLE':

        analysis_cat = 'Сложно судить о наличии влияния данного признака на целевую переменную'
    elif cat_feat == 'JOB_DIR':

        analysis_cat = 'Сложно судить о наличии влияния данного признака на целевую переменную. Мало объектов представлено' \
                   'в неосновных категориях'
    elif cat_feat == 'TERM':

        analysis_cat = 'Явного распределения в признаке срока кредита по целевой переменной нет'

    if all_data[cat_feat].nunique() <= 5:
        pie = True
    else:
        pie = False

    if st.button('Отобразить график выбранного категориального признака'):
        st.write('You selected:', cat_feat)
        st.text(analysis_cat)
        if pie:
            st.pyplot(pie_categorical_grafics(all_data, cat_feat, labels, title, width_pie, height_pie), use_container_width=False)

        st.pyplot(show_categorical_grafics(all_data, cat_feat, width_cat, height_cat), use_container_width=False)


    st.markdown("---")
    st.markdown('## Целевая переменная')
    st.pyplot(target_grafics(all_data, 5, 4), use_container_width=False)
    st.text('Несбалансированная выборка')

def show_main_page():
    url = 'https://static.tildacdn.com/tild6132-6433-4464-b637-653334663132/bank-branch-scent-1-.jpeg'

    response = requests.get(url)
    image = Image.open(BytesIO(response.content))

    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Анализ откликов на предложения банка",
        page_icon=image
    )

process_main_page()