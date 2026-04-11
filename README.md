# YouTube Migration Parser

Небольшой Python-проект для сбора YouTube-комментариев по теме миграции в России.

Скрипт помогает собрать корпус комментариев для дальнейшей фильтрации и анализа через ML- или LLM-пайплайн.

- дорогой `search.list` расходуется дозированно;
- видео ищутся по нескольким порядкам выдачи, чтобы уменьшить повторы;
- комментарии качаются по кругу между видео, а не выжигаются на одном ролике;
- часть квоты можно оставить в резерве.

## Что делает

1. Ищет YouTube-видео по запросам вроде `мигранты в россии`.
2. Получает метаданные роликов.
3. Выгружает комментарии верхнего уровня и, при желании, ответы.
4. Распределяет квоту между поиском и комментариями.
5. Качает комментарии по `round-robin`, чтобы собрать максимум видео и максимум комментариев в рамках лимита.
6. Сохраняет сырые результаты в `CSV` и сводку в `JSON`.

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Нужен ключ YouTube Data API v3:

```bash
export YOUTUBE_API_KEY="ваш_ключ"
```

## Запуск

Базовый запуск:

```bash
python3 youtube_migration_parser.py
```

Режим под квоту `10 000`, когда нужно собрать максимум видео и комментариев:

```bash
python3 youtube_migration_parser.py \
  --query-file queries.txt \
  --published-after "2023-01-01T00:00:00Z" \
  --search-order relevance \
  --search-order date \
  --max-videos-per-query 250 \
  --max-search-pages-per-query 4 \
  --max-comments-per-video 500 \
  --search-quota-share 0.35 \
  --quota-reserve 150 \
  --output-dir output_2023_2026
```

Продолжение уже собранного набора видео после сброса дневной квоты:

```bash
python3 youtube_migration_parser.py \
  --resume-from-dir output_2023_2026 \
  --exclude-video-ids-file excluded_video_ids.txt \
  --max-comments-per-video 1000 \
  --quota-reserve 150
```

Если прошлый запуск еще не сохранял `resume_state.json`, скрипт сможет продолжить и без него, но ему придется частично повторно пройти уже скачанные страницы комментариев, чтобы добраться до следующей страницы. После этого запуска `resume_state.json` появится, и дальнейшие продолжения будут заметно эффективнее.

Пример `queries.txt`:

```text
мигранты в россии
миграция в россии
нелегальные мигранты россия
отношение к мигрантам в россии
мигранты москва
мигранты спб
```

## Как тратится квота

Базовые стоимости YouTube Data API:

- `search.list` = `100` units;
- `videos.list` = `1` unit;
- `commentThreads.list` = `1` unit.

Из-за этого поиск намного дороже комментариев. Поэтому логика такая:

1. Скрипт тратит только часть квоты на поиск видео.
2. Потом почти весь оставшийся бюджет уходит на скачивание комментариев.
3. Комментарии качаются страницами по кругу между всеми видео, чтобы не потерять охват по роликам.

Главные параметры:

- `--search-quota-share` — какую долю доступной квоты отдать на поиск;
- `--quota-reserve` — сколько units не трогать вообще;
- `--max-videos-per-query` — сколько уникальных видео собирать по каждому запросу;
- `--max-search-pages-per-query` — насколько глубоко уходить в выдачу каждого запроса;
- `--max-comments-per-video` — потолок комментариев на одно видео.
- `--resume-from-dir` — продолжить скачивание комментариев по уже найденному списку видео без нового поиска.
- `--exclude-video-ids-file` — файл со списком `video_id`, которые нужно исключить и больше не парсить.
- `--exclude-known-videos-from` — папка с прошлым `output`, из которой нужно автоматически исключить все уже найденные `video_id`.


## Результаты

Скрипт создает:

- `output/youtube_videos.csv` — список найденных видео;
- `output/youtube_comments.csv` — комментарии с признаками;
- `output/video_comment_stats.csv` — сколько комментариев у видео заявлено API и сколько реально скачано в текущем запуске;
- `output/summary.json` — агрегированная сводка.

Ключевые поля в `youtube_comments.csv`:

- `text` — текст комментария;
- `video_id` — идентификатор видео;
- `thread_id` — идентификатор треда комментария;
- `comment_id` — идентификатор комментария;
- `author_name` — имя автора комментария; // можно удалить!!!
- `published_at` — дата публикации комментария.

Ключевые поля в `video_comment_stats.csv`:

- `api_reported_comment_count` — сколько комментариев по видео сообщает YouTube API;
- `estimated_downloadable_in_run` — сколько комментариев можно было скачать в рамках текущего лимита `--max-comments-per-video`;
- `downloaded_comment_count` — сколько комментариев реально сохранено;
- `comment_page_requests` — сколько раз вызывался `commentThreads.list` для этого видео;
- `download_coverage_vs_api_reported` — доля скачанного от общего числа, которое сообщает API.

В `summary.json` также есть:

- `duplicate_video_hits_skipped` — сколько повторных роликов было отброшено между запросами;
- `excluded_video_ids_count` — сколько `video_id` было подано в стоп-лист;
- `excluded_video_hits_skipped` — сколько попаданий в стоп-лист было отфильтровано;
- `quota_estimate` — оценка потраченных и оставшихся quota units;
- `query_stats` — статистика по каждому запросу;
- `search_strategy` — параметры, с которыми был запущен сбор.
- `new_comments_downloaded_this_run` — сколько новых комментариев скачано именно в текущем запуске.

Формат `excluded_video_ids.txt`:

```text
_SvvKcX3FVo
rRMODBZTvoE
cYrq3GnKjkE
```

Пример поиска только новых видео, не включая уже собранный корпус:

```bash
python3 youtube_migration_parser.py \
  --query-file queries.txt \
  --exclude-video-ids-file excluded_video_ids.txt \
  --exclude-known-videos-from output_2023_2026 \
  --published-after "2020-01-01T00:00:00Z" \
  --search-order relevance \
  --search-order date \
  --max-videos-per-query 200 \
  --max-search-pages-per-query 1 \
  --max-comments-per-video 1 \
  --search-quota-share 0.24 \
  --quota-reserve 150 \
  --output-dir output_2020_2026_search_only
```

## Важно

- Скрипт использует только публичные данные YouTube API.
- Квота считается оценочно по стандартной стоимости методов: `search=100`, `videos=1`, `commentThreads=1`.
