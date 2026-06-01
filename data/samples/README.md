# Sample & derived data

| File | Описание |
|------|----------|
| `ofz_auctions.csv` | Ручной сэмпл (короткий) |
| `ofz_auctions_historical.csv` | История ОФЗ: proxy + Wayback Minfin (где есть) + live Minfin (2024+) |
| `repo_auctions.csv` | Fallback, если ЦБ недоступен |
| `roskazna_deposits.csv` | Размещения ЕКС: SORS-proxy 2012–23 + sample 2024–25 |

```bash
python scripts/build_roskazna_from_sors.py
```

## OFZ proxy

Minfin на сайте отдаёт XLSX только за текущий год. Для истории:

```bash
python scripts/build_ofz_from_bliquidity.py
```

Строит `ofz_auctions_historical.csv`: биенедельный календарь аукционов, cover ratio оценён от `bliquidity` (столбец `source=bliquidity_proxy`). На защите укажите это явно; живые строки Minfin — `source=minfin_live`.

## OFZ Wayback (Internet Archive)

Для годов, где Minfin не отдаёт XLSX, подтягиваются HTML-таблицы со снимков archive.org:

```bash
python scripts/fetch_ofz_wayback.py --from 2019 --to 2023
```

Строки помечаются `source=wayback_html` / `wayback_xlsx`. Proxy за те же годы заменяются при совпадении даты+серии.
