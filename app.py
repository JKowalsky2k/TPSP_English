import json
import random
import hashlib
import threading
import re
import shutil
import tempfile
from copy import deepcopy
from collections import defaultdict
from collections.abc import Iterable, Sequence
from colorsys import hsv_to_rgb
from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    make_response,
    redirect,
    url_for,
    send_file,
    after_this_request,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
from weasyprint import HTML, CSS
from sqlalchemy import delete, inspect

BASE_DIR = Path(__file__).resolve().parent
DATABASE_NAME = "competitors.db"
MAX_PARCOUR_SCORE = 25
PARCOUR_FIELDS = ("parcour1", "parcour2", "parcour3", "parcour4")
DEFAULT_COMPETITORS = 66
SQUAD_SIZE = 6
SERIES_COUNT = 6
SHOTS_PER_SERIES = 5
INSTANCE_PATH = BASE_DIR / "instance"
SETTINGS_PATH = INSTANCE_PATH / "settings.json"
DATABASE_PATH = INSTANCE_PATH / DATABASE_NAME
DEFAULT_COLUMNS: list[dict[str, object]] = [
    {"name": "squad", "label": "Grupa", "editable": False},
    {"name": "number", "label": "Numer", "editable": False},
    {"name": "name", "label": "Imię", "editable": True},
    {"name": "lastname", "label": "Nazwisko", "editable": True},
    {"name": "category", "label": "Klasa", "editable": True},
    {"name": "parcour1", "label": "Parkur 1", "editable": True},
    {"name": "parcour2", "label": "Parkur 2", "editable": True},
    {"name": "parcour3", "label": "Parkur 3", "editable": True},
    {"name": "parcour4", "label": "Parkur 4", "editable": True},
    {"name": "result", "label": "Wynik", "editable": False},
]
DEFAULT_SETTINGS = {
    "page_title": "🏆 Wyniki zawodów",
    "columns": deepcopy(DEFAULT_COLUMNS),
    "competitor_count": DEFAULT_COMPETITORS,
}
BACKUP_DIR = SETTINGS_PATH.parent / "backups"

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

def load_app_settings() -> dict[str, object]:
    settings: dict[str, object] = deepcopy(DEFAULT_SETTINGS)
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open(encoding="utf-8") as handle:
                raw_settings = json.load(handle)
        except (OSError, json.JSONDecodeError):
            raw_settings = {}
        if isinstance(raw_settings, dict):
            page_title = raw_settings.get("page_title")
            if isinstance(page_title, str):
                settings["page_title"] = page_title

            columns_value = raw_settings.get("columns")
            if isinstance(columns_value, list):
                normalized_columns: list[dict[str, object]] = []
                for column in columns_value:
                    if not isinstance(column, dict):
                        continue
                    name = str(column.get("name") or "").strip()
                    label = str(column.get("label") or "").strip()
                    editable = bool(column.get("editable"))
                    if not name:
                        continue
                    if not label:
                        label = name
                    normalized_columns.append(
                        {"name": name, "label": label, "editable": editable}
                    )
                if normalized_columns:
                    settings["columns"] = normalized_columns

            competitor_count_value = raw_settings.get("competitor_count")
            try:
                competitor_count = int(competitor_count_value)
            except (TypeError, ValueError):
                competitor_count = settings.get("competitor_count", DEFAULT_COMPETITORS)
            if competitor_count > 0:
                settings["competitor_count"] = competitor_count
    return settings


def save_app_settings(settings: dict[str, object]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)

TOP_RANK_SATURATION = {1: 0.65, 2: 0.55, 3: 0.45}
TOP_RANK_VALUE = {1: 0.9, 2: 0.94, 3: 0.97}
UNCATEGORIZED_LABEL = "Uncategorized"

_app_settings_initial = load_app_settings()
COLUMNS: list[dict[str, object]] = deepcopy(_app_settings_initial.get("columns", DEFAULT_COLUMNS))
EDITABLE_COLUMNS = [column["name"] for column in COLUMNS if column.get("editable")]
PARCOUR_LABELS = [column["label"] for column in COLUMNS if column["name"] in PARCOUR_FIELDS]


def _rgb_float_to_hex(rgb: Sequence[float]) -> str:
    red, green, blue = (
        max(0, min(255, int(round(component * 255)))) for component in rgb
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def generate_category_rank_colors(categories: Iterable[str]) -> dict[str, dict[int, str]]:
    category_list = sorted(categories)
    if not category_list:
        return {}

    colors: dict[str, dict[int, str]] = {}
    hue_step = 0.6180339887498949  # irrational number for pleasant hue distribution
    for idx, category in enumerate(category_list):
        hue = (idx * hue_step) % 1.0
        rank_colors: dict[int, str] = {}
        for rank in (1, 2, 3):
            saturation = TOP_RANK_SATURATION.get(rank, 0.5)
            value = TOP_RANK_VALUE.get(rank, 0.95)
            rgb = hsv_to_rgb(hue, saturation, value)
            rank_colors[rank] = _rgb_float_to_hex(rgb)
        colors[category] = rank_colors

    return colors
FIRST_NAMES = [
    "Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Cameron", "Avery",
    "Quinn", "Harper", "Jamie", "Reese", "Logan", "Rowan", "Peyton", "Dakota",
    "Skyler", "Finley", "Emerson", "Hayden"
]
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin"
]
CATEGORY_CHOICES = ["Junior", "Lady", "Senior", "A", "B", "C"]

class Competitor(db.Model):
    __tablename__ = "competitor"
    id = db.Column(db.Integer, primary_key=True)
    squad = db.Column(db.Integer, nullable=False)
    number = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(50), default="")
    lastname = db.Column(db.String(50), default="")
    category = db.Column(db.String(50), default="")
    parcour1 = db.Column(db.Integer, default=0)
    parcour2 = db.Column(db.Integer, default=0)
    parcour3 = db.Column(db.Integer, default=0)
    parcour4 = db.Column(db.Integer, default=0)

    @property
    def result(self):
        return sum(getattr(self, field) or 0 for field in PARCOUR_FIELDS)

    def to_dict(self) -> dict[str, object]:
        return {column["name"]: getattr(self, column["name"]) for column in COLUMNS}


def _safe_int(value: object, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def calculate_competitors_hash(competitors: list["Competitor"]) -> str | None:
    if not competitors:
        return None
    payload = [
        {
            "id": competitor.id,
            **{column["name"]: getattr(competitor, column["name"]) for column in COLUMNS},
        }
        for competitor in competitors
    ]
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def calculate_records_hash(records: list[dict[str, object]]) -> str:
    normalized_payload = []
    for record in records:
        normalized_entry: dict[str, object] = {"id": record.get("id")}
        for column in COLUMNS:
            name = column["name"]
            normalized_entry[name] = record.get(name)
        normalized_payload.append(normalized_entry)

    serialized = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class AutoBackupManager:
    def __init__(self, interval_seconds: int = 300):
        self.interval_seconds = interval_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._dirty = True
        self._last_hash: str | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def mark_dirty(self) -> None:
        with self._lock:
            self._dirty = True

    def update_last_hash(self, hash_value: str | None) -> None:
        with self._lock:
            self._last_hash = hash_value
            self._dirty = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _should_backup(self, hash_value: str | None) -> bool:
        with self._lock:
            if self._dirty:
                return True
            return hash_value is not None and hash_value != self._last_hash

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            with app.app_context():
                competitors = Competitor.query.order_by(Competitor.id).all()
                hash_value = calculate_competitors_hash(competitors)

                if not competitors:
                    self.update_last_hash(None)
                    continue

                if not self._should_backup(hash_value):
                    if hash_value == self._last_hash:
                        with self._lock:
                            self._dirty = False
                    continue

                backup_path = create_competitors_backup(competitors, hash_value=hash_value)
                if backup_path:
                    self.update_last_hash(hash_value)


auto_backup_manager = AutoBackupManager()


class MetricsCache:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hash: str | None = None
        self._payload: dict[int, list[dict[str, object]]] | None = None

    def invalidate(self) -> None:
        with self._lock:
            self._hash = None
            self._payload = None

    def get(self, competitors: list[Competitor], hash_value: str | None) -> dict[int, list[dict[str, object]]]:
        if hash_value:
            with self._lock:
                if self._hash == hash_value and self._payload is not None:
                    return deepcopy(self._payload)

        metrics_payload = build_metrics_data(competitors)

        if hash_value:
            with self._lock:
                self._hash = hash_value
                self._payload = deepcopy(metrics_payload)
        return metrics_payload


metrics_cache = MetricsCache()


def sanitize_competition_folder_name(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", value.strip())
    slug = slug.strip("._-")
    return slug or "zawody"


def ensure_database_ready() -> None:
    db_path = DATABASE_PATH
    try:
        INSTANCE_PATH.mkdir(parents=True, exist_ok=True)
        if not db_path.exists():
            db.create_all()
            return

        inspector = inspect(db.engine)
        if "competitor" not in inspector.get_table_names():
            db.create_all()
    except Exception:
        db.create_all()


def serialize_competitor_for_backup(competitor: "Competitor") -> dict[str, object]:
    record = competitor.to_dict()
    record["id"] = competitor.id
    record["result"] = competitor.result
    return record


def create_competitors_backup(
    competitors: list["Competitor"], *, hash_value: str | None = None
) -> Path | None:
    if not competitors:
        return None

    if hash_value is None:
        hash_value = calculate_competitors_hash(competitors)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = BACKUP_DIR / f"competitors-{timestamp}.json"
    snapshot = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "count": len(competitors),
        "hash": hash_value,
        "columns": COLUMNS,
        "records": [serialize_competitor_for_backup(competitor) for competitor in competitors],
    }

    with backup_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, ensure_ascii=False, indent=2)

    return backup_path


def restore_competitors_from_snapshot(snapshot: dict[str, object]) -> tuple[int, str]:
    ensure_database_ready()
    records = snapshot.get("records")
    if not isinstance(records, list):
        raise ValueError("Niepoprawny format pliku kopii zapasowej.")

    try:
        db.session.execute(delete(Competitor))
        db.session.flush()

        competitors_to_insert: list[Competitor] = []
        for record in records:
            raw_id = record.get("id")
            try:
                competitor_id = int(raw_id)
            except (TypeError, ValueError):
                competitor_id = None

            competitors_to_insert.append(
                Competitor(
                    id=competitor_id,
                    squad=_safe_int(record.get("squad"), 0),
                    number=_safe_int(record.get("number"), 0),
                    name=str(record.get("name") or ""),
                    lastname=str(record.get("lastname") or ""),
                    category=str(record.get("category") or ""),
                    parcour1=clamp_score(record.get("parcour1")),
                    parcour2=clamp_score(record.get("parcour2")),
                    parcour3=clamp_score(record.get("parcour3")),
                    parcour4=clamp_score(record.get("parcour4")),
                )
            )

        if competitors_to_insert:
            db.session.bulk_save_objects(competitors_to_insert, return_defaults=False)

        db.session.commit()
    except Exception as exc:
        db.session.rollback()
        raise RuntimeError("Nie udało się przywrócić danych z kopii zapasowej.") from exc

    hash_value = snapshot.get("hash")
    if not hash_value:
        hash_value = calculate_records_hash(records)

    auto_backup_manager.update_last_hash(hash_value)
    metrics_cache.invalidate()

    return len(records), hash_value


def populate_competitors_with_examples() -> int:
    competitors = Competitor.query.order_by(Competitor.id).all()
    random.seed()

    if not competitors:
        settings = load_app_settings()
        competitor_limit = settings.get("competitor_count", DEFAULT_COMPETITORS)
        new_competitors = []
        for idx in range(competitor_limit):
            squad_number = idx // SQUAD_SIZE + 1
            new_competitors.append(
                Competitor(
                    number=idx + 1,
                    squad=squad_number,
                    name=random.choice(FIRST_NAMES),
                    lastname=random.choice(LAST_NAMES),
                    category=random.choice(CATEGORY_CHOICES),
                    parcour1=random.randint(0, MAX_PARCOUR_SCORE),
                    parcour2=random.randint(0, MAX_PARCOUR_SCORE),
                    parcour3=random.randint(0, MAX_PARCOUR_SCORE),
                    parcour4=random.randint(0, MAX_PARCOUR_SCORE),
                )
            )
        if new_competitors:
            db.session.bulk_save_objects(new_competitors, return_defaults=False)
        db.session.commit()
        metrics_cache.invalidate()
        return len(new_competitors)

    for competitor in competitors:
        competitor.name = random.choice(FIRST_NAMES)
        competitor.lastname = random.choice(LAST_NAMES)
        competitor.category = random.choice(CATEGORY_CHOICES)
        for field in PARCOUR_FIELDS:
            setattr(competitor, field, random.randint(0, MAX_PARCOUR_SCORE))

    db.session.commit()
    metrics_cache.invalidate()
    return len(competitors)


def create_placeholder_competitors(count: int) -> int:
    target_count = max(0, int(count or 0))
    if target_count <= 0:
        return 0

    existing = Competitor.query.count()
    if existing > 0:
        return 0

    placeholder_competitors: list[Competitor] = []
    for idx in range(target_count):
        squad_number = idx // SQUAD_SIZE + 1
        placeholder_competitors.append(
            Competitor(
                number=idx + 1,
                squad=squad_number,
                name="",
                lastname="",
                category="",
                parcour1=0,
                parcour2=0,
                parcour3=0,
                parcour4=0,
            )
        )

    if not placeholder_competitors:
        return 0

    db.session.bulk_save_objects(placeholder_competitors, return_defaults=False)
    db.session.commit()
    metrics_cache.invalidate()
    return len(placeholder_competitors)


def clamp_score(value: object) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(MAX_PARCOUR_SCORE, score))

def init_db():
    db.create_all()


def fetch_competitors(sort: str, order: str) -> list[Competitor]:
    requested_sort = (sort or "number").lower()
    requested_order = (order or "asc").lower()

    if requested_sort == "result":
        competitors = sorted(
            Competitor.query.all(),
            key=lambda competitor: competitor.result,
            reverse=requested_order == "desc",
        )
    elif requested_sort == "category_result":
        competitors_by_category: dict[str, list[Competitor]] = defaultdict(list)
        for competitor in Competitor.query.all():
            category = competitor.category or UNCATEGORIZED_LABEL
            competitors_by_category[category].append(competitor)

        for members in competitors_by_category.values():
            members.sort(
                key=lambda competitor: (
                    -competitor.result,
                    (competitor.lastname or "").lower(),
                    (competitor.name or "").lower(),
                    competitor.number,
                )
            )

        sorted_categories = sorted(
            competitors_by_category.items(),
            key=lambda item: item[0].lower(),
            reverse=requested_order == "desc",
        )

        competitors = [member for _, members in sorted_categories for member in members]
    else:
        column_attr = getattr(Competitor, requested_sort, Competitor.number)
        if requested_order == "desc":
            column_attr = column_attr.desc()
        competitors = Competitor.query.order_by(column_attr).all()

    return competitors

def group_competitors_by_squad(competitors: list[Competitor]) -> dict[int, list[Competitor]]:
    squads: dict[int, list[Competitor]] = defaultdict(list)
    for competitor in competitors:
        squads[competitor.squad].append(competitor)
    return squads


def build_metrics_data(competitors: list[Competitor]) -> dict[int, list[dict[str, object]]]:
    squads = group_competitors_by_squad(competitors)
    metrics: dict[int, list[dict[str, object]]] = {}
    empty_markers_template = [["" for _ in range(SHOTS_PER_SERIES)] for _ in range(SERIES_COUNT)]

    for squad_number in sorted(squads):
        members = squads[squad_number]
        row_count = max(SQUAD_SIZE, len(members))
        squad_metrics: list[dict[str, object]] = []

        for field_index, (field, label) in enumerate(zip(PARCOUR_FIELDS, PARCOUR_LABELS)):
            entries: list[dict[str, object]] = []

            member_count = len(members)
            if member_count:
                shift = field_index % member_count
            else:
                shift = 0

            for idx in range(member_count):
                competitor = members[(idx - shift) % member_count]
                target_series_index = max(
                    0, min(SERIES_COUNT - 1, row_count - idx - 1)
                )
                markers_by_series = [
                    (["X"] * SHOTS_PER_SERIES) if series_idx == target_series_index else ([""] * SHOTS_PER_SERIES)
                    for series_idx in range(SERIES_COUNT)
                ]

                display_name = f"{competitor.lastname} {competitor.name}".strip()
                entries.append(
                    {
                        "number": competitor.number,
                        "name": display_name,
                        "markers": markers_by_series,
                    }
                )

            while len(entries) < row_count:
                entries.append(
                    {
                        "number": "",
                        "name": "",
                        "markers": [row[:] for row in empty_markers_template],
                    }
                )

            squad_metrics.append(
                {
                    "label": label,
                    "competitors": entries,
                }
            )

        metrics[squad_number] = squad_metrics

    return metrics


def build_results_pdf_bytes(competitors: list[Competitor], sort: str, order: str) -> bytes:
    category_groups: dict[str, list[Competitor]] = defaultdict(list)
    competitor_categories: dict[int, str] = {}
    for competitor in competitors:
        category = competitor.category or UNCATEGORIZED_LABEL
        category_groups[category].append(competitor)
        competitor_categories[competitor.id] = category

    rank_map: dict[int, int] = {}
    for category, members in category_groups.items():
        ranked_members = sorted(members, key=lambda competitor: competitor.result, reverse=True)
        for position, competitor in enumerate(ranked_members[:3], start=1):
            rank_map[competitor.id] = position

    category_colors = generate_category_rank_colors(category_groups.keys())
    highlight_styles: dict[int, str] = {}
    for competitor_id, rank in rank_map.items():
        category = competitor_categories.get(competitor_id, UNCATEGORIZED_LABEL)
        color = category_colors.get(category, {}).get(rank)
        if color:
            highlight_styles[competitor_id] = color

    rendered = render_template(
        "pdf_raw.html",
        competitors=competitors,
        columns=COLUMNS,
        getattr=getattr,
        category_rank_map=rank_map,
        category_rank_styles=highlight_styles,
    )

    pdf_bytes = HTML(string=rendered, base_url=str(BASE_DIR)).write_pdf(
        stylesheets=[
            CSS(filename=str(BASE_DIR / "static" / "pdf.css")),
            CSS(string='@page { size: A4 portrait; margin: 10mm }'),
        ]
    )

    return pdf_bytes


def build_metrics_pdf_bytes(metrics: dict[int, list[dict[str, object]]]) -> bytes:
    rendered = render_template(
        "metrics.html",
        metrics=metrics,
        series_range=range(1, SERIES_COUNT + 1),
        shots_range=range(1, SHOTS_PER_SERIES + 1),
        shots_per_series=SHOTS_PER_SERIES,
        total_columns=4 + SERIES_COUNT * SHOTS_PER_SERIES,
    )

    pdf_bytes = HTML(string=rendered, base_url=str(BASE_DIR)).write_pdf(
        stylesheets=[
            CSS(filename=str(BASE_DIR / "static" / "metrics.css")),
            CSS(string='@page { size: A4 landscape; margin: 10mm }'),
        ]
    )

    return pdf_bytes

@app.before_request
def ensure_auto_backup_worker() -> None:
    ensure_database_ready()
    if not auto_backup_manager.is_running():
        auto_backup_manager.start()

@app.route("/setup", methods=["GET", "POST"])
def initial_setup():
    ensure_database_ready()
    existing_count = Competitor.query.count()
    if existing_count > 0:
        return redirect(url_for("index"))

    settings = load_app_settings()
    default_competitor_count = settings.get("competitor_count", DEFAULT_COMPETITORS)
    default_page_title = settings.get("page_title", DEFAULT_SETTINGS["page_title"])
    error: str | None = None
    field_value: str = str(default_competitor_count)
    name_value: str = str(default_page_title)

    if request.method == "POST":
        raw_value = (request.form.get("competitor_count") or "").strip()
        field_value = raw_value
        raw_name = (request.form.get("competition_name") or "").strip()
        cleaned_name = " ".join(raw_name.split())
        name_value = cleaned_name
        competition_name = cleaned_name[:120] if cleaned_name else DEFAULT_SETTINGS["page_title"]
        try:
            competitor_count = int(raw_value)
        except (TypeError, ValueError):
            competitor_count = 0

        if competitor_count <= 0:
            error = "Podaj dodatnią liczbę zawodników."
        elif competitor_count > 2000:
            error = "Maksymalna liczba zawodników to 2000."
        else:
            settings["competitor_count"] = competitor_count
            settings["page_title"] = competition_name
            save_app_settings(settings)
            try:
                created = create_placeholder_competitors(competitor_count)
            except Exception:
                db.session.rollback()
                error = "Nie udało się utworzyć listy zawodników. Spróbuj ponownie."
            else:
                competitors = Competitor.query.order_by(Competitor.id).all()
                hash_value = calculate_competitors_hash(competitors)
                auto_backup_manager.update_last_hash(hash_value)
                metrics_cache.invalidate()
                return redirect(url_for("index"))

    return render_template(
        "setup.html",
        error=error,
        competitor_count_value=field_value,
        default_competitor_count=default_competitor_count,
        competition_name_value=name_value,
        default_page_title=default_page_title,
    )


@app.route("/")
def index():
    ensure_database_ready()
    sort = request.args.get("sort", "number")
    order = request.args.get("order", "asc")
    if Competitor.query.count() == 0:
        return redirect(url_for("initial_setup"))
    competitors = fetch_competitors(sort, order)
    settings = load_app_settings()
    page_title = settings.get("page_title", DEFAULT_SETTINGS["page_title"])

    return render_template(
        "index.html",
        competitors=competitors,
        columns=COLUMNS,
        editable_columns=EDITABLE_COLUMNS,
        sort=sort,
        order=order,
        default_page_title=DEFAULT_SETTINGS["page_title"],
        page_title=page_title,
        getattr=getattr,
    )

@app.route("/update", methods=["POST"])
def update():
    payload = request.get_json(silent=True) or {}
    competitor_id = payload.get("id")
    field = payload.get("field")

    if competitor_id is None or field is None:
        return jsonify({"error": "Missing competitor id or field"}), 400

    competitor = Competitor.query.get(competitor_id)
    if competitor is None:
        return jsonify({"error": "Competitor not found"}), 404

    if field not in EDITABLE_COLUMNS:
        return jsonify({"error": "Field is not editable"}), 400

    value = payload.get("value", "")

    if field in PARCOUR_FIELDS:
        value = clamp_score(value)

    setattr(competitor, field, value)
    db.session.commit()
    auto_backup_manager.mark_dirty()
    metrics_cache.invalidate()

    return jsonify(competitor.to_dict())


@app.route("/clear", methods=["POST"])
def clear_competitors():
    competitors = Competitor.query.order_by(Competitor.id).all()
    hash_before = calculate_competitors_hash(competitors)
    backup_path = (
        create_competitors_backup(competitors, hash_value=hash_before) if competitors else None
    )
    cleared_count = 0

    for competitor in competitors:
        competitor.name = ""
        competitor.lastname = ""
        competitor.category = ""
        for field in PARCOUR_FIELDS:
            setattr(competitor, field, 0)
        cleared_count += 1

    if cleared_count:
        db.session.commit()
        auto_backup_manager.mark_dirty()
        metrics_cache.invalidate()

    response_payload: dict[str, object] = {"status": "ok", "cleared": cleared_count}
    if backup_path:
        try:
            response_payload["backup"] = str(backup_path.relative_to(BASE_DIR))
        except ValueError:
            response_payload["backup"] = str(backup_path)

    return jsonify(response_payload)


@app.route("/backup", methods=["POST"])
def create_backup_endpoint():
    competitors = Competitor.query.order_by(Competitor.id).all()
    if not competitors:
        return jsonify({"status": "empty", "message": "Brak danych do zapisania"}), 400

    try:
        hash_value = calculate_competitors_hash(competitors)
        backup_path = create_competitors_backup(competitors, hash_value=hash_value)
    except OSError as exc:
        return jsonify({"status": "error", "message": f"Nie udało się zapisać kopii zapasowej: {exc}"}), 500

    if not backup_path:
        return jsonify({"status": "error", "message": "Kopia zapasowa nie została utworzona"}), 500

    try:
        relative_path = str(backup_path.relative_to(BASE_DIR))
    except ValueError:
        relative_path = str(backup_path)

    auto_backup_manager.update_last_hash(hash_value)

    return jsonify({"status": "ok", "backup": relative_path})


@app.route("/backups", methods=["GET"])
def list_backups():
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_root = BACKUP_DIR.resolve()
    backup_entries = []

    for path in sorted(backup_root.glob("*.json"), key=lambda item: item.name, reverse=True):
        try:
            stat = path.stat()
            entry = {
                "filename": path.name,
                "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size,
            }
            backup_entries.append(entry)
        except OSError:
            continue

    return jsonify({"backups": backup_entries})


@app.route("/restore", methods=["POST"])
def restore_backup():
    payload = request.get_json(silent=True) or {}
    raw_filename = payload.get("filename", "")
    filename = str(raw_filename or "").strip()
    if not filename:
        return jsonify({"status": "error", "message": "Nie podano nazwy pliku kopii zapasowej."}), 400

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    backup_root = BACKUP_DIR.resolve()
    backup_path = (BACKUP_DIR / filename).resolve()

    if not str(backup_path).startswith(str(backup_root)):
        return jsonify({"status": "error", "message": "Nieprawidłowa nazwa pliku kopii zapasowej."}), 400

    if not backup_path.is_file():
        return jsonify({"status": "error", "message": "Wybrana kopia zapasowa nie istnieje."}), 404

    try:
        with backup_path.open(encoding="utf-8") as handle:
            snapshot = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return jsonify({"status": "error", "message": "Nie udało się odczytać pliku kopii zapasowej."}), 400

    try:
        restored_count, hash_value = restore_competitors_from_snapshot(snapshot)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
    except Exception:
        return jsonify({"status": "error", "message": "Nie udało się przywrócić danych z kopii zapasowej."}), 500

    return jsonify(
        {
            "status": "ok",
            "restored": restored_count,
            "backup": filename,
            "hash": hash_value,
            "message": f"Przywrócono {restored_count} rekordów z kopii {filename}.",
        }
    )


@app.route("/restore/upload", methods=["POST"])
def restore_backup_upload():
    uploaded_file = request.files.get("backup")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"status": "error", "message": "Nie wybrano pliku kopii zapasowej."}), 400

    filename = secure_filename(uploaded_file.filename)
    if not filename.lower().endswith(".json"):
        return jsonify({"status": "error", "message": "Obsługiwane są wyłącznie pliki JSON."}), 400

    try:
        content = uploaded_file.read()
    except OSError:
        return jsonify({"status": "error", "message": "Nie udało się odczytać przesłanego pliku."}), 400

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"status": "error", "message": "Plik kopii zapasowej musi być w kodowaniu UTF-8."}), 400

    try:
        snapshot = json.loads(text)
    except json.JSONDecodeError:
        return jsonify({"status": "error", "message": "Przesłany plik nie jest prawidłowym JSON-em."}), 400

    try:
        restored_count, hash_value = restore_competitors_from_snapshot(snapshot)
    except ValueError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500
    except Exception:
        return jsonify({"status": "error", "message": "Nie udało się przywrócić danych z kopii zapasowej."}), 500

    message = f"Przywrócono {restored_count} rekordów z pliku {filename}."
    return jsonify({"status": "ok", "restored": restored_count, "hash": hash_value, "message": message})


@app.route("/dev/populate", methods=["POST"])
def dev_populate():
    try:
        populated_count = populate_competitors_with_examples()
    except Exception:
        db.session.rollback()
        return jsonify({"status": "error", "message": "Nie udało się wypełnić przykładowymi danymi."}), 500

    auto_backup_manager.mark_dirty()

    competitors = Competitor.query.order_by(Competitor.id).all()
    hash_value = calculate_competitors_hash(competitors)
    auto_backup_manager.update_last_hash(hash_value)
    metrics_cache.invalidate()

    return jsonify(
        {
            "status": "ok",
            "updated": populated_count,
            "hash": hash_value,
            "message": f"Wypełniono {populated_count} rekordów przykładowymi danymi.",
        }
    )


@app.route("/settings/title", methods=["POST"])
def update_page_title():
    payload = request.get_json(silent=True) or {}
    raw_title = payload.get("title", "")
    if not isinstance(raw_title, str):
        raw_title = str(raw_title or "")
    cleaned_title = raw_title.strip()
    if cleaned_title:
        cleaned_title = " ".join(cleaned_title.splitlines())
    cleaned_title = cleaned_title[:120]
    if not cleaned_title:
        cleaned_title = DEFAULT_SETTINGS["page_title"]

    settings = load_app_settings()
    settings["page_title"] = cleaned_title
    save_app_settings(settings)

    return jsonify({"page_title": cleaned_title})


@app.route("/live")
def live():
    ensure_database_ready()
    if Competitor.query.count() == 0:
        return redirect(url_for("initial_setup"))
    # sort by result descending
    competitors = fetch_competitors("result", "desc")
    return render_template("live.html", competitors=competitors, columns=COLUMNS, getattr=getattr)

@app.route("/live-data")
def live_data():
    ensure_database_ready()
    sort = request.args.get("sort", "result")
    order = request.args.get("order", "desc")
    if Competitor.query.count() == 0:
        return jsonify([])
    competitors = fetch_competitors(sort, order)
    data = [competitor.to_dict() for competitor in competitors]
    return jsonify(data)

@app.route("/export-pdf")
def export_pdf():
    ensure_database_ready()
    if Competitor.query.count() == 0:
        return redirect(url_for("initial_setup"))
    sort = request.args.get("sort", "result")
    order = request.args.get("order", "desc")
    competitors = fetch_competitors(sort, order)
    pdf = build_results_pdf_bytes(competitors, sort, order)

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=wyniki.pdf'
    return response

@app.route("/metrics-pdf")
def export_metrics_pdf():
    ensure_database_ready()
    if Competitor.query.count() == 0:
        return redirect(url_for("initial_setup"))
    metrics_competitors = Competitor.query.order_by(Competitor.squad, Competitor.number).all()
    metrics_hash = calculate_competitors_hash(metrics_competitors)
    metrics = metrics_cache.get(metrics_competitors, metrics_hash)
    pdf = build_metrics_pdf_bytes(metrics)

    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=metryki.pdf"
    return response


@app.route("/finish")
def finish_competition():
    ensure_database_ready()
    if Competitor.query.count() == 0:
        return redirect(url_for("initial_setup"))

    settings = load_app_settings()
    competition_name = settings.get("page_title", DEFAULT_SETTINGS["page_title"])
    folder_slug = sanitize_competition_folder_name(competition_name)

    tmp_root = Path(tempfile.mkdtemp(prefix="tpsp_finish_"))
    package_dir = tmp_root / folder_slug
    package_dir.mkdir(parents=True, exist_ok=True)

    competitors_all = Competitor.query.order_by(Competitor.id).all()
    current_hash = calculate_competitors_hash(competitors_all)
    create_competitors_backup(competitors_all, hash_value=current_hash)

    db_path = DATABASE_PATH
    if db_path.exists():
        shutil.copy2(db_path, package_dir / DATABASE_NAME)

    if SETTINGS_PATH.exists():
        shutil.copy2(SETTINGS_PATH, package_dir / "settings.json")

    if BACKUP_DIR.exists():
        shutil.copytree(BACKUP_DIR, package_dir / "backups", dirs_exist_ok=True)

    results_competitors = fetch_competitors("result", "desc")
    (package_dir / "wyniki.pdf").write_bytes(build_results_pdf_bytes(results_competitors, "result", "desc"))

    metrics_competitors = Competitor.query.order_by(Competitor.squad, Competitor.number).all()
    metrics_hash = calculate_competitors_hash(metrics_competitors)
    metrics_payload = metrics_cache.get(metrics_competitors, metrics_hash)
    (package_dir / "metryki.pdf").write_bytes(build_metrics_pdf_bytes(metrics_payload))

    archive_base = tmp_root / folder_slug
    archive_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=tmp_root, base_dir=folder_slug))

    @after_this_request
    def cleanup(response):
        try:
            shutil.rmtree(tmp_root, ignore_errors=True)
        except Exception:
            pass
        return response

    response = send_file(
        archive_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"{folder_slug}.zip",
        max_age=0,
    )

    # side-effect cleanup after response
    def remove_original_data():
        try:
            db.session.remove()
        except Exception:
            pass
        try:
            db.engine.dispose()
        except Exception:
            pass

        try:
            shutil.rmtree(BACKUP_DIR, ignore_errors=True)
        except Exception:
            pass

        try:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            pass

        try:
            if SETTINGS_PATH.exists():
                SETTINGS_PATH.unlink()
        except Exception:
            pass

        metrics_cache.invalidate()

    remove_original_data()
    try:
        db.session.remove()
    except Exception:
        pass

    return response


if __name__ == "__main__":
    with app.app_context():
        init_db()
        auto_backup_manager.start()
    app.run(debug=True)
