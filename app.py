import json
import random
import hashlib
import threading
import re
import shutil
import tempfile
import functools
import html
import os
import unicodedata
from copy import deepcopy
from collections import defaultdict
from collections.abc import Iterable, Sequence
from colorsys import hsv_to_rgb
from datetime import datetime, timedelta
from io import BytesIO
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
from sqlalchemy import delete, inspect

_REPORTLAB_IMPORT_ERROR: Exception | None = None
try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
except ImportError as exc:
    REPORTLAB_AVAILABLE = False
    _REPORTLAB_IMPORT_ERROR = exc
else:
    REPORTLAB_AVAILABLE = True

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
DEFAULT_SCHEDULE_START_TIME = "08:00"
DEFAULT_SCHEDULE_SLOT_MINUTES = 20
DEFAULT_SCHEDULE_STAND_COUNT = 4
DEFAULT_SCHEDULE_END_TIME = ""
DEFAULT_SCHEDULE_ROUNDS = 2
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
    "schedule_start_time": DEFAULT_SCHEDULE_START_TIME,
    "schedule_slot_minutes": DEFAULT_SCHEDULE_SLOT_MINUTES,
    "schedule_stand_count": DEFAULT_SCHEDULE_STAND_COUNT,
    "schedule_end_time": DEFAULT_SCHEDULE_END_TIME,
}
BACKUP_DIR = SETTINGS_PATH.parent / "backups"

def _normalize_time_string(value: object, fallback: str = DEFAULT_SCHEDULE_START_TIME) -> str:
    raw_value = str(value or "").strip()
    match = re.match(r"^(\d{1,2}):(\d{2})$", raw_value)
    if not match:
        return fallback
    hour, minute = (int(part) for part in match.groups())
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return fallback
    return f"{hour:02d}:{minute:02d}"

def _clean_optional_time(value: object) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""
    normalized = _normalize_time_string(raw_value, fallback="")
    return normalized

def _sanitize_pdf_text(text: object, fallback: str = "") -> str:
    raw = "" if text is None else str(text)
    stripped = "".join(ch for ch in raw if ch.isprintable())
    normalized = unicodedata.normalize("NFKD", stripped)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii", "ignore")
    cleaned = ascii_only.strip()
    return cleaned or fallback

def _safe_int(value: object, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback

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

            schedule_start_time = _normalize_time_string(
                raw_settings.get("schedule_start_time"), DEFAULT_SCHEDULE_START_TIME
            )
            settings["schedule_start_time"] = schedule_start_time

            try:
                schedule_slot_minutes = int(
                    raw_settings.get("schedule_slot_minutes", DEFAULT_SCHEDULE_SLOT_MINUTES)
                )
            except (TypeError, ValueError):
                schedule_slot_minutes = DEFAULT_SCHEDULE_SLOT_MINUTES
            if schedule_slot_minutes > 0:
                settings["schedule_slot_minutes"] = schedule_slot_minutes

            schedule_stand_count = _safe_int(
                raw_settings.get("schedule_stand_count"), DEFAULT_SCHEDULE_STAND_COUNT
            )
            settings["schedule_stand_count"] = max(4, schedule_stand_count)

            settings["schedule_end_time"] = _clean_optional_time(
                raw_settings.get("schedule_end_time")
            )
    return settings


def save_app_settings(settings: dict[str, object]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)

TOP_RANK_SATURATION = {1: 0.65, 2: 0.55, 3: 0.45}
TOP_RANK_VALUE = {1: 0.9, 2: 0.94, 3: 0.97}
UNCATEGORIZED_LABEL = "Uncategorized"
PDF_FONT_REGULAR_NAME = "TPSP-Regular"
PDF_FONT_BOLD_NAME = "TPSP-Bold"

_registered_pdf_fonts: tuple[str, str] | None = None


def _candidate_font_directories() -> list[Path]:
    candidates: list[Path] = []
    for directory in (
        BASE_DIR / "static" / "fonts",
        BASE_DIR / "fonts",
        Path.home() / ".fonts",
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype/noto"),
        Path("/usr/share/fonts/truetype/freefont"),
        Path("/usr/share/fonts/truetype/msttcorefonts"),
        Path("/usr/local/share/fonts"),
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
    ):
        if directory.exists():
            candidates.append(directory)

    windows_font_dir = os.environ.get("WINDIR")
    if windows_font_dir:
        path = Path(windows_font_dir) / "Fonts"
        if path.exists():
            candidates.append(path)

    return candidates


def _resolve_font_files() -> tuple[Path | None, Path | None]:
    if not REPORTLAB_AVAILABLE:
        return None, None

    font_pairs: list[tuple[str, str | None]] = [
        ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
        ("arial.ttf", "arialbd.ttf"),
        ("Arial.ttf", "arialbd.ttf"),
        ("Arial.ttf", "Arial Bold.ttf"),
        ("Calibri.ttf", "calibrib.ttf"),
        ("LiberationSans-Regular.ttf", "LiberationSans-Bold.ttf"),
        ("NotoSans-Regular.ttf", "NotoSans-Bold.ttf"),
        ("FreeSans.ttf", "FreeSansBold.ttf"),
    ]

    directories = _candidate_font_directories()
    for regular_name, bold_name in font_pairs:
        regular_path: Path | None = None
        bold_path: Path | None = None

        for directory in directories:
            candidate = directory / regular_name
            if candidate.exists():
                regular_path = candidate
                break

        if not regular_path:
            continue

        if bold_name:
            for directory in directories:
                candidate_bold = directory / bold_name
                if candidate_bold.exists():
                    bold_path = candidate_bold
                    break

        if not bold_path:
            bold_path = regular_path

        return regular_path, bold_path

    return None, None


def _register_pdf_fonts() -> tuple[str, str]:
    global _registered_pdf_fonts
    if _registered_pdf_fonts is not None:
        return _registered_pdf_fonts

    _ensure_pdf_backend()
    regular_path, bold_path = _resolve_font_files()
    if regular_path is None:
        raise RuntimeError(
            "Nie znaleziono czcionki obsługującej polskie znaki. "
            "Skopiuj np. plik DejaVuSans.ttf do katalogu static/fonts i spróbuj ponownie."
        )

    try:
        pdfmetrics.registerFont(TTFont(PDF_FONT_REGULAR_NAME, str(regular_path)))
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Nie udało się załadować czcionki PDF z pliku {regular_path}") from exc

    if bold_path is None:
        bold_path = regular_path

    try:
        pdfmetrics.registerFont(TTFont(PDF_FONT_BOLD_NAME, str(bold_path)))
    except Exception:
        pdfmetrics.registerFont(TTFont(PDF_FONT_BOLD_NAME, str(regular_path)))

    _registered_pdf_fonts = (PDF_FONT_REGULAR_NAME, PDF_FONT_BOLD_NAME)
    return _registered_pdf_fonts

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


def _ensure_pdf_backend() -> None:
    if REPORTLAB_AVAILABLE:
        return
    message = "ReportLab library is required for PDF export. Install it with `pip install reportlab`."
    if _REPORTLAB_IMPORT_ERROR:
        raise RuntimeError(message) from _REPORTLAB_IMPORT_ERROR
    raise RuntimeError(message)


def _clean_cell_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return str(value)


@functools.lru_cache(maxsize=1)
def _get_pdf_styles() -> dict[str, ParagraphStyle]:
    _ensure_pdf_backend()
    regular_font, bold_font = _register_pdf_fonts()
    base_styles = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["title"] = ParagraphStyle(
        "PDFTitle",
        parent=base_styles["Heading2"],
        alignment=TA_LEFT,
        spaceAfter=6,
        fontName=bold_font,
        fontSize=14,
        leading=16,
    )
    styles["subtitle"] = ParagraphStyle(
        "PDFSubtitle",
        parent=base_styles["Heading4"],
        alignment=TA_LEFT,
        spaceAfter=4,
        fontName=regular_font,
        fontSize=11,
        leading=13,
    )
    styles["table_header"] = ParagraphStyle(
        "PDFTableHeader",
        parent=base_styles["Normal"],
        fontName=bold_font,
        fontSize=9,
        alignment=TA_CENTER,
        leading=10,
        splitLongWords=False,
    )
    styles["table_cell_center"] = ParagraphStyle(
        "PDFTableCellCenter",
        parent=base_styles["Normal"],
        fontName=regular_font,
        fontSize=9,
        alignment=TA_CENTER,
        leading=11,
        splitLongWords=False,
    )
    styles["table_cell_left"] = ParagraphStyle(
        "PDFTableCellLeft",
        parent=base_styles["Normal"],
        fontName=regular_font,
        fontSize=9,
        alignment=TA_LEFT,
        leading=11,
        splitLongWords=False,
    )
    styles["small_center"] = ParagraphStyle(
        "PDFSmallCenter",
        parent=styles["table_cell_center"],
        fontSize=8,
        leading=9,
        splitLongWords=False,
    )
    return styles


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    content = html.escape(text) if text else "&nbsp;"
    return Paragraph(content or "&nbsp;", style)
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
    _ensure_pdf_backend()

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

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )
    styles = _get_pdf_styles()
    regular_font, bold_font = _register_pdf_fonts()

    elements: list[object] = []
    elements.append(Paragraph("Wyniki zawodów", styles["title"]))

    subtitle_parts: list[str] = []
    if sort:
        subtitle_parts.append(f"Sortowanie: {sort}")
    if order:
        order_label = "malejąco" if order == "desc" else "rosnąco"
        subtitle_parts.append(f"Kolejność: {order_label}")
    if subtitle_parts:
        elements.append(Paragraph(", ".join(subtitle_parts), styles["subtitle"]))
    elements.append(Spacer(1, 6))

    table_data: list[list[object]] = []
    header_row = [_paragraph(column["label"], styles["table_header"]) for column in COLUMNS]
    table_data.append(header_row)

    for competitor in competitors:
        row_cells: list[object] = []
        for column in COLUMNS:
            column_name = column["name"]
            value = getattr(competitor, column_name)
            text_value = _clean_cell_value(value)
            if column_name in {"name", "lastname", "category"}:
                style = styles["table_cell_left"]
            else:
                style = styles["table_cell_center"]
            row_cells.append(_paragraph(text_value, style))
        table_data.append(row_cells)

    column_widths: list[float] = []
    for column in COLUMNS:
        name = column["name"]
        if name == "squad":
            column_widths.append(13 * mm)
        elif name == "number":
            column_widths.append(16 * mm)
        elif name == "name":
            column_widths.append(24 * mm)
        elif name == "lastname":
            column_widths.append(28 * mm)
        elif name == "category":
            column_widths.append(18 * mm)
        elif name.startswith("parcour"):
            column_widths.append(18 * mm)
        elif name == "result":
            column_widths.append(20 * mm)
        else:
            column_widths.append(17 * mm)

    total_width = sum(column_widths)
    if total_width > doc.width:
        scale = doc.width / total_width
        column_widths = [width * scale for width in column_widths]

    table = Table(table_data, colWidths=column_widths, repeatRows=1)
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#f0f0f0")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.black),
            ("ALIGN", (0, 0), (-1, 0), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("FONTNAME", (0, 1), (-1, -1), regular_font),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 9),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, 0), 6),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
            ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.lightgrey),
        ]
    )

    for idx, column in enumerate(COLUMNS):
        if column["name"] in {"name", "lastname", "category"}:
            table_style.add("ALIGN", (idx, 1), (idx, -1), "LEFT")

    for row_index, competitor in enumerate(competitors, start=1):
        color_hex = highlight_styles.get(competitor.id)
        if color_hex:
            table_style.add("BACKGROUND", (0, row_index), (-1, row_index), rl_colors.HexColor(color_hex))

    table.setStyle(table_style)
    elements.append(table)

    doc.build(elements)
    return buffer.getvalue()


def build_metrics_pdf_bytes(metrics: dict[int, list[dict[str, object]]]) -> bytes:
    _ensure_pdf_backend()
    styles = _get_pdf_styles()
    regular_font, bold_font = _register_pdf_fonts()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
    )

    total_shots = SERIES_COUNT * SHOTS_PER_SERIES
    elements: list[object] = []
    first_section = True

    for squad_number in sorted(metrics):
        parcours = metrics[squad_number]
        for parcour in parcours:
            if not first_section:
                elements.append(PageBreak())
            first_section = False

            title_text = f"Grupa {squad_number}"
            subtitle_text = f"Metryka strzelań — {parcour.get('label', '')}"
            elements.append(Paragraph(title_text, styles["title"]))
            elements.append(Paragraph(subtitle_text, styles["subtitle"]))
            elements.append(Spacer(1, 6))

            table_data: list[list[object]] = []
            header_row_top = [
                _paragraph("Numer startowy", styles["table_header"]),
                _paragraph("Nazwisko imię", styles["table_header"]),
                _paragraph("Wyniki strzelań", styles["table_header"]),
            ]
            header_row_top.extend(
                [_paragraph("", styles["table_header"]) for _ in range(total_shots - 1)]
            )
            header_row_top.append(_paragraph("Wynik", styles["table_header"]))
            header_row_top.append(_paragraph("Podpis", styles["table_header"]))
            table_data.append(header_row_top)

            header_row_second: list[object] = [
                _paragraph("", styles["table_header"]),
                _paragraph("", styles["table_header"]),
            ]
            for series_index in range(SERIES_COUNT):
                header_row_second.append(
                    _paragraph(f"Seria {series_index + 1}", styles["table_header"])
                )
                header_row_second.extend(
                    [_paragraph("", styles["table_header"]) for _ in range(SHOTS_PER_SERIES - 1)]
                )
            header_row_second.append(_paragraph("", styles["table_header"]))
            header_row_second.append(_paragraph("", styles["table_header"]))
            table_data.append(header_row_second)

            for entry in parcour.get("competitors", []):
                number_text = _clean_cell_value(entry.get("number"))
                name_text = _clean_cell_value(entry.get("name"))

                numbers_row: list[object] = [
                    _paragraph(number_text, styles["table_cell_center"]),
                    _paragraph(name_text, styles["table_cell_left"]),
                ]
                markers_row: list[object] = [
                    _paragraph("", styles["small_center"]),
                    _paragraph("", styles["small_center"]),
                ]

                shot_counter = 0
                markers_payload = entry.get("markers") or []
                for series_markers in markers_payload:
                    for marker_value in series_markers:
                        marker_text = _clean_cell_value(marker_value).strip().upper()
                        if marker_text != "X" and shot_counter < MAX_PARCOUR_SCORE:
                            shot_counter += 1
                            numbers_row.append(_paragraph(str(shot_counter), styles["small_center"]))
                        else:
                            numbers_row.append(_paragraph("", styles["small_center"]))
                        markers_row.append(_paragraph(marker_text, styles["small_center"]))

                while len(numbers_row) < 2 + total_shots:
                    numbers_row.append(_paragraph("", styles["small_center"]))
                while len(markers_row) < 2 + total_shots:
                    markers_row.append(_paragraph("", styles["small_center"]))

                numbers_row.append(_paragraph("", styles["small_center"]))
                numbers_row.append(_paragraph("", styles["small_center"]))
                markers_row.append(_paragraph("", styles["small_center"]))
                markers_row.append(_paragraph("", styles["small_center"]))

                table_data.append(numbers_row)
                table_data.append(markers_row)

            signature_row = [_paragraph("Podpis sędziów:", styles["table_cell_left"])]
            total_columns = 2 + total_shots + 2
            signature_row.extend(
                [_paragraph("", styles["table_cell_left"]) for _ in range(total_columns - 1)]
            )
            table_data.append(signature_row)

            number_width = 20 * mm
            name_width = 48 * mm
            result_width = 20 * mm
            signature_width = 30 * mm
            shot_width = 5.5 * mm
            column_widths = (
                [number_width, name_width]
                + [shot_width] * total_shots
                + [result_width, signature_width]
            )

            total_width = sum(column_widths)
            if total_width > doc.width:
                scale = doc.width / total_width
                column_widths = [width * scale for width in column_widths]
            else:
                remaining = doc.width - total_width
                if remaining > 0:
                    column_widths[1] += remaining

            table = Table(table_data, colWidths=column_widths, repeatRows=2)
            table_style = TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 1), rl_colors.HexColor("#f0f0f0")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("FONTNAME", (0, 0), (-1, 1), bold_font),
                    ("FONTNAME", (0, 2), (-1, -2), regular_font),
                    ("FONTSIZE", (0, 0), (-1, 1), 9),
                    ("FONTSIZE", (0, 2), (-1, -2), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 2),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                    ("TOPPADDING", (0, 0), (-1, 1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, 1), 6),
                    ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.lightgrey),
                ]
            )

            total_columns = len(table_data[0])
            shot_start = 2
            shot_end = shot_start + total_shots - 1
            table_style.add("SPAN", (0, 0), (0, 1))
            table_style.add("SPAN", (1, 0), (1, 1))
            table_style.add("SPAN", (shot_start, 0), (shot_end, 0))
            table_style.add("SPAN", (shot_end + 1, 0), (shot_end + 1, 1))
            table_style.add("SPAN", (shot_end + 2, 0), (shot_end + 2, 1))

            for series_index in range(SERIES_COUNT):
                start_col = shot_start + series_index * SHOTS_PER_SERIES
                end_col = start_col + SHOTS_PER_SERIES - 1
                table_style.add("SPAN", (start_col, 1), (end_col, 1))
                if series_index > 0:
                    table_style.add("LINEBEFORE", (start_col, 0), (start_col, -2), 0.5, rl_colors.black)
                table_style.add("LINEAFTER", (end_col, 0), (end_col, -2), 0.5, rl_colors.black)
                table_style.add("LEFTPADDING", (start_col, 2), (end_col, -2), 1)
                table_style.add("RIGHTPADDING", (start_col, 2), (end_col, -2), 1)

            competitor_count = len(parcour.get("competitors", []))
            for competitor_index in range(competitor_count):
                base_row = 2 + competitor_index * 2
                table_style.add("SPAN", (0, base_row), (0, base_row + 1))
                table_style.add("SPAN", (1, base_row), (1, base_row + 1))
                table_style.add("SPAN", (shot_end + 1, base_row), (shot_end + 1, base_row + 1))
                table_style.add("SPAN", (shot_end + 2, base_row), (shot_end + 2, base_row + 1))

            last_row_index = len(table_data) - 1
            table_style.add("SPAN", (0, last_row_index), (-1, last_row_index))
            table_style.add("ALIGN", (0, last_row_index), (-1, last_row_index), "LEFT")
            table_style.add("FONTNAME", (0, last_row_index), (-1, last_row_index), bold_font)
            table_style.add("LEFTPADDING", (shot_end + 1, 2), (shot_end + 2, -2), 1.5)
            table_style.add("RIGHTPADDING", (shot_end + 1, 2), (shot_end + 2, -2), 1.5)
            table_style.add("ALIGN", (1, 2), (1, -2), "LEFT")
            table_style.add(
                "BACKGROUND",
                (0, last_row_index),
                (-1, last_row_index),
                rl_colors.HexColor("#f9f9f9"),
            )

            table.setStyle(table_style)
            elements.append(table)

    if not elements:
        elements.append(Paragraph("Brak danych metryk do wyświetlenia.", styles["table_cell_left"]))

    doc.build(elements)
    return buffer.getvalue()

def build_start_list_pdf_bytes(competitors: list["Competitor"], competition_name: str) -> bytes:
    _ensure_pdf_backend()
    styles = _get_pdf_styles()
    regular_font, bold_font = _register_pdf_fonts()

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    sorted_competitors = sorted(
        competitors, key=lambda competitor: (competitor.squad, competitor.number)
    )
    elements: list[object] = []
    title_text = "Lista startowa"
    elements.append(Paragraph(title_text, styles["title"]))
    elements.append(Spacer(1, 6))

    if not sorted_competitors:
        elements.append(Paragraph("Brak zawodników do wyświetlenia.", styles["table_cell_left"]))
        doc.build(elements)
        return buffer.getvalue()

    # grupowanie po squad
    grouped: dict[int, list[Competitor]] = defaultdict(list)
    for competitor in sorted_competitors:
        grouped[competitor.squad].append(competitor)

    squad_numbers = sorted(grouped.keys())
    def build_single_group_table(squad_number: int | None) -> Table:
        members = grouped.get(squad_number, []) if squad_number is not None else []
        max_rows = max(len(members), 1)

        header_row = [
            _paragraph("Grupa", styles["table_header"]),
            _paragraph("Nr", styles["table_header"]),
            _paragraph("Nazwisko i imię", styles["table_header"]),
            _paragraph("Klasa", styles["table_header"]),
        ]
        table_data: list[list[object]] = [header_row]

        for row_idx in range(max_rows):
            if row_idx >= len(members):
                table_data.append(
                    [
                        _paragraph("", styles["table_cell_center"]),
                        _paragraph("", styles["table_cell_center"]),
                        _paragraph("", styles["table_cell_left"]),
                        _paragraph("", styles["table_cell_center"]),
                    ]
                )
                continue

            competitor = members[row_idx]
            name_combined = f"{_clean_cell_value(competitor.lastname)} {_clean_cell_value(competitor.name)}".strip()
            table_data.append(
                [
                    _paragraph(str(squad_number or ""), styles["table_cell_center"]) if row_idx == 0 else _paragraph("", styles["table_cell_center"]),
                    _paragraph(_clean_cell_value(competitor.number), styles["table_cell_center"]),
                    _paragraph(name_combined, styles["table_cell_left"]),
                    _paragraph(_clean_cell_value(competitor.category), styles["table_cell_center"]),
                ]
            )

        base_widths = [14 * mm, 16 * mm, 46 * mm, 18 * mm]
        base_total = sum(base_widths)
        target_width = (doc.width - 10 * mm) / 2  # leave space for gap
        scale = target_width / base_total if base_total else 1
        column_widths = [width * scale for width in base_widths]

        table = Table(table_data, colWidths=column_widths, repeatRows=1)
        table_style = TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#f0f0f0")),
                ("FONTNAME", (0, 0), (-1, 0), bold_font),
                ("FONTNAME", (0, 1), (-1, -1), regular_font),
                ("ALIGN", (0, 0), (-1, 0), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.lightgrey),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )

        if squad_number is not None and members:
            table_style.add("SPAN", (0, 1), (0, max_rows))
            table_style.add("ALIGN", (0, 1), (0, max_rows), "CENTER")

        table.setStyle(table_style)
        return table

    midpoint = (len(squad_numbers) + 1) // 2
    left_column = squad_numbers[:midpoint]
    right_column = squad_numbers[midpoint:]

    max_pairs = max(len(left_column), len(right_column))
    for pair_idx in range(max_pairs):
        left_squad = left_column[pair_idx] if pair_idx < len(left_column) else None
        right_squad = right_column[pair_idx] if pair_idx < len(right_column) else None

        left_table = build_single_group_table(left_squad) if left_squad is not None else ""
        right_table = build_single_group_table(right_squad) if right_squad is not None else ""
        gap_width = 10 * mm
        outer = Table(
            [[left_table, "", right_table]],
            colWidths=[(doc.width - gap_width) / 2, gap_width, (doc.width - gap_width) / 2],
        )
        outer.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("ALIGN", (2, 0), (2, 0), "CENTER"),
                ]
            )
        )
        elements.append(outer)
        elements.append(Spacer(1, 16))

    doc.build(elements)
    return buffer.getvalue()

def _build_schedule_rows(
    squads: list[int],
    start_time: str,
    slot_minutes: int,
    stand_count: int,
    rounds: int = DEFAULT_SCHEDULE_ROUNDS,
    end_time: str | None = None,
) -> list[dict[str, object]]:
    normalized_start = _normalize_time_string(start_time, DEFAULT_SCHEDULE_START_TIME)
    try:
        current_time = datetime.strptime(normalized_start, "%H:%M")
    except ValueError:
        current_time = datetime.strptime(DEFAULT_SCHEDULE_START_TIME, "%H:%M")

    normalized_end = _clean_optional_time(end_time)
    end_dt: datetime | None = None
    if normalized_end:
        try:
            end_dt = datetime.strptime(normalized_end, "%H:%M")
        except ValueError:
            end_dt = None
        else:
            if end_dt < current_time:
                end_dt = None

    slot_delta = timedelta(minutes=max(1, slot_minutes))
    stand_count = max(4, stand_count)
    rounds = max(1, rounds)

    rows: list[dict[str, object]] = []

    if end_dt:
        max_iterations = 24 * 60  # safety
        iterations = 0
        while current_time <= end_dt and iterations < max_iterations:
            assignments = ["" for _ in range(stand_count)]
            rows.append({"time": current_time.strftime("%H:%M"), "squads": assignments})
            current_time += slot_delta
            iterations += 1
    else:
        total_rows = max(1, len(set(squads)) * rounds) if squads else rounds
        for _ in range(total_rows):
            assignments = ["" for _ in range(stand_count)]
            rows.append({"time": current_time.strftime("%H:%M"), "squads": assignments})
            current_time += slot_delta

    return rows

def build_schedule_pdf_bytes(
    squads: list[int],
    start_time: str,
    slot_minutes: int,
    stand_count: int,
    rounds: int,
    title: str,
    end_time: str | None,
) -> bytes:
    _ensure_pdf_backend()
    styles = _get_pdf_styles()
    regular_font, bold_font = _register_pdf_fonts()
    rows = _build_schedule_rows(squads, start_time, slot_minutes, stand_count, rounds, end_time)

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    elements: list[object] = []
    normalized_start = _normalize_time_string(start_time, DEFAULT_SCHEDULE_START_TIME)
    subtitle = f"Start: {normalized_start}"
    elements.append(Paragraph(_sanitize_pdf_text(title, "Harmonogram"), styles["title"]))
    if subtitle:
        elements.append(Paragraph(subtitle, styles["subtitle"]))
    elements.append(Spacer(1, 6))

    if not rows:
        elements.append(Paragraph("Brak danych do wygenerowania harmonogramu.", styles["table_cell_left"]))
        doc.build(elements)
        return buffer.getvalue()

    header_cells = [_paragraph("Godzina", styles["table_header"])]
    for idx in range(stand_count):
        header_cells.append(_paragraph(f"Parkour {idx + 1}", styles["table_header"]))
    table_data: list[list[object]] = [header_cells]

    for row in rows:
        assignments = row.get("squads", [])
        row_cells = [_paragraph(row.get("time", ""), styles["table_cell_center"])]
        for idx in range(stand_count):
            value = assignments[idx] if idx < len(assignments) else ""
            row_cells.append(_paragraph(_clean_cell_value(value), styles["table_cell_center"]))
        table_data.append(row_cells)

    time_width = 20 * mm
    if stand_count > 0:
        remaining = doc.width - time_width
        stand_width = remaining / stand_count
        column_widths = [time_width] + [stand_width] * stand_count
    else:
        column_widths = [doc.width]

    table = Table(table_data, colWidths=column_widths, repeatRows=1)
    table_style = TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#f0f0f0")),
            ("FONTNAME", (0, 0), (-1, 0), bold_font),
            ("FONTNAME", (0, 1), (-1, -1), regular_font),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.3, rl_colors.lightgrey),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
    )

    table.setStyle(table_style)
    elements.append(table)

    doc.build(elements)
    return buffer.getvalue()

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
        schedule_start_time=settings.get("schedule_start_time", DEFAULT_SCHEDULE_START_TIME),
        schedule_slot_minutes=settings.get("schedule_slot_minutes", DEFAULT_SCHEDULE_SLOT_MINUTES),
        schedule_stand_count=settings.get("schedule_stand_count", DEFAULT_SCHEDULE_STAND_COUNT),
        schedule_end_time=settings.get("schedule_end_time", DEFAULT_SCHEDULE_END_TIME),
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

@app.route("/settings/schedule", methods=["POST"])
def update_schedule_settings():
    payload = request.get_json(silent=True) or {}
    settings = load_app_settings()
    start_time = _normalize_time_string(
        payload.get("start_time"), settings.get("schedule_start_time", DEFAULT_SCHEDULE_START_TIME)
    )
    slot_minutes = _safe_int(
        payload.get("slot_minutes"), settings.get("schedule_slot_minutes", DEFAULT_SCHEDULE_SLOT_MINUTES)
    )
    stand_count = _safe_int(
        payload.get("stand_count"), settings.get("schedule_stand_count", DEFAULT_SCHEDULE_STAND_COUNT)
    )
    settings["schedule_start_time"] = start_time
    settings["schedule_slot_minutes"] = max(1, slot_minutes)
    settings["schedule_stand_count"] = max(4, stand_count)
    settings["schedule_end_time"] = _clean_optional_time(payload.get("end_time"))
    save_app_settings(settings)

    return jsonify(
        {
            "schedule_start_time": settings["schedule_start_time"],
            "schedule_slot_minutes": settings["schedule_slot_minutes"],
            "schedule_stand_count": settings["schedule_stand_count"],
            "schedule_end_time": settings["schedule_end_time"],
        }
    )


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

@app.route("/start-list-pdf")
def export_start_list_pdf():
    ensure_database_ready()
    if Competitor.query.count() == 0:
        return redirect(url_for("initial_setup"))

    settings = load_app_settings()
    competition_name = settings.get("page_title", DEFAULT_SETTINGS["page_title"])
    competitors = Competitor.query.order_by(Competitor.squad, Competitor.number).all()
    pdf = build_start_list_pdf_bytes(competitors, competition_name)

    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=lista_startowa.pdf"
    return response

@app.route("/schedule-pdf")
def export_schedule_pdf():
    ensure_database_ready()
    if Competitor.query.count() == 0:
        return redirect(url_for("initial_setup"))

    settings = load_app_settings()
    competition_name = settings.get("page_title", DEFAULT_SETTINGS["page_title"])
    start_time_arg = request.args.get("start_time") or settings.get(
        "schedule_start_time", DEFAULT_SCHEDULE_START_TIME
    )
    slot_minutes_arg = request.args.get("slot_minutes")
    stand_count_arg = request.args.get("stand_count")
    end_time_arg = request.args.get("end_time") or settings.get("schedule_end_time", DEFAULT_SCHEDULE_END_TIME)

    start_time = _normalize_time_string(start_time_arg, settings.get("schedule_start_time", DEFAULT_SCHEDULE_START_TIME))
    slot_minutes = _safe_int(slot_minutes_arg, settings.get("schedule_slot_minutes", DEFAULT_SCHEDULE_SLOT_MINUTES))
    stand_count = _safe_int(stand_count_arg, settings.get("schedule_stand_count", DEFAULT_SCHEDULE_STAND_COUNT))
    end_time = _clean_optional_time(end_time_arg)

    squad_rows = Competitor.query.with_entities(Competitor.squad).distinct().all()
    squads = sorted({row[0] for row in squad_rows})
    safe_competition_name = _sanitize_pdf_text(competition_name, "Zawody")
    pdf = build_schedule_pdf_bytes(
        squads,
        start_time,
        max(1, slot_minutes),
        max(4, stand_count),
        DEFAULT_SCHEDULE_ROUNDS,
        "Harmonogram",
        end_time,
    )

    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=harmonogram.pdf"
    return response

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
