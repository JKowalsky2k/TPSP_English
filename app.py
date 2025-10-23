import json
import random
from collections import defaultdict
from colorsys import hsv_to_rgb
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from flask import Flask, render_template, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from weasyprint import HTML, CSS

BASE_DIR = Path(__file__).resolve().parent
COLUMNS_PATH = BASE_DIR / "columns.json"
DATABASE_NAME = "competitors.db"
MAX_PARCOUR_SCORE = 25
PARCOUR_FIELDS = ("parcour1", "parcour2", "parcour3", "parcour4")
DEFAULT_COMPETITORS = 66
SQUAD_SIZE = 6
SERIES_COUNT = 6
SHOTS_PER_SERIES = 5
SETTINGS_PATH = BASE_DIR / "instance" / "settings.json"
DEFAULT_SETTINGS = {"page_title": "🏆 Wyniki zawodów"}

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_NAME}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)


def _load_columns_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


columns_data = _load_columns_config(COLUMNS_PATH)
COLUMNS = columns_data["columns"]
EDITABLE_COLUMNS = [column["name"] for column in COLUMNS if column["editable"]]
PARCOUR_LABELS = [column["label"] for column in COLUMNS if column["name"] in PARCOUR_FIELDS]
TOP_RANK_SATURATION = {1: 0.65, 2: 0.55, 3: 0.45}
TOP_RANK_VALUE = {1: 0.9, 2: 0.94, 3: 0.97}
UNCATEGORIZED_LABEL = "Uncategorized"

def load_app_settings() -> Dict[str, str]:
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_PATH.exists():
        try:
            with SETTINGS_PATH.open(encoding="utf-8") as handle:
                raw_settings = json.load(handle)
        except (OSError, json.JSONDecodeError):
            raw_settings = {}
        if isinstance(raw_settings, dict):
            for key, value in raw_settings.items():
                if isinstance(value, str):
                    settings[key] = value
                elif value is not None:
                    settings[key] = str(value)
    return settings


def save_app_settings(settings: Dict[str, str]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(settings, handle, ensure_ascii=False, indent=2)


def _rgb_float_to_hex(rgb: Sequence[float]) -> str:
    red, green, blue = (
        max(0, min(255, int(round(component * 255)))) for component in rgb
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def generate_category_rank_colors(categories: Iterable[str]) -> Dict[str, Dict[int, str]]:
    category_list = sorted(categories)
    if not category_list:
        return {}

    colors: Dict[str, Dict[int, str]] = {}
    hue_step = 0.6180339887498949  # irrational number for pleasant hue distribution
    for idx, category in enumerate(category_list):
        hue = (idx * hue_step) % 1.0
        rank_colors: Dict[int, str] = {}
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

    def to_dict(self) -> Dict[str, Any]:
        return {column["name"]: getattr(self, column["name"]) for column in COLUMNS}

def clamp_score(value: Any) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(MAX_PARCOUR_SCORE, score))

def init_db():
    db.create_all()
    if Competitor.query.count() == 0:
        competitors = []
        random.seed()
        for idx in range(DEFAULT_COMPETITORS):
            squad_number = idx // SQUAD_SIZE + 1
            competitors.append(
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
        db.session.add_all(competitors)
        db.session.commit()


def fetch_competitors(sort: str, order: str) -> List[Competitor]:
    requested_sort = (sort or "number").lower()
    requested_order = (order or "asc").lower()

    if requested_sort == "result":
        competitors = sorted(
            Competitor.query.all(),
            key=lambda competitor: competitor.result,
            reverse=requested_order == "desc",
        )
    elif requested_sort == "category_result":
        competitors_by_category: Dict[str, List[Competitor]] = defaultdict(list)
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

def group_competitors_by_squad() -> Dict[int, List[Competitor]]:
    squads: Dict[int, List[Competitor]] = defaultdict(list)
    for competitor in Competitor.query.order_by(Competitor.squad, Competitor.number).all():
        squads[competitor.squad].append(competitor)
    return dict(sorted(squads.items()))

def build_metrics_data() -> Dict[int, List[Dict[str, Any]]]:
    squads = group_competitors_by_squad()
    metrics: Dict[int, List[Dict[str, Any]]] = {}

    for squad_number, members in squads.items():
        row_count = max(SQUAD_SIZE, len(members))
        squad_metrics: List[Dict[str, Any]] = []

        for field_index, (field, label) in enumerate(zip(PARCOUR_FIELDS, PARCOUR_LABELS)):
            entries: List[Dict[str, Any]] = []

            if members:
                shift = field_index % len(members)
                if shift:
                    rotated_members = members[-shift:] + members[:-shift]
                else:
                    rotated_members = list(members)
            else:
                rotated_members = []

            for idx, competitor in enumerate(rotated_members):
                target_series_index = max(
                    0, min(SERIES_COUNT - 1, row_count - idx - 1)
                )
                markers_by_series = []
                for series_idx in range(SERIES_COUNT):
                    if series_idx == target_series_index:
                        markers_by_series.append(["X"] * SHOTS_PER_SERIES)
                    else:
                        markers_by_series.append([""] * SHOTS_PER_SERIES)

                display_name = f"{competitor.lastname} {competitor.name}".strip()
                entries.append(
                    {
                        "number": competitor.number,
                        "name": display_name,
                        "markers": markers_by_series,
                    }
                )

            empty_markers = [["" for _ in range(SHOTS_PER_SERIES)] for _ in range(SERIES_COUNT)]
            while len(entries) < row_count:
                entries.append(
                    {
                        "number": "",
                        "name": "",
                        "markers": [row[:] for row in empty_markers],
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

@app.route("/")
def index():
    sort = request.args.get("sort", "number")
    order = request.args.get("order", "asc")
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

    return jsonify(competitor.to_dict())


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
    # sort by result descending
    competitors = fetch_competitors("result", "desc")
    return render_template("live.html", competitors=competitors, columns=COLUMNS, getattr=getattr)

@app.route("/live-data")
def live_data():
    sort = request.args.get("sort", "result")
    order = request.args.get("order", "desc")
    competitors = fetch_competitors(sort, order)
    data = [competitor.to_dict() for competitor in competitors]
    return jsonify(data)

@app.route("/export-pdf")
def export_pdf():
    sort = request.args.get("sort", "result")
    order = request.args.get("order", "desc")
    competitors = fetch_competitors(sort, order)
    category_groups: Dict[str, List[Competitor]] = defaultdict(list)
    competitor_categories: Dict[int, str] = {}
    for competitor in competitors:
        category = competitor.category or UNCATEGORIZED_LABEL
        category_groups[category].append(competitor)
        competitor_categories[competitor.id] = category

    rank_map: Dict[int, int] = {}
    for category, members in category_groups.items():
        ranked_members = sorted(members, key=lambda competitor: competitor.result, reverse=True)
        for position, competitor in enumerate(ranked_members[:3], start=1):
            rank_map[competitor.id] = position

    category_colors = generate_category_rank_colors(category_groups.keys())
    highlight_styles: Dict[int, str] = {}
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

    # PDF in A4 portrait format
    pdf = HTML(string=rendered, base_url=str(BASE_DIR)).write_pdf(
        stylesheets=[
            CSS(filename=str(BASE_DIR / "static" / "pdf.css")),
            CSS(string='@page { size: A4 portrait; margin: 10mm }'),
        ]
    )

    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=results.pdf'
    return response

@app.route("/metrics-pdf")
def export_metrics_pdf():
    metrics = build_metrics_data()
    rendered = render_template(
        "metrics.html",
        metrics=metrics,
        series_range=range(1, SERIES_COUNT + 1),
        shots_range=range(1, SHOTS_PER_SERIES + 1),
        shots_per_series=SHOTS_PER_SERIES,
        total_columns=4 + SERIES_COUNT * SHOTS_PER_SERIES,
    )

    pdf = HTML(string=rendered, base_url=str(BASE_DIR)).write_pdf(
        stylesheets=[
            CSS(filename=str(BASE_DIR / "static" / "metrics.css")),
            CSS(string='@page { size: A4 landscape; margin: 10mm }'),
        ]
    )

    response = make_response(pdf)
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=metrics.pdf"
    return response


if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True)
