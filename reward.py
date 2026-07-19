"""Explainable proxy reward for detailed-prompt -> SVG logo generation.

The metric intentionally gives most weight to properties that can be checked
reliably without a vision model: a complete standalone SVG, safe/allowed
markup, drawable geometry intersecting the 256 x 256 canvas, non-degenerate
complexity, and prompt colour/structure cues.  It uses only the Python
standard library so the submitted reward is easy to reproduce.
"""

from __future__ import annotations

import math
import re
import statistics
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any


REWARD_VERSION = "3.0"
SVG_NS = "http://www.w3.org/2000/svg"
CANVAS = (0.0, 0.0, 256.0, 256.0)

SECTION_MAX = {
    "validity": 30.0,
    "safety": 15.0,
    "geometry": 15.0,
    "nondegeneracy": 15.0,
    "palette": 15.0,
    "prompt_fidelity": 10.0,
}

PRIMITIVES = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}
CONTAINERS = {"svg", "g", "defs", "clipPath", "mask"}
PAINT_SERVERS = {"linearGradient", "radialGradient"}
FILTER_TAGS = {
    "filter", "feGaussianBlur", "feDropShadow", "feColorMatrix", "feOffset",
    "feMerge", "feMergeNode", "feTurbulence", "feDisplacementMap",
    "feBlend", "feComposite", "feFlood",
}
ALLOWED_TAGS = PRIMITIVES | CONTAINERS | PAINT_SERVERS | FILTER_TAGS | {
    "stop", "title", "desc", "use",
}
DANGEROUS_TAGS = {
    "script", "image", "foreignobject", "iframe", "object", "embed",
    "audio", "video", "animate", "animatemotion", "animatetransform", "set",
}
NON_RENDERED_CONTAINERS = {"defs", "clipPath", "mask"} | PAINT_SERVERS | FILTER_TAGS

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
NUMBER_RE = re.compile(NUMBER)
LENGTH_RE = re.compile(rf"({NUMBER})(%|px)?", re.I)
HEX_RE = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{4}|[0-9a-fA-F]{3})\b")
RGB_RE = re.compile(r"rgba?\(([^)]*)\)", re.I)
SVG_RE = re.compile(r"<svg\b[\s\S]*?</svg\s*>", re.I)
OPEN_SVG_RE = re.compile(r"<svg\b", re.I)
EVENT_ATTR_RE = re.compile(r"\son[a-z]+\s*=", re.I)
URL_RE = re.compile(r"url\(\s*['\"]?([^)'\"\s]+)", re.I)
PATH_TOKEN_RE = re.compile(rf"[AaCcHhLlMmQqSsTtVvZz]|{NUMBER}")
TRANSFORM_RE = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")
VIEWBOX_RE = re.compile(rf"\s*({NUMBER})[\s,]+({NUMBER})[\s,]+({NUMBER})[\s,]+({NUMBER})\s*")

CSS_COLOURS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0),
    "green": (0, 128, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0),
    "orange": (255, 165, 0), "purple": (128, 0, 128), "pink": (255, 192, 203),
    "brown": (150, 75, 0), "gray": (128, 128, 128), "grey": (128, 128, 128),
    "navy": (0, 0, 128), "teal": (0, 128, 128), "gold": (255, 215, 0),
    "coral": (255, 127, 80), "maroon": (128, 0, 0), "olive": (128, 128, 0),
    "lime": (0, 255, 0), "aqua": (0, 255, 255), "cyan": (0, 255, 255),
    "magenta": (255, 0, 255), "silver": (192, 192, 192),
    "beige": (245, 245, 220), "ivory": (255, 255, 240), "khaki": (240, 230, 140),
}
PROMPT_COLOURS: dict[str, tuple[int, int, int]] = {
    **CSS_COLOURS,
    "cream": (255, 248, 225), "off-white": (250, 249, 246),
    "charcoal": (54, 69, 79), "mustard": (225, 173, 1), "sage": (154, 184, 122),
    "turquoise": (64, 224, 208), "amber": (255, 191, 0), "tan": (210, 180, 140),
    "peach": (255, 218, 185), "mint": (152, 255, 152), "indigo": (75, 0, 130),
    "lavender": (230, 230, 250), "burgundy": (128, 0, 32),
}

# SVG affine matrix (a, b, c, d, e, f): x'=a*x+c*y+e, y'=b*x+d*y+f.
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _length(value: str | None, percent_base: float = 256.0) -> float | None:
    if value is None:
        return None
    match = LENGTH_RE.fullmatch(value.strip())
    if not match:
        return None
    number = float(match.group(1))
    if not math.isfinite(number):
        return None
    return number * percent_base / 100.0 if match.group(2) == "%" else number


def _unit_interval(value: str | None, default: float = 1.0) -> float | None:
    if value is None:
        return default
    parsed = _length(value, 1.0)
    if parsed is None:
        return None
    return max(0.0, min(1.0, parsed))


def _style(elem: ET.Element) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in re.findall(r"([\w-]+)\s*:\s*([^;]+)", elem.attrib.get("style", "")):
        result[key.lower()] = value.strip()
    return result


def _property(elem: ET.Element, style: dict[str, str], name: str, inherited: str) -> str:
    return style.get(name, elem.attrib.get(name, inherited)).strip()


def _matrix_multiply(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (
        a * g + c * h, b * g + d * h,
        a * i + c * j, b * i + d * j,
        a * k + c * l + e, b * k + d * l + f,
    )


def _parse_transform(value: str | None) -> tuple[tuple[float, ...], bool]:
    if not value:
        return IDENTITY, False
    matrix = IDENTITY
    consumed = []
    invalid = False
    for match in TRANSFORM_RE.finditer(value):
        consumed.append(match.span())
        name = match.group(1).lower()
        raw = match.group(2)
        numbers = [float(item) for item in NUMBER_RE.findall(raw)]
        leftover = NUMBER_RE.sub("", raw)
        if leftover.strip(" ,\t\r\n"):
            invalid = True
            continue
        local: tuple[float, ...] | None = None
        if name == "matrix" and len(numbers) == 6:
            local = tuple(numbers)
        elif name == "translate" and len(numbers) in {1, 2}:
            local = (1.0, 0.0, 0.0, 1.0, numbers[0], numbers[1] if len(numbers) == 2 else 0.0)
        elif name == "scale" and len(numbers) in {1, 2}:
            local = (numbers[0], 0.0, 0.0, numbers[-1], 0.0, 0.0)
        elif name == "rotate" and len(numbers) in {1, 3}:
            angle = math.radians(numbers[0])
            rotation = (math.cos(angle), math.sin(angle), -math.sin(angle), math.cos(angle), 0.0, 0.0)
            if len(numbers) == 3:
                cx, cy = numbers[1], numbers[2]
                local = _matrix_multiply(
                    _matrix_multiply((1, 0, 0, 1, cx, cy), rotation),
                    (1, 0, 0, 1, -cx, -cy),
                )
            else:
                local = rotation
        elif name == "skewx" and len(numbers) == 1:
            local = (1.0, 0.0, math.tan(math.radians(numbers[0])), 1.0, 0.0, 0.0)
        elif name == "skewy" and len(numbers) == 1:
            local = (1.0, math.tan(math.radians(numbers[0])), 0.0, 1.0, 0.0, 0.0)
        else:
            invalid = True
        if local is not None and all(math.isfinite(number) for number in local):
            matrix = _matrix_multiply(matrix, local)
        elif local is not None:
            invalid = True
    outside = TRANSFORM_RE.sub("", value)
    if outside.strip(" ,\t\r\n") or not consumed:
        invalid = True
    return matrix, invalid


def _transform_bbox(bbox: tuple[float, float, float, float], matrix: tuple[float, ...]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    a, b, c, d, e, f = matrix
    points = [(a * x + c * y + e, b * x + d * y + f) for x in (x1, x2) for y in (y1, y2)]
    return min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points)


def _bbox_intersection(bbox: tuple[float, float, float, float], canvas: tuple[float, float, float, float] = CANVAS) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    a, b, c, d = canvas
    intersection = max(x1, a), max(y1, b), min(x2, c), min(y2, d)
    return intersection if intersection[2] >= intersection[0] and intersection[3] >= intersection[1] else None


def _bbox_union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )


def _path_geometry(data: str) -> tuple[tuple[float, float, float, float] | None, int, bool]:
    """Conservatively validate path syntax and bound explicit/control points."""
    tokens = PATH_TOKEN_RE.findall(data)
    if not tokens or PATH_TOKEN_RE.sub("", data).strip(" ,\t\r\n"):
        return None, 0, True
    arity = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "A": 7}
    points: list[tuple[float, float]] = []
    current = (0.0, 0.0)
    subpath_start = current
    command: str | None = None
    index = 0
    segments = 0
    invalid = False

    def is_command(token: str) -> bool:
        return len(token) == 1 and token.isalpha()

    while index < len(tokens):
        if is_command(tokens[index]):
            command = tokens[index]
            index += 1
            if command in "Zz":
                if current != subpath_start:
                    segments += 1
                    points.extend([current, subpath_start])
                current = subpath_start
                command = None
                continue
        if command is None:
            invalid = True
            break
        upper = command.upper()
        needed = arity.get(upper)
        if needed is None or index + needed > len(tokens) or any(is_command(token) for token in tokens[index:index + needed]):
            invalid = True
            break
        values = [float(token) for token in tokens[index:index + needed]]
        if not all(math.isfinite(value) for value in values):
            invalid = True
            break
        index += needed
        relative = command.islower()
        old = current

        def point(x: float, y: float) -> tuple[float, float]:
            return (x + old[0], y + old[1]) if relative else (x, y)

        if upper in {"M", "L", "T"}:
            end = point(values[0], values[1])
            points.extend([old, end])
            current = end
            if upper == "M":
                subpath_start = end
                command = "l" if relative else "L"
            elif end != old:
                segments += 1
        elif upper == "H":
            end = (old[0] + values[0], old[1]) if relative else (values[0], old[1])
            points.extend([old, end]); current = end
            segments += int(end != old)
        elif upper == "V":
            end = (old[0], old[1] + values[0]) if relative else (old[0], values[0])
            points.extend([old, end]); current = end
            segments += int(end != old)
        elif upper == "C":
            controls = [point(values[i], values[i + 1]) for i in (0, 2, 4)]
            points.extend([old, *controls]); current = controls[-1]
            segments += int(any(item != old for item in controls))
        elif upper in {"S", "Q"}:
            controls = [point(values[i], values[i + 1]) for i in (0, 2)]
            points.extend([old, *controls]); current = controls[-1]
            segments += int(any(item != old for item in controls))
        elif upper == "A":
            rx, ry, _, large_arc, sweep, x, y = values
            if rx < 0 or ry < 0 or large_arc not in {0.0, 1.0} or sweep not in {0.0, 1.0}:
                invalid = True
                break
            end = point(x, y)
            points.extend([
                old, end, (old[0] - rx, old[1] - ry), (old[0] + rx, old[1] + ry),
                (end[0] - rx, end[1] - ry), (end[0] + rx, end[1] + ry),
            ])
            current = end
            segments += int(end != old and rx > 0 and ry > 0)
    if invalid or segments == 0 or not points:
        return None, segments, invalid
    return (
        min(x for x, _ in points), min(y for _, y in points),
        max(x for x, _ in points), max(y for _, y in points),
    ), segments, False


def _points_geometry(value: str, polygon: bool) -> tuple[tuple[float, float, float, float] | None, bool]:
    values = [float(item) for item in NUMBER_RE.findall(value)]
    if NUMBER_RE.sub("", value).strip(" ,\t\r\n") or len(values) % 2:
        return None, True
    minimum = 6 if polygon else 4
    if len(values) < minimum or not all(math.isfinite(item) for item in values):
        return None, True
    points = list(zip(values[::2], values[1::2]))
    if len(set(points)) < (3 if polygon else 2):
        return None, False
    return (min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points)), False


def _element_geometry(elem: ET.Element) -> tuple[tuple[float, float, float, float] | None, bool, bool]:
    """Return (bbox, invalid, zero/degenerate)."""
    tag = _local_name(elem.tag)
    if tag == "circle":
        cx, cy = _length(elem.attrib.get("cx", "0")), _length(elem.attrib.get("cy", "0"))
        radius = _length(elem.attrib.get("r"))
        if None in {cx, cy, radius}:
            return None, True, False
        if radius <= 0:
            return None, False, True
        return (cx - radius, cy - radius, cx + radius, cy + radius), False, False
    if tag == "ellipse":
        cx, cy = _length(elem.attrib.get("cx", "0")), _length(elem.attrib.get("cy", "0"))
        rx, ry = _length(elem.attrib.get("rx")), _length(elem.attrib.get("ry"))
        if None in {cx, cy, rx, ry}:
            return None, True, False
        if rx <= 0 or ry <= 0:
            return None, False, True
        return (cx - rx, cy - ry, cx + rx, cy + ry), False, False
    if tag == "rect":
        x, y = _length(elem.attrib.get("x", "0")), _length(elem.attrib.get("y", "0"))
        width, height = _length(elem.attrib.get("width")), _length(elem.attrib.get("height"))
        if None in {x, y, width, height}:
            return None, True, False
        if width <= 0 or height <= 0:
            return None, False, True
        return (x, y, x + width, y + height), False, False
    if tag == "line":
        values = [_length(elem.attrib.get(name, "0")) for name in ("x1", "y1", "x2", "y2")]
        if any(value is None for value in values):
            return None, True, False
        x1, y1, x2, y2 = values
        if x1 == x2 and y1 == y2:
            return None, False, True
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)), False, False
    if tag in {"polygon", "polyline"}:
        bbox, invalid = _points_geometry(elem.attrib.get("points", ""), tag == "polygon")
        return bbox, invalid, bbox is None and not invalid
    if tag == "path":
        bbox, segments, invalid = _path_geometry(elem.attrib.get("d", ""))
        return bbox, invalid, segments == 0 and not invalid
    return None, False, False


def _parse_hex(value: str) -> tuple[int, int, int] | None:
    match = HEX_RE.fullmatch(value.strip())
    if not match:
        return None
    raw = value.lstrip("#")
    if len(raw) in {3, 4}:
        raw = "".join(char * 2 for char in raw[:3])
    else:
        raw = raw[:6]
    return tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))  # type: ignore[return-value]


def _parse_rgb(value: str) -> tuple[int, int, int] | None:
    match = RGB_RE.fullmatch(value.strip())
    if not match:
        return None
    parts = [part.strip() for part in match.group(1).split(",")]
    if len(parts) not in {3, 4}:
        return None
    result = []
    for part in parts[:3]:
        try:
            number = float(part[:-1]) * 2.55 if part.endswith("%") else float(part)
        except ValueError:
            return None
        if not math.isfinite(number) or not 0 <= number <= 255:
            return None
        result.append(round(number))
    return tuple(result)  # type: ignore[return-value]


def _paint_rgb(value: str) -> tuple[int, int, int] | None:
    lowered = value.strip().lower()
    return _parse_hex(lowered) or _parse_rgb(lowered) or CSS_COLOURS.get(lowered)


def _prompt_colours(prompt: str) -> list[tuple[int, int, int]]:
    colours = [_parse_hex(token) for token in HEX_RE.findall(prompt)]
    lowered = prompt.lower()
    for name, rgb in PROMPT_COLOURS.items():
        if re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", lowered):
            colours.append(rgb)
    return list(dict.fromkeys(colour for colour in colours if colour is not None))


def _colour_similarity(left: tuple[int, int, int], right: tuple[int, int, int]) -> float:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    return max(0.0, 1.0 - distance / 300.0)


def _analyse_scene(root: ET.Element) -> dict[str, Any]:
    id_map = {elem.attrib["id"]: elem for elem in root.iter() if elem.attrib.get("id")}
    rendered_tags: Counter[str] = Counter()
    boxes: list[tuple[float, float, float, float]] = []
    serialised: list[str] = []
    paint_values: set[str] = set()
    invalid_paints = 0
    invalid_geometry = 0
    zero_geometry = 0
    off_canvas = 0
    extreme_geometry = 0
    visible_geometry = 0
    visible_strokes = 0
    used_gradients: set[str] = set()
    invalid_local_refs = 0

    def add_paint(value: str) -> None:
        nonlocal invalid_paints, invalid_local_refs
        raw = value.strip()
        lowered = raw.lower()
        if lowered in {"none", "transparent"}:
            return
        if lowered == "currentcolor":
            lowered = "black"
        url_match = re.fullmatch(r"url\(\s*['\"]?(#[^)'\"\s]+)['\"]?\s*\)", raw, re.I)
        if url_match:
            identifier = url_match.group(1)[1:]
            server = id_map.get(identifier)
            if server is None or _local_name(server.tag) not in PAINT_SERVERS:
                invalid_local_refs += 1
                return
            used_gradients.add(identifier)
            for stop in server.iter():
                if _local_name(stop.tag) != "stop":
                    continue
                stop_style = _style(stop)
                colour = stop_style.get("stop-color", stop.attrib.get("stop-color", "black"))
                if _paint_rgb(colour) is None:
                    invalid_paints += 1
                else:
                    paint_values.add(colour.strip().lower())
            return
        if _paint_rgb(lowered) is None:
            invalid_paints += 1
        else:
            paint_values.add(lowered)

    def visit(
        elem: ET.Element,
        parent_matrix: tuple[float, ...],
        parent_hidden: bool,
        inherited_fill: str,
        inherited_stroke: str,
        inherited_opacity: float,
        inherited_fill_opacity: float,
        inherited_stroke_opacity: float,
        inherited_stroke_width: float,
        render_context: bool,
        referenced: set[str],
    ) -> None:
        nonlocal invalid_geometry, zero_geometry, off_canvas, extreme_geometry
        nonlocal visible_geometry, visible_strokes, invalid_local_refs
        tag = _local_name(elem.tag)
        style = _style(elem)
        display = style.get("display", elem.attrib.get("display", "")).strip().lower()
        visibility = style.get("visibility", elem.attrib.get("visibility", "")).strip().lower()
        own_opacity = _unit_interval(style.get("opacity", elem.attrib.get("opacity")), 1.0)
        if own_opacity is None:
            invalid_geometry += 1
            own_opacity = 0.0
        opacity = inherited_opacity * own_opacity
        hidden = parent_hidden or display == "none" or visibility in {"hidden", "collapse"} or opacity <= 0
        local_matrix, bad_transform = _parse_transform(elem.attrib.get("transform"))
        invalid_geometry += int(bad_transform)
        matrix = _matrix_multiply(parent_matrix, local_matrix)
        fill = _property(elem, style, "fill", inherited_fill)
        stroke = _property(elem, style, "stroke", inherited_stroke)
        fill_opacity = _unit_interval(
            style.get("fill-opacity", elem.attrib.get("fill-opacity")), inherited_fill_opacity
        )
        stroke_opacity = _unit_interval(
            style.get("stroke-opacity", elem.attrib.get("stroke-opacity")), inherited_stroke_opacity
        )
        stroke_width_raw = style.get("stroke-width", elem.attrib.get("stroke-width"))
        stroke_width = inherited_stroke_width if stroke_width_raw is None else _length(stroke_width_raw)
        if fill_opacity is None or stroke_opacity is None or stroke_width is None or stroke_width < 0:
            invalid_geometry += 1
            fill_opacity, stroke_opacity, stroke_width = 0.0, 0.0, 0.0

        if tag == "use" and render_context and not hidden:
            href = elem.attrib.get("href", elem.attrib.get("{http://www.w3.org/1999/xlink}href", ""))
            if not href.startswith("#") or href[1:] not in id_map or href[1:] in referenced:
                invalid_local_refs += 1
            else:
                x, y = _length(elem.attrib.get("x", "0")), _length(elem.attrib.get("y", "0"))
                if x is None or y is None:
                    invalid_geometry += 1
                else:
                    use_matrix = _matrix_multiply(matrix, (1, 0, 0, 1, x, y))
                    visit(
                        id_map[href[1:]], use_matrix, False, fill, stroke, opacity,
                        fill_opacity, stroke_opacity, stroke_width, True, referenced | {href[1:]},
                    )
            return

        if tag in PRIMITIVES and render_context:
            bbox, bad_geometry, degenerate = _element_geometry(elem)
            invalid_geometry += int(bad_geometry)
            zero_geometry += int(degenerate)
            fill_visible = fill.strip().lower() not in {"none", "transparent"} and fill_opacity > 0
            stroke_visible = stroke.strip().lower() not in {"none", "transparent"} and stroke_opacity > 0 and stroke_width > 0
            if tag == "line":
                fill_visible = False
            if not hidden and bbox is not None and (fill_visible or stroke_visible):
                transformed = _transform_bbox(bbox, matrix)
                if stroke_visible:
                    pad = stroke_width / 2.0
                    transformed = (transformed[0] - pad, transformed[1] - pad, transformed[2] + pad, transformed[3] + pad)
                visible_geometry += 1
                visible_strokes += int(stroke_visible)
                if _bbox_intersection(transformed) is None:
                    off_canvas += 1
                else:
                    clipped = _bbox_intersection(transformed)
                    boxes.append(clipped)
                    rendered_tags[tag] += 1
                    serialised.append(ET.tostring(elem, encoding="unicode"))
                    if fill_visible:
                        add_paint(fill)
                    if stroke_visible:
                        add_paint(stroke)
                covers_canvas = transformed[0] <= 0 and transformed[1] <= 0 and transformed[2] >= 256 and transformed[3] >= 256
                if not covers_canvas and any(abs(number) > 4096 for number in transformed):
                    extreme_geometry += 1

        child_render_context = render_context and tag not in NON_RENDERED_CONTAINERS
        for child in elem:
            visit(
                child, matrix, hidden, fill, stroke, opacity,
                fill_opacity, stroke_opacity, stroke_width, child_render_context, referenced,
            )

    visit(root, IDENTITY, False, "black", "none", 1.0, 1.0, 1.0, 1.0, True, set())
    union = _bbox_union(boxes)
    coverage = 0.0
    if union:
        coverage = max(0.0, union[2] - union[0]) * max(0.0, union[3] - union[1]) / (256.0 * 256.0)
    duplicate_ratio = 1.0 - len(set(serialised)) / len(serialised) if serialised else 0.0
    return {
        "rendered_tags": rendered_tags,
        "canvas_primitive_count": len(boxes),
        "visible_geometry_count": visible_geometry,
        "off_canvas_count": off_canvas,
        "invalid_geometry_count": invalid_geometry,
        "zero_geometry_count": zero_geometry,
        "extreme_geometry_count": extreme_geometry,
        "visible_stroke_count": visible_strokes,
        "coverage_ratio": coverage,
        "duplicate_ratio": duplicate_ratio,
        "paint_values": paint_values,
        "paint_rgbs": {_paint_rgb(value) for value in paint_values if _paint_rgb(value) is not None},
        "invalid_paints": invalid_paints,
        "used_gradients": used_gradients,
        "invalid_local_refs": invalid_local_refs,
    }


def _shape_fidelity(prompt: str, scene: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    lowered = prompt.lower()
    tags: Counter[str] = scene["rendered_tags"]
    checks: list[tuple[str, bool]] = []

    def mentioned(pattern: str) -> bool:
        return re.search(pattern, lowered, re.I) is not None

    if mentioned(r"\b(circle|circular|disc|medallion|round badge)\b"):
        checks.append(("circular form", tags["circle"] + tags["ellipse"] > 0))
    if mentioned(r"\b(rounded[ -]?square|square badge|rectangle|rectangular)\b"):
        checks.append(("rectangular form", tags["rect"] > 0))
    if mentioned(r"\b(dot|dots|bubble|bubbles)\b"):
        checks.append(("dots/bubbles", tags["circle"] + tags["ellipse"] >= 2))
    if mentioned(r"\b(gradient|ombre)\b"):
        checks.append(("used gradient", bool(scene["used_gradients"])))
    if mentioned(r"\b(outline|ring|border|linework|stroke)\b"):
        checks.append(("visible outline/stroke", scene["visible_stroke_count"] > 0))
    if mentioned(r"\b(ray|rays|line|lines|stem|staff)\b"):
        checks.append(("linear feature", tags["line"] + tags["polyline"] > 0 or scene["visible_stroke_count"] > 0))
    if mentioned(r"\b(triangle|triangular|diamond|star|polygon)\b"):
        checks.append(("angular feature", tags["polygon"] > 0))
    if not checks:
        return 0.5, [], []
    return (
        sum(passed for _, passed in checks) / len(checks),
        [name for name, passed in checks if not passed],
        [name for name, _ in checks],
    )


def extract_svg(text: str) -> str | None:
    if not isinstance(text, str):
        return None
    match = SVG_RE.search(text)
    return match.group(0) if match else None


def _empty_result(reason: str) -> dict[str, Any]:
    return {
        "reward_version": REWARD_VERSION,
        "total": 0.0,
        "normalised": 0.0,
        "is_valid": False,
        "sections": {name: 0.0 for name in SECTION_MAX},
        "max_sections": dict(SECTION_MAX),
        "details": {"fatal_error": reason},
    }


def score_svg(prompt: str, output: str) -> dict[str, Any]:
    """Return a 0..100 proxy score plus auditable sub-scores and diagnostics."""
    if not isinstance(output, str) or not output.strip():
        return _empty_result("empty output")
    svg = extract_svg(output)
    if svg is None:
        return _empty_result("incomplete SVG" if OPEN_SVG_RE.search(output) else "no complete SVG found")

    sections = {name: 0.0 for name in SECTION_MAX}
    details: dict[str, Any] = {}
    exact_only = output.strip() == svg
    document_count = len(SVG_RE.findall(output))
    sections["validity"] += 6.0 if exact_only else 0.0
    sections["validity"] += 2.0 if document_count == 1 else 0.0
    sections["validity"] += 2.0 if re.search(r"</svg\s*>\s*$", svg, re.I) else 0.0
    details.update({"output_only_svg": exact_only, "svg_document_count": document_count})

    forbidden_declaration = bool(re.search(r"<!DOCTYPE|<!ENTITY", svg, re.I))
    try:
        if forbidden_declaration:
            raise ET.ParseError("DOCTYPE/ENTITY declarations are forbidden")
        root = ET.fromstring(svg)
    except ET.ParseError as exc:
        sections["validity"] += 1.0
        details["parse_error"] = str(exc)
        total = round(sum(sections.values()), 3)
        return {
            "reward_version": REWARD_VERSION,
            "total": total, "normalised": round(total / 100, 5), "is_valid": False,
            "sections": sections, "max_sections": dict(SECTION_MAX), "details": details,
        }

    sections["validity"] += 10.0
    root_is_svg = _local_name(root.tag) == "svg"
    namespace_ok = root.tag == f"{{{SVG_NS}}}svg"
    sections["validity"] += 2.0 if root_is_svg else 0.0
    sections["validity"] += 3.0 if namespace_ok else 0.0
    viewbox_raw = root.attrib.get("viewBox")
    viewbox_match = VIEWBOX_RE.fullmatch(viewbox_raw or "")
    viewbox_values = tuple(float(viewbox_match.group(index)) for index in range(1, 5)) if viewbox_match else None
    exact_viewbox = viewbox_values == (0.0, 0.0, 256.0, 256.0)
    sections["validity"] += 5.0 if exact_viewbox else 0.0
    details.update({
        "xml_parseable": True, "root_is_svg": root_is_svg, "correct_svg_namespace": namespace_ok,
        "viewbox": viewbox_raw, "exact_256_viewbox": exact_viewbox,
    })

    # Safety and specification compliance.
    tags = [_local_name(elem.tag) for elem in root.iter()]
    tag_counts = Counter(tags)
    unknown_tags = sorted({tag for tag in tags if tag not in ALLOWED_TAGS})
    dangerous_tags = sorted({tag for tag in tags if tag.lower() in DANGEROUS_TAGS})
    external_refs: list[str] = []
    duplicate_ids = 0
    ids = [elem.attrib["id"] for elem in root.iter() if elem.attrib.get("id")]
    duplicate_ids = len(ids) - len(set(ids))
    unsafe_css = False
    for elem in root.iter():
        for raw_name, raw_value in elem.attrib.items():
            name = _local_name(raw_name).lower()
            value = raw_value.strip()
            if name == "href" and not value.startswith("#"):
                external_refs.append(value)
            for reference in URL_RE.findall(value):
                if not reference.startswith("#"):
                    external_refs.append(reference)
        style_text = elem.attrib.get("style", "")
        if re.search(r"@import|expression\s*\(|(?:java|vb)script:", style_text, re.I):
            unsafe_css = True
    dangerous_text = bool(EVENT_ATTR_RE.search(svg) or forbidden_declaration or unsafe_css)
    sections["safety"] += 4.0 if not unknown_tags else 0.0
    sections["safety"] += 8.0 if not dangerous_tags and not dangerous_text and not external_refs else 0.0
    sections["safety"] += 3.0 if duplicate_ids == 0 else 0.0
    details.update({
        "tag_counts": dict(tag_counts), "unknown_tags": unknown_tags, "dangerous_tags": dangerous_tags,
        "external_references": external_refs, "duplicate_ids": duplicate_ids,
    })

    scene = _analyse_scene(root)
    invalid_geometry = scene["invalid_geometry_count"] + scene["invalid_local_refs"]
    canvas_count = scene["canvas_primitive_count"]
    visible_count = scene["visible_geometry_count"]
    canvas_ratio = canvas_count / visible_count if visible_count else 0.0

    # Geometry (15).
    sections["geometry"] += 5.0 if invalid_geometry == 0 else max(0.0, 5.0 - invalid_geometry)
    sections["geometry"] += 3.0 if scene["zero_geometry_count"] == 0 else max(0.0, 3.0 - scene["zero_geometry_count"])
    sections["geometry"] += 4.0 * canvas_ratio
    sections["geometry"] += 3.0 if scene["extreme_geometry_count"] == 0 else 0.0

    # Non-degeneracy (15).
    sections["nondegeneracy"] += 5.0 if canvas_count > 0 else 0.0
    if 3 <= canvas_count <= 60:
        complexity = 3.0
    elif canvas_count in {1, 2}:
        complexity = 1.5
    elif 61 <= canvas_count <= 120:
        complexity = 2.0
    else:
        complexity = 0.0
    sections["nondegeneracy"] += complexity
    coverage = scene["coverage_ratio"]
    sections["nondegeneracy"] += 3.0 * min(1.0, coverage / 0.04) if coverage > 0 else 0.0
    sections["nondegeneracy"] += 2.0 * max(0.0, 1.0 - scene["duplicate_ratio"] * 2.0)
    sections["nondegeneracy"] += 2.0 if 100 <= len(svg) <= 20_000 else (1.0 if len(svg) <= 40_000 else 0.0)

    details.update({
        "canvas_primitive_count": canvas_count, "visible_geometry_count": visible_count,
        "off_canvas_count": scene["off_canvas_count"], "invalid_geometry_count": invalid_geometry,
        "zero_geometry_count": scene["zero_geometry_count"],
        "extreme_geometry_count": scene["extreme_geometry_count"],
        "coverage_ratio": round(coverage, 5), "duplicate_primitive_ratio": round(scene["duplicate_ratio"], 5),
        "svg_characters": len(svg), "rendered_tag_counts": dict(scene["rendered_tags"]),
    })

    # Palette fidelity and cohesion (15), using only paints on canvas geometry
    # and gradient servers actually referenced by such geometry.
    requested_colours = _prompt_colours(prompt or "")
    generated_colours = list(scene["paint_rgbs"])
    if requested_colours:
        similarities = [
            max((_colour_similarity(requested, generated) for generated in generated_colours), default=0.0)
            for requested in requested_colours
        ]
        colour_coverage = statistics.mean(similarities)
        sections["palette"] += 8.0 * colour_coverage
    else:
        colour_coverage = None
        sections["palette"] += 4.0  # Neutral score when the prompt has no checkable colour.
    colour_count = len(generated_colours)
    if 1 <= colour_count <= 8:
        cohesion = 1.0
    elif colour_count == 0:
        cohesion = 0.0
    else:
        cohesion = max(0.0, 1.0 - (colour_count - 8) / 16.0)
    sections["palette"] += 5.0 * cohesion
    sections["palette"] += 2.0 if scene["invalid_paints"] == 0 else 0.0
    details.update({
        "requested_colour_count": len(requested_colours),
        "generated_visible_paints": sorted(scene["paint_values"]),
        "visible_colour_count": colour_count,
        "colour_coverage": None if colour_coverage is None else round(colour_coverage, 5),
        "invalid_paints": scene["invalid_paints"], "used_gradients": sorted(scene["used_gradients"]),
    })

    fidelity, missing_features, checked_features = _shape_fidelity(prompt or "", scene)
    sections["prompt_fidelity"] = 10.0 * fidelity
    details.update({"checked_prompt_features": checked_features, "missing_prompt_features": missing_features})

    total = sum(sections.values())
    degenerate_ratio = scene["zero_geometry_count"] / max(1, canvas_count + scene["zero_geometry_count"])
    details["degenerate_geometry_ratio"] = round(degenerate_ratio, 5)
    if dangerous_tags or dangerous_text or external_refs:
        total = min(total, 20.0); details["hard_cap"] = "unsafe SVG"
    elif canvas_count == 0:
        total = min(total, 20.0); details["hard_cap"] = "no drawable geometry on canvas"
    elif invalid_geometry or degenerate_ratio > 0.25:
        total = min(total, 55.0); details["hard_cap"] = "invalid or degenerate geometry"
    elif not namespace_ok or not exact_viewbox or unknown_tags:
        total = min(total, 60.0); details["hard_cap"] = "SVG specification violation"
    elif not exact_only:
        total = min(total, 75.0); details["hard_cap"] = "output contains non-SVG wrapper text"
    if canvas_count > 250:
        total = min(total, 40.0); details["hard_cap"] = "element flooding"
    elif canvas_count > 120:
        total = min(total, 65.0); details["hard_cap"] = "excessive element count"

    sections = {name: round(value, 3) for name, value in sections.items()}
    total = round(max(0.0, min(100.0, total)), 3)
    is_valid = bool(
        exact_only and document_count == 1 and root_is_svg and namespace_ok and exact_viewbox
        and not unknown_tags and not dangerous_tags and not dangerous_text and not external_refs
        and duplicate_ids == 0 and invalid_geometry == 0 and degenerate_ratio <= 0.25
        and scene["invalid_paints"] == 0 and canvas_count > 0
    )
    return {
        "reward_version": REWARD_VERSION,
        "total": total, "normalised": round(total / 100.0, 5), "is_valid": is_valid,
        "sections": sections, "max_sections": dict(SECTION_MAX), "details": details,
    }


def reward(prompt: str, output: str) -> float:
    return float(score_svg(prompt, output)["normalised"])


def compute_reward(prompt: str, output: str) -> float:
    return reward(prompt, output)


__all__ = [
    "REWARD_VERSION", "SECTION_MAX", "compute_reward", "extract_svg", "reward", "score_svg",
]
