"""Regression tests for the programmatic SVG reward."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward import score_svg


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
LOW_QUALITY_CAP = 25.0
UNSAFE_CAP = 20.0


def make_svg(
    body: str,
    *,
    namespace: str | None = SVG_NAMESPACE,
    viewbox: str | None = "0 0 256 256",
    viewbox_name: str = "viewBox",
) -> str:
    """Build an SVG with no prose or Markdown outside the document."""
    namespace_attribute = "" if namespace is None else f' xmlns="{namespace}"'
    viewbox_attribute = "" if viewbox is None else f' {viewbox_name}="{viewbox}"'
    return f"<svg{namespace_attribute}{viewbox_attribute}>{body}</svg>"


def high_quality_svg(*, namespace: str = SVG_NAMESPACE) -> str:
    return make_svg(
        '<circle cx="128" cy="128" r="104" fill="#1D4ED8" '
        'stroke="#F59E0B" stroke-width="8"/>'
        '<rect x="72" y="88" width="112" height="80" rx="16" '
        'fill="#1D4ED8"/>'
        '<line x1="88" y1="128" x2="168" y2="128" '
        'stroke="#F59E0B" stroke-width="10"/>',
        namespace=namespace,
    )


class RewardTests(unittest.TestCase):
    def assert_degenerate(self, body: str, message: str) -> None:
        result = score_svg("A simple #1D4ED8 logo", make_svg(body))
        self.assertFalse(result["is_valid"], message)
        self.assertLessEqual(result["total"], LOW_QUALITY_CAP, message)

    def test_high_quality_svg_is_valid_and_scores_highly(self) -> None:
        prompt = (
            "Create a circular blue #1D4ED8 logo with a gold #F59E0B "
            "outline, a rounded rectangle, and a horizontal line."
        )
        result = score_svg(prompt, high_quality_svg())

        self.assertTrue(result["is_valid"])
        self.assertGreaterEqual(result["total"], 90.0)
        self.assertGreaterEqual(result["normalised"], 0.9)

    def test_viewbox_is_strict(self) -> None:
        body = '<circle cx="128" cy="128" r="64" fill="#1D4ED8"/>'
        correct = score_svg("A blue circle", make_svg(body))
        self.assertTrue(correct["is_valid"])

        malformed_documents = {
            "wrong dimensions": make_svg(body, viewbox="0 0 255 256"),
            "extra token": make_svg(body, viewbox="0 0 256 256 trailing"),
            "missing viewBox": make_svg(body, viewbox=None),
            "wrong attribute case": make_svg(
                body, viewbox="0 0 256 256", viewbox_name="viewbox"
            ),
        }
        for label, document in malformed_documents.items():
            with self.subTest(label=label):
                result = score_svg("A blue circle", document)
                self.assertFalse(result["is_valid"])

    def test_wrong_svg_namespace_is_invalid(self) -> None:
        prompt = "A circular #1D4ED8 and #F59E0B logo"
        correct = score_svg(prompt, high_quality_svg())
        wrong = score_svg(
            prompt,
            high_quality_svg(namespace="http://www.w3.org/svg"),
        )

        self.assertFalse(wrong["is_valid"])
        self.assertLess(wrong["sections"]["validity"], correct["sections"]["validity"])

    def test_move_only_path_is_degenerate(self) -> None:
        self.assert_degenerate(
            '<path d="M 128 128" fill="#1D4ED8"/>',
            "A move-only path draws no visible geometry",
        )

    def test_zero_radius_circle_is_degenerate(self) -> None:
        self.assert_degenerate(
            '<circle cx="128" cy="128" r="0" fill="#1D4ED8"/>',
            "A zero-radius circle draws no visible geometry",
        )

    def test_primitive_entirely_outside_canvas_is_degenerate(self) -> None:
        self.assert_degenerate(
            '<rect x="300" y="300" width="24" height="24" fill="#1D4ED8"/>',
            "A primitive wholly outside the viewBox is not visible",
        )

    def test_parent_display_none_hides_descendants(self) -> None:
        self.assert_degenerate(
            '<g display="none">'
            '<circle cx="128" cy="128" r="64" fill="#1D4ED8"/>'
            "</g>",
            "Visibility must account for hidden ancestor groups",
        )

    def test_parent_fill_opacity_zero_hides_fill_only_descendants(self) -> None:
        self.assert_degenerate(
            '<g fill-opacity="0">'
            '<circle cx="128" cy="128" r="64" fill="#1D4ED8"/>'
            "</g>",
            "Inherited fill opacity must affect descendant visibility",
        )

    def test_shapes_inside_defs_are_not_visible(self) -> None:
        self.assert_degenerate(
            '<defs><circle id="shape" cx="128" cy="128" r="64" '
            'fill="#1D4ED8"/></defs>',
            "Definitions do not render unless referenced",
        )

    def test_garbage_path_data_is_degenerate(self) -> None:
        self.assert_degenerate(
            '<path d="M 10 10 L ???" fill="#1D4ED8"/>',
            "Malformed path data must not count as visible geometry",
        )

    def test_script_is_invalid_and_hard_capped(self) -> None:
        document = make_svg(
            "<script>alert('unsafe')</script>"
            '<rect x="32" y="32" width="192" height="192" fill="#1D4ED8"/>'
        )
        result = score_svg("A blue square logo", document)

        self.assertFalse(result["is_valid"])
        self.assertLessEqual(result["total"], UNSAFE_CAP)

    def test_external_href_is_invalid_and_hard_capped(self) -> None:
        document = make_svg(
            '<defs><linearGradient id="paint" '
            'href="https://evil.example/gradient.svg#paint">'
            '<stop offset="0" stop-color="#1D4ED8"/>'
            "</linearGradient></defs>"
            '<rect x="32" y="32" width="192" height="192" fill="url(#paint)"/>'
        )
        result = score_svg("A blue square logo", document)

        self.assertFalse(result["is_valid"])
        self.assertLessEqual(result["total"], UNSAFE_CAP)

    def test_prompt_colour_match_beats_wrong_colour(self) -> None:
        prompt = "Create a logo whose main colour is #123456."
        matching = score_svg(
            prompt,
            make_svg('<rect x="32" y="32" width="192" height="192" fill="#123456"/>'),
        )
        wrong = score_svg(
            prompt,
            make_svg('<rect x="32" y="32" width="192" height="192" fill="#FEDCBA"/>'),
        )

        self.assertGreater(matching["sections"]["palette"], wrong["sections"]["palette"])
        self.assertGreater(matching["total"], wrong["total"])


if __name__ == "__main__":
    unittest.main()
