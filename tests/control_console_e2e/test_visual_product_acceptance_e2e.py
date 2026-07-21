from __future__ import annotations

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


def test_desktop_visual_acceptance_for_cards_buttons_and_branding(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify desktop visual invariants that matter for human review."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        page.set_viewport_size({"width": 1600, "height": 900})
        page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
        page.locator("#login").click()
        page.wait_for_selector("#overview-grid .metric")
        assert page.locator("#login-form").is_hidden()
        assert page.locator("#brand-name").inner_text().strip()

        pages = [
            "overview",
            "services",
            "logs",
            "debug",
            "events",
            "character",
            "users",
            "groups",
            "calendar",
            "background",
            "health",
            "audit",
        ]
        for page_name in pages:
            page.locator(f"[data-page-link='{page_name}']").click()
            page.wait_for_timeout(100)
            card_overflows = page.evaluate(
                """() => Array.from(document.querySelectorAll(
                    '[data-page].active .card, [data-page].active .metric'
                  ))
                  .filter((element) => element.scrollWidth > element.clientWidth + 1)
                  .map((element) => element.textContent.trim().slice(0, 80))"""
            )
            button_overflows = page.evaluate(
                """() => Array.from(document.querySelectorAll(
                    '[data-page].active button, .topbar button'
                  ))
                  .filter((element) => element.scrollWidth > element.clientWidth + 1)
                  .map((element) => element.textContent.trim())"""
            )
            assert card_overflows == []
            assert button_overflows == []

        page.locator("[data-page-link='users']").click()
        user_alignment = _button_input_bottom_delta(
            page,
            "#refresh-users",
            "#user-platform-user-id",
        )
        assert user_alignment <= 1
        assert page.locator("#refresh-users").inner_text() == "Search"

        page.locator("[data-page-link='groups']").click()
        group_alignment = _button_input_bottom_delta(
            page,
            "#refresh-groups",
            "#group-id",
        )
        assert group_alignment <= 1
        assert page.locator("#refresh-groups").inner_text() == "Search"

        page.locator("[data-page-link='logs']").click()
        page.evaluate(
            """() => {
              document.querySelector('#log-table').innerHTML = `
                <tr class="log-row wrap">
                  <td><code>2026-06-19T00:00:00Z</code><br>brain stderr</td>
                  <td>long production log line with Chinese text 中文 and enough content to wrap inside the message column without changing the action width</td>
                  <td><button class="btn log-copy" data-copy-log="sample" type="button">Copy</button></td>
                </tr>`;
            }"""
        )
        copy_metrics = page.locator(".log-copy").evaluate(
            """(element) => ({
              clientWidth: element.clientWidth,
              offsetWidth: element.offsetWidth,
              scrollWidth: element.scrollWidth,
              text: element.textContent,
            })"""
        )
        assert copy_metrics["text"] == "Copy"
        assert copy_metrics["offsetWidth"] == 56
        assert copy_metrics["scrollWidth"] <= copy_metrics["clientWidth"]

        assert getattr(page, "kazusa_console_messages", None) == []

        summary = e2e_summary_writer(
            name="visual_product_acceptance",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "viewport": "1600x900",
                "checked_pages": pages,
                "checked_invariants": [
                    "token field hidden after login",
                    "database identity brand is non-empty",
                    "active page cards do not horizontally overflow",
                    "visible buttons do not clip labels",
                    "live-log copy button keeps fixed width",
                ],
            },
        )

    assert summary.exists()


def _button_input_bottom_delta(page, button_selector: str, input_selector: str) -> float:
    """Return the vertical bottom-edge delta between a button and input."""

    metrics = page.evaluate(
        """([buttonSelector, inputSelector]) => {
          const button = document.querySelector(buttonSelector);
          const input = document.querySelector(inputSelector);
          const buttonRect = button.getBoundingClientRect();
          const inputRect = input.getBoundingClientRect();
          return Math.abs(buttonRect.bottom - inputRect.bottom);
        }""",
        [button_selector, input_selector],
    )
    return float(metrics)
