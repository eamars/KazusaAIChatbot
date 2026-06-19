from __future__ import annotations

from browser_harness import DEFAULT_E2E_OPERATOR_TOKEN


def test_each_sidebar_page_has_connected_or_explicitly_gated_state(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Verify each sidebar page activates and exposes connected/gated content."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)

        _open_page(page, "overview", "Overview")
        assert page.locator("#overview-grid .metric").count() > 0
        assert page.locator("#overview-cognition-graph").inner_text().strip()

        _open_page(page, "services", "Services")
        assert page.locator("#service-grid .service-card").count() >= 3

        _open_page(page, "debug", "Debug chat")
        assert page.locator("#debug-send").is_disabled()
        assert "Start or connect the brain service" in page.locator(
            "textarea[name='message_text']"
        ).get_attribute("placeholder")

        _open_page(page, "events", "Event monitor")
        with page.expect_response(lambda response: "/api/events" in response.url):
            page.locator("#refresh-events").click()
        assert page.locator("#event-table").inner_text().strip()

        with page.expect_response(
            lambda response: "/api/entities/character" in response.url
        ):
            _open_page(page, "character", "Character")
        assert page.locator("#character-profile-table").inner_text().strip()
        assert page.locator("#character-state-table").inner_text().strip()
        assert page.locator("#character-self-image-table").inner_text().strip()
        assert page.locator("#character-growth-table").inner_text().strip()
        assert page.locator("#character-memory-table").count() == 0

        _open_page(page, "users", "Users")
        assert page.locator("#users-status").inner_text() == "needs input"
        assert page.locator("#user-global-user-id").count() == 0
        assert page.locator("select#user-platform").count() == 1
        page.locator("#user-platform").select_option("qq")
        page.locator("#user-platform-user-id").fill("e2e-user")
        with page.expect_response(
            lambda response: (
                "/api/entities/user" in response.url
                and "platform=qq" in response.url
                and "platform_user_id=e2e-user" in response.url
                and "global_user_id" not in response.url
            )
        ):
            page.locator("#refresh-users").click()
        assert page.locator("#user-profile-table").inner_text().strip()
        assert page.locator("#user-memory-table").inner_text().strip()
        assert page.locator("#user-style-table").inner_text().strip()

        _open_page(page, "groups", "Groups")
        assert page.locator("#groups-status").inner_text() == "needs input"
        assert page.locator("#group-global-user-id").count() == 0
        assert page.locator("select#group-platform").count() == 1
        page.locator("#group-platform").select_option("debug")
        page.locator("#group-id").fill("e2e-group")
        with page.expect_response(
            lambda response: (
                "/api/entities/group" in response.url
                and "platform=debug" in response.url
                and "group_id=e2e-group" in response.url
                and "global_user_id" not in response.url
            )
        ):
            page.locator("#refresh-groups").click()
        assert page.locator("#group-style-table").inner_text().strip()
        assert page.locator("#group-progress-table").count() == 0
        assert page.locator("#group-guidance-table").count() == 0

        with page.expect_response(
            lambda response: "/api/lookups/calendar" in response.url
        ):
            _open_page(page, "calendar", "Calendar")
        assert page.locator("#calendar-status").inner_text() != "not loaded"
        assert page.locator("#calendar-table").inner_text().strip()

        with page.expect_response(
            lambda response: "/api/lookups/background" in response.url
        ):
            _open_page(page, "background", "Background work")
        assert page.locator("#background-status").inner_text() != "not loaded"
        assert page.locator("#background-table").inner_text().strip()

        _open_page(page, "health", "Health/cache")
        assert page.locator("#health-brain-status").inner_text() != "locked"
        assert page.locator("#health-runtime-table").inner_text().strip()

        with page.expect_response(lambda response: "/api/audit" in response.url):
            _open_page(page, "audit", "Audit")
        assert page.locator("#audit-table").inner_text().strip()

        summary = e2e_summary_writer(
            name="page_navigation_connected_states",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "pages": [
                    "overview",
                    "services",
                    "debug",
                    "events",
                    "character",
                    "users",
                    "groups",
                    "calendar",
                    "background",
                    "health",
                    "audit",
                ],
            },
        )

    assert summary.exists()


def test_owner_entity_unavailable_panels_render_reasons(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Unavailable owner panels should show reasons, not generic success rows."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/entities/user')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'unavailable',
                    owner: 'user',
                    identity: {platform: 'qq', platform_user_id: 'e2e-user'},
                    panels: {
                      profile: {
                        status: 'empty',
                        reason: 'No profile row matched this platform user.',
                        items: []
                      },
                      relationship: {
                        status: 'empty',
                        reason: 'No relationship summary matched this user.',
                        items: []
                      },
                      memory: {
                        status: 'empty',
                        reason: 'No user memory rows matched this lookup.',
                        items: []
                      },
                      style: {
                        status: 'unavailable',
                        reason: 'User style helper is unavailable.',
                        items: []
                      }
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                if (url.includes('/api/entities/group')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'unavailable',
                    owner: 'group',
                    identity: {platform: 'debug', group_id: 'e2e-group'},
                    panels: {
                      style: {
                        status: 'unavailable',
                        reason: 'Group style helper is unavailable.',
                        items: []
                      },
                      progress: {
                        status: 'unavailable',
                        reason: 'Group progress source is unavailable.',
                        items: []
                      },
                      guidance: {
                        status: 'unavailable',
                        reason: 'Reflection guidance source is unavailable.',
                        items: []
                      }
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                return originalFetch(input, init);
              };
            }"""
        )

        _open_page(page, "users", "Users")
        page.locator("#user-platform").select_option("qq")
        page.locator("#user-platform-user-id").fill("e2e-user")
        page.locator("#refresh-users").click()
        page.wait_for_selector("#users-status:text('unavailable')")
        assert "User style helper is unavailable." in page.locator(
            "#user-style-table",
        ).inner_text()
        assert "No user style guidance rows are available." not in page.locator(
            "#user-style-table",
        ).inner_text()

        _open_page(page, "groups", "Groups")
        page.locator("#group-platform").select_option("debug")
        page.locator("#group-id").fill("e2e-group")
        page.locator("#refresh-groups").click()
        page.wait_for_selector("#groups-status:text('unavailable')")
        assert "Group style helper is unavailable." in page.locator(
            "#group-style-table",
        ).inner_text()
        assert page.locator("#group-guidance-table").count() == 0

        summary = e2e_summary_writer(
            name="owner_entity_unavailable_panel_states",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked_panels": [
                    "user-style",
                    "group-style",
                ],
            },
        )

    assert summary.exists()


def test_owner_lookup_tables_render_nested_values_readably(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Nested DB-shaped values should not render as object placeholders."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/entities/character')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    owner: 'character',
                    identity: {character_name: 'Nested Test'},
                    panels: {
                      profile: {
                        status: 'available',
                        items: [{
                          name: 'Nested Test',
                          description: 'First profile line\\nSecond profile line\\n\\nThird profile line',
                          personality_brief: {
                            core: 'quiet',
                            traits: ['observant', 'reserved']
                          }
                        }]
                      },
                      self_image: {
                        status: 'available',
                        items: [{
                          historical_summary: 'First self image line\\nSecond self image line',
                          milestones: [
                            {title: 'joined band', year: '2026'}
                          ]
                        }]
                      },
                      state: {status: 'empty', items: [], reason: 'none'},
                      growth: {status: 'empty', items: [], reason: 'none'},
                      memory: {status: 'empty', items: [], reason: 'none'},
                      learning: {status: 'empty', items: [], reason: 'none'}
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                return originalFetch(input, init);
              };
            }"""
        )

        _open_page(page, "character", "Character")
        page.locator("#character-profile-table").get_by_text(
            "Nested Test",
        ).wait_for()

        profile_text = page.locator("#character-profile-table").inner_text()
        self_image_text = page.locator("#character-self-image-table").inner_text()
        combined_text = f"{profile_text}\n{self_image_text}"
        assert "[object Object]" not in combined_text
        assert "core" in profile_text.lower()
        assert "quiet" in profile_text
        assert "traits" in profile_text.lower()
        assert "observant" in profile_text
        assert "reserved" in profile_text
        assert "First profile line" in profile_text
        assert "Second profile line" in profile_text
        assert "First self image line" in self_image_text
        assert "Second self image line" in self_image_text
        assert "title: joined band" in self_image_text
        for selector, expected_text in (
            ("#character-profile-table .character-prose", "First profile line"),
            ("#character-self-image-table .character-prose", "First self image line"),
        ):
            locator = page.locator(selector).filter(has_text=expected_text).first
            locator.wait_for()
            white_space = locator.evaluate(
                "element => getComputedStyle(element).whiteSpace",
            )
            assert white_space == "pre-line"

        summary = e2e_summary_writer(
            name="owner_lookup_nested_value_rendering",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked": [
                    "nested object value",
                    "array of strings",
                    "array of objects",
                ],
            },
        )

    assert summary.exists()


def test_character_page_uses_readable_profile_self_image_and_growth_panels(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Character prose and traits should not render as raw key-value tables."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/entities/character')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    owner: 'character',
                    identity: {character_name: 'Panel Test'},
                    panels: {
                      profile: {
                        status: 'available',
                        items: [{
                          name: 'Panel Test',
                          description: 'First readable profile line\\nSecond readable profile line',
                          gender: 'female',
                          age: 15,
                          birthday: 'Aug 5',
                          personality_brief: {
                            logic: 'leads with emotional anchors',
                            tempo: 'quiet until interested',
                            defense: 'uses distance to hide uncertainty'
                          },
                          updated_at: '2026-06-19T03:20:12+00:00'
                        }]
                      },
                      self_image: {
                        status: 'available',
                        items: [{
                          historical_summary: 'First self image paragraph\\n\\nSecond self image paragraph',
                          recent_window: [
                            {
                              timestamp: '2026-06-19T01:21:23+00:00',
                              summary: 'She relaxed into ordinary conversation.'
                            },
                            {
                              timestamp: '2026-06-19T03:20:12+00:00',
                              summary: 'She answered with a warmer, steadier tone.'
                            }
                          ],
                          last_updated: '2026-06-19T03:20:12+00:00',
                          synthesis_count: 1685
                        }]
                      },
                      state: {status: 'empty', items: [], reason: 'none'},
                      growth: {
                        status: 'available',
                        items: [
                          {
                            trait_id: 'gct_raw_identifier',
                            growth_axis: 'clarity',
                            trait_name: 'Clear intent handling',
                            guidance: 'Ask one grounded follow-up before guessing.',
                            strength: 0.36272249999999995,
                            status: 'active',
                            maturity_band: 'emerging',
                            evidence_count: 3,
                            updated_at: '2026-06-19T03:20:12+00:00'
                          },
                          {
                            trait_id: 'gct_second_identifier',
                            growth_axis: 'boundary_timing',
                            trait_name: 'Boundary timing',
                            guidance: 'Hold the refusal posture without overexplaining.',
                            strength: 0.27,
                            status: 'active',
                            maturity_band: 'observed',
                            evidence_count: 2
                          }
                        ]
                      },
                      memory: {status: 'empty', items: [], reason: 'none'},
                      learning: {status: 'empty', items: [], reason: 'none'}
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                return originalFetch(input, init);
              };
            }"""
        )

        _open_page(page, "character", "Character")
        page.locator("#character-profile-table").get_by_text("Panel Test").wait_for()

        profile = page.locator("#character-profile-table")
        assert profile.locator(".character-title").inner_text() == "Panel Test"
        assert profile.locator(".detail-chip").count() >= 3
        profile_body = profile.locator(".character-prose").filter(
            has_text="First readable profile line",
        ).first
        profile_body.wait_for()
        assert profile_body.evaluate(
            "element => getComputedStyle(element).fontWeight",
        ) not in {"600", "700", "bold"}
        assert profile_body.evaluate(
            "element => getComputedStyle(element).whiteSpace",
        ) == "pre-line"

        self_image = page.locator("#character-self-image-table")
        assert self_image.locator(".timeline-item").count() == 2
        self_image_text = self_image.inner_text()
        assert "She relaxed into ordinary conversation." in self_image_text
        assert "She answered with a warmer, steadier tone." in self_image_text
        assert "Long-term self-image" in self_image_text
        assert "Historical summary" not in self_image_text
        assert self_image_text.index("Recent window") < self_image_text.index(
            "Long-term self-image"
        )
        assert "timestamp:" not in self_image_text
        assert "summary:" not in self_image_text

        growth = page.locator("#character-growth-table")
        assert growth.locator(".trait-card").count() == 2
        growth_text = growth.inner_text()
        assert "Clear intent handling" in growth_text
        assert "Ask one grounded follow-up before guessing." in growth_text
        assert "0.363" in growth_text
        assert "0.36272249999999995" not in growth_text
        assert "trait_id" not in growth_text
        assert "trait id" not in growth_text.lower()
        assert "gct_raw_identifier" not in growth_text
        assert growth.locator("tr").count() == 0

        summary = e2e_summary_writer(
            name="character_readable_profile_self_image_growth",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked": [
                    "profile prose normal weight",
                    "recent-window timeline entries",
                    "growth trait cards without raw trait ids",
                ],
            },
        )

    assert summary.exists()


def test_style_overlay_rows_use_readable_two_column_layout(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Style guidance should not be squeezed into four narrow columns."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/entities/group')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    owner: 'group',
                    identity: {platform: 'qq', group_id: 'group-1'},
                    panels: {
                      style: {
                        status: 'available',
                        items: [{
                          scope: 'group_channel_style',
                          field: 'speech_guidelines',
                          guidelines: [
                            'keep the technical topic visible',
                            'avoid turning the thread into one-line noise'
                          ],
                          confidence: 'high'
                        }]
                      },
                      progress: {status: 'empty', items: [], reason: 'none'},
                      guidance: {status: 'empty', items: [], reason: 'none'}
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                return originalFetch(input, init);
              };
            }"""
        )

        _open_page(page, "groups", "Groups")
        page.locator("#group-platform").select_option("qq")
        page.locator("#group-id").fill("group-1")
        page.locator("#refresh-groups").click()
        page.locator("#group-style-table").get_by_text(
            "group_channel_style",
        ).wait_for()

        row_count = page.locator("#group-style-table tr").count()
        assert row_count == 1
        for index in range(row_count):
            row = page.locator("#group-style-table tr").nth(index)
            cell_count = row.locator("td").count()
            assert cell_count <= 2
        style_text = page.locator("#group-style-table").inner_text()
        assert "Guidance" in style_text
        assert "Scope" not in style_text
        assert "Confidence" not in style_text
        assert "group_channel_style" in style_text
        assert "confidence: high" in style_text
        assert "keep the technical topic visible" in style_text
        assert "[object Object]" not in style_text

        summary = e2e_summary_writer(
            name="style_overlay_two_column_layout",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked": "group style rows use detail labels instead of four-column table cells",
            },
        )

    assert summary.exists()


def test_owner_tables_use_panel_specific_readable_layouts(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Owner tables should format state and memory rows by their meaning."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/entities/character')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    owner: 'character',
                    identity: {character_name: 'Layout Test'},
                    panels: {
                      profile: {status: 'empty', items: [], reason: 'none'},
                      self_image: {status: 'empty', items: [], reason: 'none'},
                      state: {
                        status: 'available',
                        items: [
                          {key: 'mood', value: 'focused'},
                          {key: 'global_vibe', value: 'steady'}
                        ]
                      },
                      growth: {status: 'empty', items: [], reason: 'none'},
                      memory: {status: 'empty', items: [], reason: 'none'},
                      learning: {status: 'empty', items: [], reason: 'none'}
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                if (url.includes('/api/entities/user')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    owner: 'user',
                    identity: {
                      platform: 'qq',
                      platform_user_id: 'platform-user-1'
                    },
                    panels: {
                      profile: {status: 'empty', items: [], reason: 'none'},
                      relationship: {
                        status: 'available',
                        items: [
                          {key: 'affinity', value: 742},
                          {key: 'relationship_summary', value: 'trusts direct review'}
                        ]
                      },
                      memory: {
                        status: 'available',
                        items: [
                          {
                            unit_id: 'unit-1',
                            unit_type: 'stable_pattern',
                            status: 'active',
                            fact: 'User wants product-grade UI checks.',
                            relationship_signal: 'prefers direct review',
                            subjective_appraisal: 'high operator trust',
                            updated_at: '2026-06-19T00:00:00+00:00'
                          },
                          {
                            unit_id: 'unit-2',
                            unit_type: 'objective_fact',
                            status: 'active',
                            fact: 'User reviews every visible workflow.'
                          }
                        ]
                      },
                      style: {status: 'empty', items: [], reason: 'none'}
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                return originalFetch(input, init);
              };
            }"""
        )

        _open_page(page, "character", "Character")
        page.locator("#character-state-table").get_by_text("focused").wait_for()
        state_rows = page.locator("#character-state-table tr")
        assert state_rows.count() == 2
        assert state_rows.nth(0).locator("td").nth(0).inner_text() == "mood"
        assert state_rows.nth(0).locator("td").nth(1).inner_text() == "focused"
        state_text = page.locator("#character-state-table").inner_text()
        assert "key" not in state_text.lower()
        assert "value" not in state_text.lower()

        _open_page(page, "users", "Users")
        page.locator("#user-platform").select_option("qq")
        page.locator("#user-platform-user-id").fill("platform-user-1")
        page.locator("#refresh-users").click()
        page.locator("#user-memory-table").get_by_text(
            "User wants product-grade UI checks.",
        ).wait_for()
        profile_text = page.locator("#user-profile-table").inner_text()
        assert "affinity" in profile_text
        assert "742" in profile_text
        assert "relationship summary" in profile_text
        assert "trusts direct review" in profile_text
        assert page.locator("#user-relationship-table").count() == 0
        user_profile_value = page.locator("#user-profile-table .table-primary").filter(
            has_text="trusts direct review",
        ).first
        user_profile_value.wait_for()
        assert user_profile_value.evaluate(
            "element => getComputedStyle(element).whiteSpace",
        ) == "pre-line"

        memory_rows = page.locator("#user-memory-table tr")
        assert memory_rows.count() == 2
        for index in range(memory_rows.count()):
            assert memory_rows.nth(index).locator("td").count() == 2
        first_memory_cells = memory_rows.nth(0).locator("td")
        first_memory_left = first_memory_cells.nth(0).inner_text()
        first_memory_right = first_memory_cells.nth(1).inner_text()
        assert "active" in first_memory_left
        assert "updated: 2026-06-19T00:00:00+00:00" in first_memory_left
        assert "updated: 2026-06-19T00:00:00+00:00" not in first_memory_right
        assert "relationship: prefers direct review" in first_memory_right
        assert "appraisal: high operator trust" in first_memory_right
        memory_text = page.locator("#user-memory-table").inner_text()
        assert "stable pattern" in memory_text
        assert "prefers direct review" in memory_text
        assert "unit_id" not in memory_text
        assert "unit-1" not in memory_text

        summary = e2e_summary_writer(
            name="owner_table_panel_specific_layouts",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked": [
                    "key-value state rows use field labels",
                    "memory units render one row per unit",
                ],
            },
        )

    assert summary.exists()


def _login(page) -> None:
    """Authenticate the browser page as the E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_selector("body[data-auth-state='authenticated']")
    page.wait_for_selector("#overview-grid .metric")


def _open_page(page, page_name: str, expected_heading: str) -> None:
    """Open a sidebar page and assert the active page heading."""

    page.locator(f"[data-page-link='{page_name}']").click()
    active_page = page.locator(f"[data-page='{page_name}']")
    active_page.evaluate(
        "element => { if (!element.classList.contains('active')) throw new Error('page not active'); }"
    )
    assert active_page.locator("h2").first.inner_text() == expected_heading
