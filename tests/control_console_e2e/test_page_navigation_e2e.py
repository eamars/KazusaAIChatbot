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
        assert page.locator("#overview-service-summary-table").inner_text().strip()
        assert page.locator("#overview-readiness-table").inner_text().strip()
        assert page.locator("#overview-cognition-graph").inner_text().strip()

        _open_page(page, "services", "Services")
        page.wait_for_selector("#service-grid .service-card")
        assert page.locator("#service-grid .service-card").count() >= 3
        page.wait_for_selector(".brain-route-tile.selected")
        selected_route = page.locator(".brain-route-tile.selected")
        assert selected_route.locator("code").count() == 1
        assert selected_route.locator("code").inner_text().strip()
        assert selected_route.locator(".brain-route-meta .badge").count() == 3
        assert selected_route.evaluate(
            "element => getComputedStyle(element).boxShadow"
        ) == "none"
        runtime_box = page.locator(".brain-runtime-panel").bounding_box()
        routes_box = page.locator(".brain-routes-panel").bounding_box()
        assert runtime_box is not None
        assert routes_box is not None
        assert runtime_box["height"] < routes_box["height"]
        assert abs(runtime_box["x"] - routes_box["x"]) < 1
        assert abs(runtime_box["width"] - routes_box["width"]) < 1
        assert page.locator(".brain-route-editor").inner_text().strip()

        _open_page(page, "logs", "Live logs")
        assert page.locator("#log-table").inner_text().strip()
        assert page.locator("#log-stream-status").inner_text() != "signed out"

        _open_page(page, "debug", "Debug chat")
        assert page.locator("#debug-send").is_disabled()
        assert "current browser session" in page.locator(
            "[data-page='debug']"
        ).inner_text().lower()
        assert "Start or connect the brain service" in page.locator(
            "textarea[name='message_text']"
        ).get_attribute("placeholder")

        _open_page(page, "events", "Event monitor")
        with page.expect_response(lambda response: "/api/events" in response.url):
            page.locator("#refresh-events").click()
        assert page.locator("#event-table").inner_text().strip()
        assert "Events updated." in page.locator("#ui-notice").inner_text()

        with page.expect_response(
            lambda response: "/api/entities/character" in response.url
        ):
            _open_page(page, "character", "Character")
        assert page.locator("#ui-notice").is_hidden()
        assert page.locator("#character-profile-table").inner_text().strip()
        assert page.locator(
            "#character-cognition-state-table"
        ).inner_text().strip()
        assert page.locator("#character-self-image-table").inner_text().strip()
        assert page.locator("#character-growth-table").inner_text().strip()
        assert page.locator("#character-carry-over-table").inner_text().strip()

        _open_page(page, "users", "Users")
        assert page.locator("#user-directory-table").inner_text().strip()
        assert page.locator("#user-global-user-id").count() == 0
        assert page.locator("select#user-platform").count() == 1
        page.locator("#user-platform").select_option("qq")
        page.locator("#user-platform-user-id").fill("e2e-user")
        with page.expect_response(
            lambda response: (
                "/api/entities/users/qq/e2e-user" in response.url
                and "global_user_id" not in response.url
            )
        ):
            page.locator("#refresh-users").click()
        assert page.locator("#user-profile-table").inner_text().strip()
        assert page.locator("#user-relationship-table").inner_text().strip()
        assert page.locator("#user-cognition-state-table").inner_text().strip()
        assert page.locator("#user-memory-table").inner_text().strip()
        assert page.locator("#user-style-table").inner_text().strip()
        assert page.locator(
            "#user-conversation-progress-table"
        ).inner_text().strip()
        assert page.locator("#user-carry-over-table").inner_text().strip()

        _open_page(page, "groups", "Groups")
        assert page.locator("#group-directory-table").inner_text().strip()
        assert page.locator("#group-global-user-id").count() == 0
        assert page.locator("select#group-platform").count() == 1
        page.locator("#group-platform").select_option("debug")
        page.locator("#group-id").fill("e2e-group")
        with page.expect_response(
            lambda response: (
                "/api/entities/groups/debug/e2e-group" in response.url
                and "global_user_id" not in response.url
            )
        ):
            page.locator("#refresh-groups").click()
        assert page.locator("#group-activity-table").inner_text().strip()
        assert page.locator("#group-review-table").inner_text().strip()
        assert page.locator("#group-style-table").inner_text().strip()
        assert page.locator("#group-carry-over-table").inner_text().strip()
        assert page.locator(
            "#group-participant-progress-table"
        ).inner_text().strip()

        with page.expect_response(
            lambda response: "/api/lookups/calendar" in response.url
        ):
            _open_page(page, "calendar", "Calendar")
        assert page.locator("#calendar-status").inner_text() != "not loaded"
        assert page.locator("#calendar-summary-table").inner_text().strip()
        assert page.locator("#calendar-schedules-table").inner_text().strip()
        assert page.locator("#calendar-runs-table").inner_text().strip()
        assert page.locator(
            "#calendar-cognition-visibility-table"
        ).inner_text().strip()

        with page.expect_response(
            lambda response: "/api/lookups/background-work" in response.url
        ):
            _open_page(page, "background", "Background work")
        assert page.locator("#background-status").inner_text() != "not loaded"
        assert page.locator("#background-summary-table").inner_text().strip()
        assert page.locator("#background-jobs-table").inner_text().strip()
        assert page.locator("#background-worker-table").inner_text().strip()
        assert page.locator("#background-errors-table").inner_text().strip()
        assert page.locator("#background-delivery-table").inner_text().strip()

        with page.expect_response(lambda response: "/api/health" in response.url):
            _open_page(page, "health", "Health/cache")
        assert page.locator("#health-readiness-table").inner_text().strip()
        assert page.locator("#health-workers-table").inner_text().strip()
        assert page.locator("#health-cache-table").inner_text().strip()

        with page.expect_response(lambda response: "/api/audit" in response.url):
            _open_page(page, "audit", "Audit")
        assert page.locator("#audit-table").inner_text().strip()
        assert page.locator("#audit-view-summary").inner_text().strip()
        visible_text = page.locator("main").inner_text()
        for forbidden_text in (
            "[object Object]",
            "panel_contract",
            "projection_owner",
            "scope_order",
            "scope_summary",
            "Growth Runs Audit",
            "Event stream",
        ):
            assert forbidden_text not in visible_text

        summary = e2e_summary_writer(
            name="page_navigation_connected_states",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "pages": [
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
                if (url.includes('/api/entities/users/qq/e2e-user')) {
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
                        reason: 'No V2 relationship state matched this user.',
                        items: []
                      },
                      cognition_state: {
                        status: 'empty',
                        reason: 'No V2 cognition state matched this user.',
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
                      },
                      conversation_progress: {
                        status: 'needs_input',
                        reason: 'Select a thread scope.',
                        items: []
                      },
                      carry_over: {
                        status: 'needs_input',
                        reason: 'Select a thread scope.',
                        items: []
                      }
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                if (url.includes('/api/entities/groups/debug/e2e-group')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'unavailable',
                    owner: 'group',
                    identity: {platform: 'debug', group_id: 'e2e-group'},
                    panels: {
                      activity: {
                        status: 'empty',
                        reason: 'No group activity matched this scope.',
                        items: []
                      },
                      review: {
                        status: 'empty',
                        reason: 'No group review matched this scope.',
                        items: []
                      },
                      style: {
                        status: 'unavailable',
                        reason: 'Group style helper is unavailable.',
                        items: []
                      },
                      carry_over: {
                        status: 'empty',
                        reason: 'No promoted carry-over matched this scope.',
                        items: []
                      },
                      participant_progress: {
                        status: 'needs_input',
                        reason: 'Select an optional participant.',
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
        group_headings = page.locator(
            "[data-page='groups'] h3",
        ).all_inner_texts()
        assert "Cognition state" not in group_headings
        assert "Relationship" not in group_headings

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
                          self_concept: 'First self image line\\nSecond self image line',
                          current_growth_edges: ['joined band']
                        }]
                      },
                      cognition_state: {
                        status: 'empty',
                        items: [],
                        reason: 'none'
                      },
                      growth: {status: 'empty', items: [], reason: 'none'},
                      carry_over: {status: 'empty', items: [], reason: 'none'}
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
        assert "joined band" in self_image_text
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
                          self_concept: 'First self image paragraph\\n\\nSecond self image paragraph',
                          current_growth_edges: [
                            'She relaxed into ordinary conversation.',
                            'She answered with a warmer, steadier tone.'
                          ]
                        }]
                      },
                      cognition_state: {
                        status: 'empty',
                        items: [],
                        reason: 'none'
                      },
                      growth: {
                        status: 'available',
                        items: [
                          {
                            kind: 'identity_candidate',
                            change_kind: 'inferred_growth',
                            status: 'emerging',
                            proposed_paths: ['self_image.self_concept'],
                            base_revision_number: 0,
                            root_count: 3,
                            local_date_count: 2,
                            updated_at: '2026-06-19T03:20:12+00:00'
                          },
                          {
                            kind: 'identity_candidate',
                            change_kind: 'explicit_self_redefinition',
                            status: 'ready',
                            proposed_paths: ['boundary_profile.control_sensitivity'],
                            base_revision_number: 0,
                            root_count: 1,
                            local_date_count: 1
                          }
                        ]
                      },
                      carry_over: {status: 'empty', items: [], reason: 'none'}
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
        self_image_text = self_image.inner_text()
        assert "She relaxed into ordinary conversation." in self_image_text
        assert "She answered with a warmer, steadier tone." in self_image_text
        assert "Current self-concept" in self_image_text
        assert "Current growth edges" in self_image_text
        assert "timestamp:" not in self_image_text
        assert "summary:" not in self_image_text
        assert "synthesis count" not in self_image_text.lower()

        growth = page.locator("#character-growth-table")
        assert growth.locator(".record-card").count() == 2
        growth_text = growth.inner_text()
        assert "inferred growth candidate" in growth_text.lower()
        assert "explicit self redefinition candidate" in growth_text.lower()
        assert "self concept" in growth_text.lower()
        assert "control sensitivity" in growth_text.lower()
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


def test_style_overlay_rows_use_readable_record_cards(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
) -> None:
    """Style guidance should render semantic cards without object dumps."""

    with e2e_console() as console:
        page = e2e_browser_page(console.base_url)
        _login(page)
        page.evaluate(
            """() => {
              const originalFetch = window.fetch.bind(window);
              window.fetch = (input, init) => {
                const url = String(input);
                if (url.includes('/api/entities/groups/qq/group-1')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    owner: 'group',
                    identity: {platform: 'qq', group_id: 'group-1'},
                    panels: {
                      activity: {status: 'empty', items: [], reason: 'none'},
                      review: {status: 'empty', items: [], reason: 'none'},
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
                      carry_over: {status: 'empty', items: [], reason: 'none'},
                      participant_progress: {
                        status: 'needs_input',
                        items: [],
                        reason: 'none'
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

        _open_page(page, "groups", "Groups")
        page.locator("#group-platform").select_option("qq")
        page.locator("#group-id").fill("group-1")
        page.locator("#refresh-groups").click()
        page.locator("#group-style-table").get_by_text(
            "group_channel_style",
        ).wait_for()

        assert page.locator("#group-style-table .record-card").count() == 1
        assert page.locator("#group-style-table tr").count() == 0
        style_text = page.locator("#group-style-table").inner_text()
        assert "speech_guidelines" in style_text
        assert "scope" in style_text.lower()
        assert "confidence" in style_text.lower()
        assert "group_channel_style" in style_text
        assert "high" in style_text
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


def test_owner_panels_use_panel_specific_readable_layouts(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
    e2e_artifact_dir,
) -> None:
    """Owner panels should format state and memory rows by their meaning."""

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
                      cognition_state: {
                        status: 'available',
                        items: [
                          {
                            key: 'drives',
                            value: ['protect honest review', 'stay grounded']
                          },
                          {
                            key: 'standards',
                            value: {directness: 'high', restraint: 'steady'}
                          }
                        ]
                      },
                      growth: {status: 'empty', items: [], reason: 'none'},
                      carry_over: {status: 'empty', items: [], reason: 'none'}
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                if (url.includes('/api/entities/users/qq/platform-user-1')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    owner: 'user',
                    identity: {
                      platform: 'qq',
                      platform_user_id: 'platform-user-1',
                      global_user_id: 'global-user-001'
                    },
                    panels: {
                      profile: {
                        status: 'available',
                        items: [{
                          platform: 'qq',
                          platform_user_id: 'platform-user-1',
                          global_user_id: 'global-user-001',
                          display_name: 'Review User',
                          alias_count: 2
                        }]
                      },
                      relationship: {
                        status: 'available',
                        items: [
                          {axis: 'trust', value: 37, band: 'positive'},
                          {axis: 'familiarity', value: 68, band: 'established'}
                        ],
                        evidence_count: 2,
                        updated_at: '2026-06-19T00:00:00+00:00'
                      },
                      cognition_state: {
                        status: 'available',
                        items: [
                          {
                            key: 'goals',
                            value: [{summary: 'complete a direct review'}]
                          }
                        ]
                      },
                      memory: {
                        status: 'available',
                        items: [
                          {
                            unit_type: 'stable_pattern',
                            status: 'active',
                            fact: 'User wants product-grade UI checks.',
                            relationship_signal: 'prefers direct review',
                            subjective_appraisal: 'high operator trust',
                            updated_at: '2026-06-19T00:00:00+00:00'
                          },
                          {
                            unit_type: 'objective_fact',
                            status: 'active',
                            fact: 'User reviews every visible workflow.'
                          }
                        ]
                      },
                      style: {status: 'empty', items: [], reason: 'none'},
                      conversation_progress: {
                        status: 'needs_input',
                        items: [],
                        reason: 'Select a thread scope.'
                      },
                      carry_over: {
                        status: 'needs_input',
                        items: [],
                        reason: 'Select a thread scope.'
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

        _open_page(page, "character", "Character")
        cognition = page.locator("#character-cognition-state-table")
        cognition.get_by_text("protect honest review").wait_for()
        cognition_text = cognition.inner_text()
        assert "drives" in cognition_text.lower()
        assert "standards" in cognition_text.lower()
        assert "protect honest review" in cognition_text
        assert "directness" in cognition_text.lower()
        assert "[object Object]" not in cognition_text

        _open_page(page, "users", "Users")
        page.locator("#user-platform").select_option("qq")
        page.locator("#user-platform-user-id").fill("platform-user-1")
        page.locator("#refresh-users").click()
        page.locator("#user-memory-table").get_by_text(
            "User wants product-grade UI checks.",
        ).wait_for()
        profile_text = page.locator("#user-profile-table").inner_text()
        assert "Review User" in profile_text
        assert "global-user-001" in profile_text
        relationship_text = page.locator(
            "#user-relationship-table",
        ).inner_text()
        assert "trust" in relationship_text.lower()
        assert "37" in relationship_text
        assert "positive" in relationship_text
        assert "familiarity" in relationship_text.lower()
        assert "68" in relationship_text
        assert "affinity" not in relationship_text.lower()
        cognition_text = page.locator(
            "#user-cognition-state-table",
        ).inner_text()
        assert "complete a direct review" in cognition_text
        assert "[object Object]" not in cognition_text

        memory_cards = page.locator("#user-memory-table .record-card")
        assert memory_cards.count() == 2
        assert page.locator("#user-memory-table tr").count() == 0
        first_memory_card = memory_cards.nth(0)
        first_memory_text = first_memory_card.inner_text()
        assert "active" in first_memory_text
        assert "updated" in first_memory_text
        assert "2026-06-19T00:00:00+00:00" in first_memory_text
        assert "relationship" in first_memory_text
        assert "prefers direct review" in first_memory_text
        assert "appraisal" in first_memory_text
        assert "high operator trust" in first_memory_text
        memory_text = page.locator("#user-memory-table").inner_text()
        assert "stable pattern" in memory_text
        assert "prefers direct review" in memory_text
        assert "unit_id" not in memory_text
        assert "unit-1" not in memory_text
        user_screenshot_path = e2e_artifact_dir / "user_global_id_reference.png"
        page.screenshot(path=str(user_screenshot_path), full_page=True)
        assert user_screenshot_path.exists()

        summary = e2e_summary_writer(
            name="owner_panel_specific_layouts",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked": [
                    "native V2 state renders nested meaning",
                    "V2 relationship axes render separately from profile",
                    "memory units render one card per unit",
                ],
                "screenshot": str(user_screenshot_path),
            },
        )

    assert summary.exists()


def test_semantic_owner_surfaces_exclude_internal_projection_metadata(
    e2e_console,
    e2e_browser_page,
    e2e_summary_writer,
    e2e_artifact_dir,
) -> None:
    """Semantic panels should render meaning without projection machinery."""

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
                    panels: {
                      profile: {
                        status: 'available',
                        items: [{name: 'Panel Order', description: 'Static profile'}]
                      },
                      self_image: {status: 'empty', items: [], reason: 'none'},
                      cognition_state: {
                        status: 'available',
                        items: [{
                          key: 'goals',
                          value: [{summary: 'answer direct review clearly'}]
                        }]
                      },
                      growth: {
                        status: 'available',
                        items: [
                          {
                            kind: 'identity_candidate',
                            change_kind: 'inferred_growth',
                            status: 'emerging',
                            proposed_paths: ['self_image.self_concept'],
                            base_revision_number: 0,
                            root_count: 2,
                            local_date_count: 1
                          },
                          {
                            kind: 'identity_growth_run',
                            run_kind: 'episode',
                            disposition: 'candidate_updated',
                            lifecycle_state: 'complete',
                            latest_reason_code: 'candidate_emerging',
                            base_revision_number: 0,
                            root_count: 2
                          }
                        ]
                      },
                      carry_over: {
                        status: 'available',
                        items: [
                          {
                            kind: 'identity_growth_health',
                            state: 'waiting_for_evidence',
                            latest_reason_code: 'candidate_emerging',
                            latest_revision_number: 0,
                            latest_consumed_revision_number: 0,
                            root_count: 2,
                            local_date_count: 1
                          },
                          {
                            kind: 'identity_revision',
                            revision_number: 0,
                            revision_kind: 'seed',
                            is_current: true,
                            base_revision_number: null,
                            evidence_root_count: 0,
                            evidence_local_date_count: 0,
                            change_diff: []
                          }
                        ]
                      }
                    },
                    redaction: {model_inputs: 'excluded'}
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                if (url.includes('/api/lookups/calendar')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    panels: {
                      summary: {
                        status: 'available',
                        items: [{
                          active: 1,
                          upcoming: 1,
                          overdue: 0,
                          running: 0,
                          completed: 1,
                          failed: 0,
                          skipped: 0
                        }]
                      },
                      schedules: {
                        status: 'available',
                        items: [{
                          label: 'Daily reflection',
                          calendar_schedule_id: 'calendar-schedule-001',
                          source_llm_trace_id: 'calendar-source-trace-001',
                          trigger_kind: 'future_cognition',
                          status: 'active',
                          next_run_at: '2026-06-20T00:00:00+00:00'
                        }]
                      },
                      runs: {
                        status: 'available',
                        items: [{
                          calendar_run_id: 'calendar-run-001',
                          calendar_schedule_id: 'calendar-schedule-001',
                          source_llm_trace_id: 'calendar-run-source-trace-001',
                          run_kind: 'reflection',
                          status: 'completed',
                          scheduled_for: '2026-06-19T23:59:00+00:00',
                          completed_at: '2026-06-20T00:00:01+00:00',
                          result_summary: 'Reflection completed.'
                        }]
                      },
                      cognition_visibility: {
                        status: 'needs_input',
                        reason: 'Enter a user and channel scope.',
                        items: []
                      }
                    }
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                if (url.includes('/api/lookups/background-work')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    status: 'available',
                    panels: {
                      summary: {
                        status: 'available',
                        items: [{
                          queued: 0,
                          running: 0,
                          completed: 1,
                          failed: 1,
                          delivery_ready: 1
                        }]
                      },
                      jobs: {
                        status: 'available',
                        items: [{
                          background_work_job_id: 'job-console-001',
                          accepted_task_id: 'accepted-task-console-001',
                          source_action_attempt_id: 'action-attempt-console-001',
                          source_llm_trace_id: 'job-source-trace-001',
                          worker: 'coding_agent',
                          status: 'completed',
                          delivery_state: 'ready',
                          created_at: '2026-06-18T00:00:00+00:00',
                          completed_at: '2026-06-19T00:00:00+00:00',
                          updated_at: '2026-06-19T00:30:00+00:00'
                        }]
                      },
                      worker_activity: {
                        status: 'available',
                        items: [{
                          worker_name: 'text_artifact',
                          event_count: 3,
                          processed_count: 4,
                          succeeded_count: 3,
                          failed_count: 1,
                          skipped_count: 0,
                          deferred_count: 0,
                          last_status: 'failed'
                        }]
                      },
                      errors: {
                        status: 'available',
                        items: [{
                          worker_name: 'text_artifact',
                          background_work_job_id: 'job-error-console-001',
                          error: 'one bounded worker failure',
                          created_at: '2026-06-19T00:00:00+00:00'
                        }]
                      },
                      delivery_detail: {
                        status: 'available',
                        items: [{
                          background_work_job_id: 'job-console-001',
                          parent_llm_trace_id: 'job-source-trace-001',
                          child_llm_trace_id: 'job-child-trace-001',
                          source_background_work_job_id: 'job-console-001',
                          worker: 'coding_agent',
                          delivery_state: 'ready',
                          delivery_attempt_count: 1
                        }]
                      }
                    }
                  }), {
                    status: 200,
                    headers: {'Content-Type': 'application/json'}
                  }));
                }
                if (url.includes('/api/events')) {
                  return Promise.resolve(new Response(JSON.stringify({
                    items: [{
                      source: 'kazusa',
                      component: 'brain_service',
                      event_type: 'resource_health',
                      level: 'warning',
                      status: 'degraded',
                      duration_ms: 42,
                      error_class: 'TimeoutError',
                      error_preview: 'bounded timeout',
                      correlation_id: 'cc-req-1',
                      request_id: 'event-request-001',
                      tracking_id: 'event-tracking-001',
                      run_id: 'event-run-001',
                      trigger_id: 'event-trigger-001',
                      attempt_id: 'event-attempt-001',
                      created_at: '2026-06-19T00:00:00+00:00'
                    }],
                    facets: {
                      levels: {warning: 1},
                      statuses: {degraded: 1},
                      components: {brain_service: 1}
                    }
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
        page.locator("#character-carry-over-table").get_by_text(
            "Growth pipeline health",
        ).first.wait_for()
        growth_text = page.locator("#character-growth-table").inner_text()
        assert "inferred growth candidate" in growth_text.lower()
        assert "Latest reason: candidate emerging" in growth_text
        assert "run_id" not in growth_text
        _open_page(page, "calendar", "Calendar")
        page.locator("#refresh-calendar").click()
        page.locator("#calendar-runs-table").get_by_text(
            "Reflection completed.",
        ).wait_for()
        assert "completed" in page.locator(
            "#calendar-summary-table",
        ).inner_text().lower()
        assert page.locator("#calendar-schedules-table .record-card").count() == 1
        assert page.locator("#calendar-runs-table .record-card").count() == 1
        page.locator(
            "#calendar-schedules-table details.graph-run-reference summary"
        ).click()
        page.locator(
            "#calendar-runs-table details.graph-run-reference summary"
        ).click()
        calendar_text = page.locator("[data-page='calendar']").inner_text()
        assert "Schedule reference" in calendar_text
        assert "calendar-schedule-001" in calendar_text
        assert "calendar-source-trace-001" in calendar_text
        assert "Run reference" in calendar_text
        assert "calendar-run-001" in calendar_text
        assert "calendar-schedule-001" in calendar_text
        assert "calendar-run-source-trace-001" in calendar_text
        calendar_screenshot_path = e2e_artifact_dir / "calendar_id_references.png"
        page.screenshot(path=str(calendar_screenshot_path), full_page=True)
        assert calendar_screenshot_path.exists()

        _open_page(page, "background", "Background work")
        page.locator("#refresh-background").click()
        page.locator("#background-worker-table").get_by_text(
            "text artifact",
        ).wait_for()
        assert page.locator(
            "#background-worker-table tr",
        ).first.locator("td").all_inner_texts() == [
            "text artifact",
            "3",
            "4",
            "3",
            "1",
            "0",
            "0",
            "failed",
        ]
        assert page.locator("#background-jobs-table .record-card").count() == 1
        jobs_card = page.locator("#background-jobs-table .record-card").first
        job_reference = jobs_card.locator("details.graph-run-reference")
        assert job_reference.count() == 1
        assert job_reference.locator("summary").inner_text() == "Job reference"
        job_reference.locator("summary").click()
        jobs_text = jobs_card.inner_text()
        for expected_value in (
            "job-console-001",
            "accepted-task-console-001",
            "action-attempt-console-001",
            "job-source-trace-001",
            "coding_agent",
            "ready",
        ):
            assert expected_value in jobs_text
        assert "JOB ID" not in jobs_text
        assert page.locator(
            "#background-errors-table details.graph-run-reference"
        ).count() == 1
        page.locator(
            "#background-errors-table details.graph-run-reference summary"
        ).click()
        assert page.locator(
            "#background-delivery-table details.graph-run-reference"
        ).count() == 0
        screenshot_path = e2e_artifact_dir / "background_jobs_job_reference.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        assert screenshot_path.exists()
        assert "one bounded worker failure" in page.locator(
            "#background-errors-table",
        ).inner_text()
        assert "job-error-console-001" in page.locator(
            "#background-errors-table",
        ).inner_text()
        delivery_text = page.locator("#background-delivery-table").inner_text()
        for expected_value in (
            "job-console-001",
            "job-source-trace-001",
            "job-child-trace-001",
        ):
            assert expected_value in delivery_text
        assert "ready" in page.locator(
            "#background-delivery-table",
        ).inner_text()

        _open_page(page, "events", "Event monitor")
        page.locator("#refresh-events").click()
        page.locator("#event-table").get_by_text("resource health").wait_for()
        event_row = page.locator("#event-table tr").first
        event_row.locator("summary").click()
        event_text = event_row.inner_text()
        for expected_value in (
            "event-request-001",
            "cc-req-1",
            "event-tracking-001",
            "event-run-001",
            "event-trigger-001",
            "event-attempt-001",
        ):
            assert expected_value in event_text
        event_screenshot_path = e2e_artifact_dir / "event_id_references.png"
        page.screenshot(path=str(event_screenshot_path), full_page=True)
        assert event_screenshot_path.exists()
        assert page.locator("[data-page='events'] thead th").all_inner_texts() == [
            "TIME",
            "SEVERITY",
            "COMPONENT",
            "EVENT",
            "OUTCOME",
            "DURATION",
            "ERROR",
        ]

        _open_page(page, "audit", "Audit")
        assert page.locator("[data-page='audit'] thead th").all_inner_texts() == [
            "TIME",
            "ACTION",
            "TARGET",
            "OUTCOME",
            "OPERATOR",
            "REASON",
        ]
        visible_text = page.locator("main").inner_text()
        for forbidden in (
            "[object Object]",
            "panel_contract",
            "projection_owner",
            "scope_order",
            "scope_summary",
            "Growth Runs Audit",
            "Prompt View",
            "Operational Backing",
        ):
            assert forbidden not in visible_text

        summary = e2e_summary_writer(
            name="semantic_owner_surface_structures",
            conclusion="pass",
            details={
                "console_url": console.base_url,
                "checked": [
                    "semantic character growth and carry-over",
                    "calendar and background outcomes",
                    "event and audit tables",
                    "projection metadata absent",
                    "background Jobs job reference screenshot",
                ],
                "screenshots": {
                    "background": str(screenshot_path),
                    "event": str(event_screenshot_path),
                },
            },
        )

    assert summary.exists()


def _login(page) -> None:
    """Authenticate the browser page as the E2E operator."""

    page.locator("#token").fill(DEFAULT_E2E_OPERATOR_TOKEN)
    page.locator("#login").click()
    page.wait_for_function(
        """() => (
          document.querySelector('#overview-service-status')?.textContent
          !== 'not loaded'
        )"""
    )


def _open_page(page, page_name: str, expected_heading: str) -> None:
    """Open a sidebar page and assert the active page heading."""

    page.locator(f"[data-page-link='{page_name}']").click()
    active_page = page.locator(f"[data-page='{page_name}']")
    active_page.evaluate(
        "element => { if (!element.classList.contains('active')) throw new Error('page not active'); }"
    )
    assert active_page.locator("h2").first.inner_text() == expected_heading
