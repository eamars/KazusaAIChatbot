from issue_tracker import api
from issue_tracker.store import InMemoryIssueStore


def test_api_delete_removes_issue() -> None:
    store = InMemoryIssueStore()
    issue = api.create_issue(store, "Fix import")

    deleted = api.delete_issue(store, issue.id)

    assert deleted is True
    assert api.get_issue(store, issue.id) is None


def test_api_list_returns_created_issues() -> None:
    store = InMemoryIssueStore()
    first = api.create_issue(store, "Fix import")
    second = api.create_issue(store, "Update docs")

    issues = api.list_issues(store)

    assert [issue.id for issue in issues] == [first.id, second.id]
