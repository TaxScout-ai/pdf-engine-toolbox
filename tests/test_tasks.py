"""Test background task polling endpoint."""

from app.services import task_service


def test_get_task_requires_authentication(client):
    """Task results can contain full PDFs and must never be public."""
    response = client.get("/tasks/nonexistent-id")
    assert response.status_code == 401


def test_get_task_not_found(client, auth_headers):
    """Requesting a non-existent task should return an error."""
    path = "/tasks/nonexistent-id"
    response = client.get(path, headers=auth_headers("GET", path, ""))
    data = response.json()
    assert data["success"] is False


def test_task_lifecycle(client, auth_headers):
    """Create, process, and complete a task."""
    task = task_service.create_task("test_op")
    assert task.status == task_service.TaskStatus.PENDING
    path = f"/tasks/{task.id}"
    headers = auth_headers("GET", path, "")

    # Poll - should show pending
    response = client.get(path, headers=headers)
    data = response.json()
    assert data["success"] is True
    assert data["data"]["status"] == "pending"

    # Mark processing
    task_service.set_processing(task.id)
    response = client.get(path, headers=headers)
    assert response.json()["data"]["status"] == "processing"

    # Complete
    task_service.complete_task(task.id, {"pages_processed": 10})
    response = client.get(path, headers=headers)
    data = response.json()["data"]
    assert data["status"] == "completed"
    assert data["result"]["pages_processed"] == 10


def test_task_failure(client, auth_headers):
    """Failed tasks should report the error."""
    task = task_service.create_task("failing_op")
    task_service.fail_task(task.id, "Something went wrong")
    path = f"/tasks/{task.id}"

    response = client.get(path, headers=auth_headers("GET", path, ""))
    data = response.json()["data"]
    assert data["status"] == "failed"
    assert "Something went wrong" in data["error"]
