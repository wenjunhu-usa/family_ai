from threading import Thread
from time import sleep

from family_ai.task_queue import ModelTaskQueue


def test_queue_runs_fifo_one_at_a_time():
    queue = ModelTaskQueue()
    first = queue.reserve()
    second = queue.reserve()
    events = []

    def run(ticket, name):
        with queue.run(ticket):
            events.append(f"start-{name}")
            sleep(0.01)
            events.append(f"end-{name}")

    one = Thread(target=run, args=(first, "one"))
    two = Thread(target=run, args=(second, "two"))
    two.start()
    one.start()
    one.join()
    two.join()
    assert events == ["start-one", "end-one", "start-two", "end-two"]
