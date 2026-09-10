from trafficdet.counter import LineCounter, crossed_line


def test_crossed_line_detects_vertical_crossing():
    line = (0.0, 100.0, 200.0, 100.0)  # horizontal line at y=100

    assert crossed_line((50.0, 80.0), (50.0, 120.0), line) is True


def test_crossed_line_false_when_staying_on_same_side():
    line = (0.0, 100.0, 200.0, 100.0)

    assert crossed_line((50.0, 80.0), (60.0, 90.0), line) is False


def test_crossed_line_false_when_moving_away():
    line = (0.0, 100.0, 200.0, 100.0)

    assert crossed_line((50.0, 120.0), (50.0, 200.0), line) is False


def test_line_counter_counts_each_track_once():
    counter = LineCounter(line=(0.0, 100.0, 200.0, 100.0))

    counter.update(track_id=1, class_name="car", centroid=(50.0, 80.0))
    counted = counter.update(track_id=1, class_name="car", centroid=(50.0, 120.0))

    assert counted is True
    assert counter.counts == {"car": 1}
    assert counter.total == 1

    # 同じトラックがさらに動いても二重カウントしない
    counted_again = counter.update(track_id=1, class_name="car", centroid=(50.0, 160.0))
    assert counted_again is False
    assert counter.total == 1


def test_line_counter_tracks_multiple_classes_independently():
    counter = LineCounter(line=(0.0, 100.0, 200.0, 100.0))

    counter.update(track_id=1, class_name="car", centroid=(50.0, 80.0))
    counter.update(track_id=1, class_name="car", centroid=(50.0, 120.0))
    counter.update(track_id=2, class_name="bus", centroid=(60.0, 80.0))
    counter.update(track_id=2, class_name="bus", centroid=(60.0, 120.0))

    assert counter.counts == {"car": 1, "bus": 1}
    assert counter.total == 2


def test_line_counter_first_observation_does_not_count():
    counter = LineCounter(line=(0.0, 100.0, 200.0, 100.0))

    # 初回観測時はprev_pointがないのでカウントされない
    counted = counter.update(track_id=1, class_name="car", centroid=(50.0, 120.0))

    assert counted is False
    assert counter.total == 0
