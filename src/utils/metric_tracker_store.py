import numpy as np


class MetricTrackerStore:
    def __init__(self):
        self.trackers = []

    def add_tracker(self, tracker):
        self.trackers.append(tracker)

    # Given some set of attributes (e.g. depth or num_prototypes), returns the matching metric trackers.
    # Needs to match ALL the non-None parameters
    def get_matching_metric_trackers(self, req_depth=None, req_num_prototypes=None, req_identity=None,
                                     req_inverted=None):
        matching_trackers = []
        for tracker in self.trackers:
            if (req_depth is None or req_depth == tracker.depth) and\
                    (req_num_prototypes is None or req_num_prototypes == tracker.num_prototypes) and\
                    (req_identity is None or req_identity == tracker.identity) and\
                    (req_inverted is None or req_inverted == tracker.inverted):
                matching_trackers.append(tracker)
        return matching_trackers

    def get_matching_metric_values(self, metric_name, req_depth=None, req_num_prototypes=None, req_identity=None,
                                   req_inverted=None):
        matching_trackers = self.get_matching_metric_trackers(req_depth=req_depth, req_num_prototypes=req_num_prototypes,
                                                              req_identity=req_identity, req_inverted=req_inverted)
        if not matching_trackers:
            return None
        collapsed_metric_values = []
        for tracker in matching_trackers:
            tracker_metric_values = tracker.get_metrics(metric_name)
            collapsed_metric_values.extend(tracker_metric_values)
        numpy_version = np.asarray(collapsed_metric_values)
        return numpy_version
