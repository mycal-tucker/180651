class MetricTracker:
    def __init__(self, depth, num_prototypes, model_id, inverted=False, identity=False):
        self.depth = depth
        self.num_prototypes = num_prototypes
        self.model_id = model_id
        self.inverted = inverted
        self.identity = identity
        self.metric_map = {}

    def record_metrics(self, metric_name, metric_values):
        if metric_name not in self.metric_map.keys():
            self.metric_map[metric_name] = []
        self.metric_map.get(metric_name).append(metric_values)

    def get_metrics(self, metric_name):
        return self.metric_map.get(metric_name)
