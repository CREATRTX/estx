
        if self.mode == 'compute_stats':
            return self.compute_stats(idx)
        elif self.mode == 'extract_data':
            self.extract_data(idx)
            return 0
        else:
            return self.load_data(idx)

    @abc.abstractmethod
    def compute_stats(self, idx: int) -> Dict:
        """
        Function to compute dataset statistics like max surrounding agents, max nodes, max edges etc.
        :param idx: data index
        """
        raise NotImplementedError()

    def extract_data(self, idx: int):
        """
        Function to extract data. Bulk of the dataset functionality will be implemented by this method.
        :param idx: data index
        """
        inputs = self.get_inputs(idx)
        ground_truth = self.get_ground_truth(idx)
        data = {'inputs': inputs, 'ground_truth': ground_truth}
        self.save_data(idx, data)

    @abc.abstractmethod
    def get_inputs(self, idx: int) -> Dict:
        """
        Extracts model inputs.
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_ground_truth(self, idx: int) -> Dict:
        """
        Extracts ground truth 'labels' for training.
        :param idx: data index
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load_data(self, idx: int) -> Dict:
        """
        Function to load extracted data.
        :param idx: data index
        :return data: Dictionary with pre-processed data
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def save_data(self, idx: int, data: Dict):
        """
        Function to save extracted pre-processed data.
        :param idx: data index
        :param data: Dictionary with pre-processed data
        """
        raise NotImplementedError()


class SingleAgentDataset(TrajectoryDataset):
    """
    Base class for single agent dataset. While we implicitly model all surrounding agents in the scene, predictions
    are made for a single target agent at a time.
    """

    @abc.abstractmethod
    def get_map_representation(self, idx: int) -> Union[np.ndarray, Dict]:
