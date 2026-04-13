from torch.utils.data import Dataset


class QoSDataset(Dataset):
    TASK_RT = 'rt'
    TASK_TP = 'tp'
    TASK_RA = 'ra'
    TASK_BOTH = 'both'

    def split_field(self):
        pass

    def field_dims(self):
        pass

    def field_order(self):
        pass

    def dataset_fields(self):
        pass
