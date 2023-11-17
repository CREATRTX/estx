import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

# 定义图神经网络模型from nuscenes.panoptic.panoptic_utils import get_frame_panoptic_instances, get_panoptic_instances_stats
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_io import load_bin_file


def truncate_class_name(class_name: str) -> str:
    """
    Truncate a given class name according to a pre-defined map.
    :param class_name: The long form (i.e. original form) of the class name.
    :return: The truncated form of the class name.
    """

    string_mapper = {
        "noise": 'noise',
        "human.pedestrian.adult": 'adult',
        "human.pedestrian.child": 'child',
        "human.pedestrian.wheelchair": 'wheelchair',
        "human.pedestrian.stroller": 'stroller',
        "human.pedestrian.personal_mobility": 'p.mobility',
        "human.pedestrian.police_officer": 'police',
        "human.pedestrian.construction_worker": 'worker',
        "animal": 'animal',
        "vehicle.car": 'car',
        "vehicle.motorcycle": 'motorcycle',
        "vehicle.bicycle": 'bicycle',
        "vehicle.bus.bendy": 'bus.bendy',
        "vehicle.bus.rigid": 'bus.rigid',
        "vehicle.truck": 'truck',
        "vehicle.construction": 'constr. veh',
        "vehicle.emergency.ambulance": 'ambulance',
        "vehicle.emergency.police": 'police car',
        "vehicle.trailer": 'trailer',
        "movable_object.barrier": 'barrier',
        "movable_object.trafficcone": 'trafficcone',
        "movable_object.pushable_pullable": 'push/pullable',
        "movable_object.debris": 'debris',
        "static_object.bicycle_rack": 'bicycle racks',
        "flat.driveable_surface": 'driveable',
        "flat.sidewalk": 'sidewalk',
        "flat.terrain": 'terrain',
        "flat.other": 'flat.other',
        "static.manmade": 'manmade',
        "static.vegetation": 'vegetation',
        "static.other": 'static.other',
        "vehicle.ego": "ego"
    }

    return string_mapper[class_name]


def render_histogram(nusc: NuScenes,
                     sort_by: str = 'count_desc',
                     verbose: bool = True,
                     font_size: int = 20,
                     save_as_img_name: str = None) -> None:
    """
    Render two histograms for the given nuScenes split. The top histogram depicts the number of scan-wise instances
    for each class, while the bottom histogram depicts the number of points for each class.
    :param nusc: A nuScenes object.
    :param sort_by: How to sort the classes to display in the plot (note that the x-axis, where the class names will be
        displayed on, is shared by the two histograms):
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :param verbose: Whether to display the plot in a window after rendering.
    :param font_size: Size of the font to use for the plot.
    :param save_as_img_name: Path (including image name and extension) to save the plot as.
class GNNModel(nn.Module):
    def __init__(self, in_features, hidden_features, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 加载数据集
dataset = Planetoid(root='/path/to/dataset', name='Cora')
data = dataset[0]

# 创建图神经网络模型
model = GNNModel(dataset.num_features, 16, dataset.num_classes)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# 模型训练
def train():
    model.train()
    optimizer.zero_grad()
    output = model(data.x, data.edge_index)
    loss = criterion(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# 模型评估
def evaluate():
    model.eval()
    output = model(data.x, data.edge_index)
    pred = output.argmax(dim=1)
    acc = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# 迭代训练和评估
for epoch in range(200):
    train()
    acc = evaluate()
    print(f'Epoch: {epoch+1}, Test Accuracy: {acc:.4f}')
