import argparse
import importlib.util
import json
import os
import sys
import numpy as np
import cv2

# 导入相机内外参数 在colmap_loader.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLMAP_LOADER_PATH = os.path.join(REPO_ROOT, "scene", "colmap_loader.py")

spec = importlib.util.spec_from_file_location("colmap_loader_local", COLMAP_LOADER_PATH)
colmap_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colmap_loader)


def read_colmap(source_path):
    # 读取COLMAP稀疏相机内参数/外参数  source_path/sparse/0.
    sparse_dir = os.path.join(source_path, "sparse", "0")
    try:
        images = colmap_loader.read_extrinsics_binary(os.path.join(sparse_dir, "images.bin"))
        cameras = colmap_loader.read_intrinsics_binary(os.path.join(sparse_dir, "cameras.bin"))
    except Exception:
        images = colmap_loader.read_extrinsics_text(os.path.join(sparse_dir, "images.txt"))
        cameras = colmap_loader.read_intrinsics_text(os.path.join(sparse_dir, "cameras.txt"))
    return images, cameras


def pinhole_params(camera):
    # COLMAP相机模型转换为 fx, fy, cx, cy
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = camera.params[0]
        cx, cy = camera.params[1], camera.params[2]
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    else:
        raise ValueError(f"Only SIMPLE_PINHOLE/PINHOLE are supported, got {camera.model}.")
    return float(fx), float(fy), float(cx), float(cy)


def select_image_record(colmap_images, image_path, camera_name):
    # 找到与渲染图像相对应的COLMAP图像记录。
    records = sorted(colmap_images.values(), key=lambda item: item.name)
    image_base = os.path.basename(image_path)
    stem = os.path.splitext(image_base)[0]

    # 如果用户提供了原始的COLMAP图像名称，则信任该名称
    if camera_name:
        for record in records:
            if record.name == camera_name or os.path.splitext(record.name)[0] == camera_name:
                return record
        raise ValueError(f"Camera image name not found in COLMAP: {camera_name}")

    for record in records:
        if record.name == image_base or os.path.splitext(record.name)[0] == stem:
            return record

    # Rendered files are often named 00000.png. In that case use sorted camera index.
    if stem.isdigit():
        idx = int(stem)
        if 0 <= idx < len(records):
            return records[idx]

    raise ValueError(
        "无法从图像名称读取相机参数"
    )


def default_depth_path(image_path):
    # 优先选择渲染文件旁同名NPY深度文件，然后尝试 renders/../depths查找
    same_dir = os.path.splitext(image_path)[0] + ".npy"
    if os.path.exists(same_dir):
        return same_dir

    image_dir = os.path.dirname(image_path)
    parent = os.path.dirname(image_dir)
    if os.path.basename(image_dir) == "renders":
        sibling_depth = os.path.join(parent, "depths", os.path.splitext(os.path.basename(image_path))[0] + ".npy")
        if os.path.exists(sibling_depth):
            return sibling_depth

    return same_dir


def load_depth(path):
    # 加载由render.py保存的原始深度数组，数据类型：float32、int64
    depth = np.load(path).astype(np.float64)
    return np.squeeze(depth)


def rect_from_points(p0, p1, width, height):
    # 将拖动的起点/终点转换为裁剪后的x0, y0, x1, y1矩形
    x0, x1 = sorted((p0[0], p1[0]))
    y0, y1 = sorted((p0[1], p1[1]))
    x0 = int(np.clip(x0, 0, width - 1))
    x1 = int(np.clip(x1, 0, width - 1))
    y0 = int(np.clip(y0, 0, height - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def depth_to_z(depth_value, depth_mode):
    # 将保存的深度值转换为相机空间中的z值
    if depth_mode == "inverse":
        return 1.0 / np.maximum(depth_value, 1e-8)
    return depth_value


def backproject_bbox(depth, rect, image_record, camera_record, depth_mode, stride):
    # 将2D矩形中的采样像素反投影到世界空间边界框中
    fx, fy, cx, cy = pinhole_params(camera_record)
    x0, y0, x1, y1 = rect

    # 对网格进行采样，而不是对每个像素进行采样。对于大盒子来说，这样会快得多，
    # 而且这些信息仍然足以估算出一个大致的修复体积。
    xs = np.arange(x0, x1 + 1, stride)
    ys = np.arange(y0, y1 + 1, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)

    # 深度图的分辨率可以与图像窗口的分辨率不同。
    # 将图像像素坐标缩放到深度数组坐标中。
    sx = depth.shape[1] / camera_record.width
    sy = depth.shape[0] / camera_record.height
    dx = np.clip(np.round(grid_x * sx).astype(np.int32), 0, depth.shape[1] - 1)
    dy = np.clip(np.round(grid_y * sy).astype(np.int32), 0, depth.shape[0] - 1)
    sampled = depth[dy, dx]

    valid = np.isfinite(sampled) & (sampled > 0)
    if not valid.any():
        raise ValueError("No valid depth in selected rectangle.")

    #针孔相机背投影：
    #u, v 是图像像素; z 是相机空间深度.
    #x = (u - cx) * z / fx, y = (v - cy) * z / fy.
    u = grid_x[valid].astype(np.float64)
    v = grid_y[valid].astype(np.float64)
    z = depth_to_z(sampled[valid], depth_mode)

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_cam = np.stack([x, y, z], axis=1)

    # COLMAP将世界到相机的信息存储为 X_cam = R_w2c * X_world + t_w2c.
    # 将其反转以获得 X_world = R_c2w * (X_cam - t_w2c).
    R_w2c = colmap_loader.qvec2rotmat(image_record.qvec)
    t_w2c = np.asarray(image_record.tvec, dtype=np.float64)
    R_c2w = R_w2c.T
    points_world = (R_c2w @ (points_cam - t_w2c).T).T

    # 世界坐标系中的轴对齐三维立方体（AABB）.
    bbox_min = points_world.min(axis=0)
    bbox_max = points_world.max(axis=0)
    bbox_center = (bbox_min + bbox_max) * 0.5
    bbox_size = bbox_max - bbox_min

    return {
        "rect_xyxy": list(map(int, rect)),
        "num_points": int(points_world.shape[0]),
        "depth_mode": depth_mode,
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_center": bbox_center.tolist(),
        "bbox_size": bbox_size.tolist(),
    }


class App:
    def __init__(self, args):
        if cv2 is None:
            raise RuntimeError("OpenCV is not installed. Run: pip install opencv-python")

        # 1. 从命令行中选择并加载单个渲染图像
        self.args = args
        self.image = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if self.image is None:
            raise FileNotFoundError(args.image)

        # 2. 加载对应的原始深度NPY文件。
        self.depth_path = args.depth or default_depth_path(args.image)  #npy文件名称与渲染图象名称一致
        self.depth = load_depth(self.depth_path)

        # 3. 加载摄像机的内参数/外参数，并将此图像与某个摄像机进行匹配。
        colmap_images, colmap_cameras = read_colmap(os.path.abspath(args.source_path))
        self.image_record = select_image_record(colmap_images, args.image, args.camera_name)
        self.camera_record = colmap_cameras[self.image_record.camera_id]

        self.dragging = False  #定义鼠标参量
        self.start = None
        self.end = None
        self.result = None

    def mouse(self, event, x, y, flags, userdata):
        # OpenCV鼠标：按下，移动，释放
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.end = (x, y)   #end坐标更新到释放为止
            rect = rect_from_points(self.start, self.end, self.image.shape[1], self.image.shape[0])
            if rect is not None:
                # 整个工具的核心操作都在这里（所有数学过程的代码实现最重要）！！！
                self.result = backproject_bbox(
                    self.depth,
                    rect,
                    self.image_record,
                    self.camera_record,
                    self.args.depth_mode,
                    self.args.stride,
                )    #计算出3D点（X，Y，Z）坐标，并反投影至3D世界
                self.result.update({
                    "image": os.path.abspath(self.args.image),
                    "depth": os.path.abspath(self.depth_path),
                    "camera_name": self.image_record.name,
                })
                self.save() #将结果存储到json文件中
                print(json.dumps(self.result, indent=2))

    def save(self):
        # 将最新的边界框（bbox）结果写入JSON格式
        os.makedirs(os.path.dirname(os.path.abspath(self.args.output)), exist_ok=True)
        with open(self.args.output, "w", encoding="utf-8") as f:
            json.dump(self.result, f, indent=2)

    def draw(self):
        # 绘制图像和当前矩形
        view = self.image.copy()
        if self.start is not None and self.end is not None:
            rect = rect_from_points(self.start, self.end, view.shape[1], view.shape[0])
            if rect is not None:
                x0, y0, x1, y1 = rect
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 230, 255), 2)
        return view

    def run(self):
        # OpenCV循环 按q与
        window = "depth bbox"
        window_flags = cv2.WINDOW_NORMAL
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            window_flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(window, window_flags)
        cv2.setMouseCallback(window, self.mouse)  #这里调用mouse不传参，否则报错
        while True:
            cv2.imshow(window, self.draw())
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Draw one 2D box and back-project same-name NPY depth to a 3D bbox.")
    parser.add_argument("--image", required=True, help="Rendered image path, e.g. output/.../renders/00000.png.")
    parser.add_argument("--source_path", "-s", required=True, help="Dataset root containing sparse/0.")
    parser.add_argument("--depth", default=None, help="Depth .npy path. Default: same basename as --image.")
    parser.add_argument("--camera_name", default=None, help="Original COLMAP image name if it cannot be inferred.")
    parser.add_argument("--depth_mode", default="inverse", choices=["inverse", "linear"], help="Use inverse for current 3DGS depth output.")
    parser.add_argument("--stride", default=4, type=int, help="Sample every N pixels in the selected box.")
    parser.add_argument("--output", default="repair_bbox.json", help="Output bbox JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    App(args).run()


if __name__ == "__main__":
    main()
