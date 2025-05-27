import os
import os.path as osp
import jittor as jt
import numpy as np

from ..register import HOOKS
from .base_hook import BaseHook


@HOOKS.register_module()
class DetVisualizationHook(BaseHook):
    """用于检测结果可视化的钩子。

    Args:
        enable (bool): 是否启用可视化钩子，默认为True。
        show (bool): 是否在窗口中显示可视化结果，默认为False。
        show_dir (str): 保存可视化结果的目录，如果为None则不保存。默认为None。
        wait_time (float): 显示间隔时间，单位为秒。默认为2。
        score_thr (float): 检测结果的置信度阈值。默认为0.3。
    """

    def __init__(self,
                 enable=True,
                 show=False,
                 show_dir=None,
                 wait_time=2,
                 score_thr=0.3):
        self.enable = enable
        self.show = show
        self.show_dir = show_dir
        self.wait_time = wait_time
        self.score_thr = score_thr
        self.drawn_img_indices = set()
        # 检查是否有显示环境
        self.has_gui = self._check_display_available()
        if self.show and not self.has_gui:
            print("警告: 检测到无图形界面环境，自动禁用图像显示功能，仅保存图像")
            self.show = False
    
    def _check_display_available(self):
        """检查是否可以显示图像（是否有显示环境）"""
        # 检查环境变量
        if "DISPLAY" not in os.environ:
            return False
            
        # 尝试导入并初始化cv2
        try:
            import cv2
            # 尝试创建一个小窗口并立即销毁
            try:
                cv2.namedWindow("test_window", cv2.WINDOW_NORMAL)
                cv2.destroyWindow("test_window")
                return True
            except:
                return False
        except:
            return False

    def after_test_iter(self,
                        runner,
                        batch_idx,
                        data_batch=None,
                        outputs=None):
        """测试迭代后的钩子。

        在测试迭代之后调用，用于可视化检测结果。

        Args:
            runner: 运行器对象
            batch_idx (int): 当前批次的索引
            data_batch: 数据批次
            outputs: 模型输出
        """
        if not self.enable:
            return
        
        if self.show is False and self.show_dir is None:
            return
        
        # 打印数据批次结构，帮助调试
        if batch_idx == 0:
            runner.logger.info("=== 可视化钩子调试信息 ===")
            runner.logger.info(f"显示环境可用: {self.has_gui}")
            runner.logger.info(f"data_batch keys: {data_batch.keys()}")
            if 'data_samples' in data_batch:
                runner.logger.info(f"data_samples is a: {type(data_batch['data_samples'])}")
                if isinstance(data_batch['data_samples'], list) and len(data_batch['data_samples']) > 0:
                    sample = data_batch['data_samples'][0]
                    # 使用dir()获取所有属性
                    runner.logger.info(f"data_samples[0] attrs: {dir(sample) if hasattr(sample, '__dir__') else 'no attrs'}")
                    if hasattr(sample, 'metainfo'):
                        runner.logger.info(f"metainfo attrs: {dir(sample.metainfo) if hasattr(sample.metainfo, '__dir__') else 'no attrs'}")
            runner.logger.info(f"outputs is a: {type(outputs)}")
            if isinstance(outputs, list) and len(outputs) > 0:
                runner.logger.info(f"outputs[0] attrs: {dir(outputs[0]) if hasattr(outputs[0], '__dir__') else 'no attrs'}")
                if hasattr(outputs[0], 'pred_instances'):
                    runner.logger.info(f"pred_instances attrs: {dir(outputs[0].pred_instances) if hasattr(outputs[0].pred_instances, '__dir__') else 'no attrs'}")
        
        # 获取数据样本和输出
        data_samples = data_batch.get('data_samples', [])
        if not isinstance(data_samples, list):
            data_samples = [data_samples]
        
        # 确保outputs是list类型
        if not isinstance(outputs, list):
            outputs = [outputs]
        
        # 确保samples和outputs长度匹配
        num_samples = min(len(data_samples), len(outputs))
        if num_samples == 0:
            runner.logger.warning(f"样本数量为零，跳过批次 {batch_idx}")
            return
        
        # 遍历每个样本
        for i in range(num_samples):
            sample = data_samples[i]
            output = outputs[i]
            
            # 生成图像索引
            img_idx = batch_idx * len(data_samples) + i
            
            # 避免重复处理相同图像
            if img_idx in self.drawn_img_indices:
                continue
            self.drawn_img_indices.add(img_idx)
            
            # 获取图像路径 (尝试从不同位置获取)
            img_path = None
            if hasattr(sample, 'img_path'):
                img_path = sample.img_path
            elif hasattr(sample, 'metainfo') and hasattr(sample.metainfo, 'img_path'):
                img_path = sample.metainfo.img_path
            elif hasattr(sample, 'metainfo') and hasattr(sample.metainfo, 'filename'):
                img_path = sample.metainfo.filename
            
            if img_path is None:
                runner.logger.warning(f"无法获取图像 {img_idx} 的路径，尝试从数据加载器获取")
                # 尝试从数据集获取
                if hasattr(runner, 'test_dataset') and hasattr(runner.test_dataset, 'data_infos'):
                    try:
                        index = getattr(sample, 'index', i)
                        img_path = runner.test_dataset.data_infos[index].get('filename')
                        runner.logger.info(f"从数据集获取图像路径: {img_path}")
                    except:
                        pass
            
            if img_path is None:
                runner.logger.warning(f"无法获取图像 {img_idx} 的路径，跳过")
                continue
            
            runner.logger.info(f"处理图像: {img_path}")
            
            # 获取检测结果 (从pred_instances中)
            if not hasattr(output, 'pred_instances'):
                runner.logger.warning(f"输出中没有pred_instances属性，跳过图像 {img_idx}")
                continue
                
            pred_instances = output.pred_instances
            
            # 获取边界框、标签和分数
            bboxes = None
            labels = None
            scores = None
            
            if hasattr(pred_instances, 'bboxes'):
                bboxes = pred_instances.bboxes
            
            if hasattr(pred_instances, 'labels'):
                labels = pred_instances.labels
            
            if hasattr(pred_instances, 'scores'):
                scores = pred_instances.scores
            
            if bboxes is None or labels is None:
                runner.logger.warning(f"图像 {img_idx} 缺少边界框或标签信息")
                continue
            
            runner.logger.info(f"检测到 {len(bboxes)} 个目标")
            
            # 过滤低置信度的检测结果
            if scores is not None:
                mask = scores > self.score_thr
                bboxes = bboxes[mask]
                labels = labels[mask]
                scores = scores[mask]
                runner.logger.info(f"阈值过滤后保留 {len(bboxes)} 个目标")
            
            # 读取原始图像
            img = self._get_img_from_path(img_path, runner)
            if img is None:
                continue
            
            # 绘制检测结果
            drawn_img = self._draw_det_result(img, bboxes, labels, scores)
            
            # 显示图像（如果环境支持）
            if self.show and self.has_gui:
                try:
                    win_name = f"检测结果_{img_idx}"
                    if hasattr(sample, 'img_id'):
                        win_name = f"检测结果_{sample.img_id}"
                    self._show_img(drawn_img, win_name)
                    runner.logger.info(f"显示图像 {img_idx}")
                except Exception as e:
                    runner.logger.warning(f"显示图像失败: {e}")
            
            # 保存图像
            if self.show_dir is not None:
                save_path = self._save_img(drawn_img, img_path, img_idx, runner)
                runner.logger.info(f"保存图像到: {save_path}")
    
    def _get_img_from_path(self, img_path, runner=None):
        """从路径加载图像"""
        try:
            import cv2
            import os
            
            if runner is not None:
                runner.logger.info(f"尝试读取图像: {img_path}")
            
            # 尝试直接读取
            img = cv2.imread(img_path)
            
            if img is None:
                # 尝试的路径列表
                paths_to_try = []
                
                # 1. 绝对路径
                if not os.path.isabs(img_path):
                    abs_path = os.path.abspath(img_path)
                    paths_to_try.append(abs_path)
                
                # 2. 相对于data目录的路径
                data_path = os.path.join('data', img_path)
                paths_to_try.append(data_path)
                
                # 3. 相对于工作目录的路径
                if hasattr(runner, 'work_dir'):
                    work_dir_path = os.path.join(runner.work_dir, img_path)
                    paths_to_try.append(work_dir_path)
                
                # 4. 去掉可能的"data/"前缀
                if img_path.startswith('data/'):
                    no_prefix_path = img_path[5:]
                    paths_to_try.append(no_prefix_path)
                
                # 5. coco数据集常用路径
                basename = os.path.basename(img_path)
                coco_paths = [
                    os.path.join('data', 'coco', 'val2017', basename),
                    os.path.join('data', 'coco', 'train2017', basename),
                    os.path.join('data', 'coco', 'test2017', basename)
                ]
                paths_to_try.extend(coco_paths)
                
                # 尝试所有可能的路径
                for path in paths_to_try:
                    if runner is not None:
                        runner.logger.info(f"尝试路径: {path}")
                    img = cv2.imread(path)
                    if img is not None:
                        if runner is not None:
                            runner.logger.info(f"成功读取图像: {path}, 尺寸: {img.shape}")
                        break
                
                if img is None and runner is not None:
                    runner.logger.warning(f"所有路径尝试均失败，无法读取图像: {img_path}")
            else:
                if runner is not None:
                    runner.logger.info(f"成功读取图像: {img_path}, 尺寸: {img.shape}")
            
            return img
        except Exception as e:
            if runner is not None:
                runner.logger.error(f"读取图像时出错: {e}")
            return None
    
    def _draw_det_result(self, img, bboxes, labels, scores=None):
        """在图像上绘制检测结果"""
        import cv2
        
        drawn_img = img.copy()
        
        # 确保bboxes和labels是numpy数组
        if isinstance(bboxes, jt.Var):
            bboxes = bboxes.numpy()
        if isinstance(labels, jt.Var):
            labels = labels.numpy()
        if scores is not None and isinstance(scores, jt.Var):
            scores = scores.numpy()
        
        # 为不同类别生成不同颜色
        num_classes = max(labels) + 1 if len(labels) > 0 else 0
        colors = self._generate_colors(num_classes)
        
        # 绘制每个检测框
        for i, (bbox, label) in enumerate(zip(bboxes, labels)):
            # 获取坐标
            x1, y1, x2, y2 = bbox.astype(np.int32)
            
            # 获取标签颜色
            color = colors[label]
            
            # 绘制边界框
            cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, 2)
            
            # 构建标签文本
            label_text = f'class:{label}'
            if scores is not None:
                label_text += f' {scores[i]:.2f}'
            
            # 计算文本位置
            text_size, baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x1
            text_y = y1 - 5 if y1 - text_size[1] >= 5 else y1 + text_size[1] + 5
            
            # 绘制文本背景
            cv2.rectangle(drawn_img, (text_x, text_y - text_size[1]), 
                         (text_x + text_size[0], text_y + baseline), color, -1)
            
            # 绘制文本
            cv2.putText(drawn_img, label_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return drawn_img
    
    def _generate_colors(self, num_classes):
        """为每个类别生成颜色"""
        import random
        
        random.seed(42)  # 固定随机种子，保证颜色一致性
        colors = []
        for i in range(num_classes):
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            colors.append(color)
        return colors
    
    def _show_img(self, img, win_name):
        """显示图像"""
        # 如果没有显示环境，则跳过显示
        if not self.has_gui:
            return
            
        try:
            import cv2
            
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.imshow(win_name, img)
            cv2.waitKey(int(self.wait_time * 1000))
        except Exception as e:
            print(f"显示图像失败: {e}")
            # 如果显示失败，将has_gui标记为False，避免后续尝试显示
            self.has_gui = False
    
    def _save_img(self, img, img_path, img_idx, runner):
        """保存图像"""
        import cv2
        
        # 创建保存目录
        show_dir = self.show_dir
        if not osp.isabs(show_dir):
            show_dir = osp.join(runner.work_dir, show_dir)
        
        if not osp.exists(show_dir):
            os.makedirs(show_dir, exist_ok=True)
        
        # 获取文件名
        if img_path:
            filename = osp.splitext(osp.basename(img_path))[0]
        else:
            filename = f'img_{img_idx}'
        
        # 保存图像
        out_file = osp.join(show_dir, f'{filename}.jpg')
        cv2.imwrite(out_file, img)
        
        return out_file 