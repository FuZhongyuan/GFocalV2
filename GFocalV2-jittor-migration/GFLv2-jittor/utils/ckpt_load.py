import jittor as jt
import os

def load_ckpt(model, pretrained_path):
    """
    加载预训练权重
    
    参数:
    - model: 要加载权重的模型
    - pretrained_path: 预训练模型的路径，可以是文件路径或jittorhub路径
    
    返回值:
    - 加载预训练权重的模型
    """
    if pretrained_path.startswith("jittorhub://"):
        # 从jittorhub加载
        hub_path = pretrained_path.replace("jittorhub://", "")
        print(f"[INFO] 从jittorhub加载预训练权重: {hub_path}")
        try:
            # 使用正确的方式处理jittorhub路径
            if hub_path == "resnet50.pkl":
                # 使用jittor.models加载预训练模型
                from jittor import models
                temp_model = models.resnet50(pretrained=True)
                state_dict = temp_model.state_dict()
            else:
                # 尝试使用jittor的hub机制加载
                state_dict = jt.load(f"jittor-hub://{hub_path}")
                
            # 加载权重到模型
            if isinstance(state_dict, dict) and 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            else:
                model.load_state_dict(state_dict)
            print(f"[INFO] 成功从jittorhub加载预训练权重: {hub_path}")
        except Exception as e:
            print(f"[ERROR] 从jittorhub加载预训练权重失败: {e}")
            print("[INFO] 尝试从本地缓存加载预训练权重")
            try:
                # 尝试从Jittor的缓存目录加载
                jt_home = os.environ.get('JT_HOME', os.path.expanduser('~/.cache/jittor'))
                cache_path = os.path.join(jt_home, 'models', hub_path)
                if os.path.exists(cache_path):
                    state_dict = jt.load(cache_path)
                    if isinstance(state_dict, dict) and 'model' in state_dict:
                        model.load_state_dict(state_dict['model'])
                    elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                        model.load_state_dict(state_dict['state_dict'])
                    else:
                        model.load_state_dict(state_dict)
                    print(f"[INFO] 成功从本地缓存加载预训练权重: {cache_path}")
                else:
                    print(f"[WARNING] 本地缓存中没有找到预训练权重: {cache_path}")
            except Exception as e2:
                print(f"[ERROR] 从本地缓存加载预训练权重失败: {e2}")
    elif os.path.exists(pretrained_path):
        # 从本地文件加载
        print(f"[INFO] 从本地文件加载预训练权重: {pretrained_path}")
        try:
            state_dict = jt.load(pretrained_path)
            if isinstance(state_dict, dict) and 'model' in state_dict:
                model.load_state_dict(state_dict['model'])
            elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
                model.load_state_dict(state_dict['state_dict'])
            elif isinstance(state_dict, dict) and 'ema' in state_dict:
                model.load_state_dict(state_dict['ema'])
            else:
                model.load_state_dict(state_dict)
            print(f"[INFO] 成功从本地文件加载预训练权重: {pretrained_path}")
        except Exception as e:
            print(f"[ERROR] 从本地文件加载预训练权重失败: {e}")
    else:
        print(f"[WARNING] 未找到预训练权重文件: {pretrained_path}")
    
    return model 