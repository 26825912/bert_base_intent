"""
配置加载器 - 统一管理项目配置
"""
import os
import yaml
from typing import Dict, Any
from pathlib import Path

# 全局配置缓存
_config_cache = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径，默认为 config/config.yaml

    Returns:
        配置字典
    """
    global _config_cache

    if config_path is None:
        # 默认配置文件路径
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 解析相对路径为绝对路径
    project_root = config_path.parent.parent
    if 'paths' in config:
        for key, value in config['paths'].items():
            if value and value.startswith('./'):
                config['paths'][key] = str(project_root / value[2:])

    _config_cache = config
    return config


def get_config() -> Dict[str, Any]:
    """
    获取配置（使用缓存）

    Returns:
        配置字典
    """
    global _config_cache
    if _config_cache is None:
        _config_cache = load_config()
    return _config_cache


def update_config_from_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    """
    从命令行参数更新配置

    Args:
        config: 配置字典
        args: argparse 参数对象

    Returns:
        更新后的配置字典
    """
    # 更新路径配置
    if hasattr(args, 'data_dir') and args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    if hasattr(args, 'output_dir') and args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    if hasattr(args, 'model_path') and args.model_path:
        config['paths']['base_model_path'] = args.model_path

    # 更新训练配置
    if hasattr(args, 'batch_size') and args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if hasattr(args, 'num_epochs') and args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs

    return config
