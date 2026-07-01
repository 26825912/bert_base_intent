"""
API 测试客户端 - 意图分类推理接口测试
"""
import requests
import json
import time


class IntentAPIClient:
    """意图分类 API 客户端"""

    def __init__(self, base_url: str = "http://localhost:8085"):
        """
        初始化客户端

        Args:
            base_url: API服务地址
                     - 本地测试: http://localhost:8085
                     - 生产环境: http://your-server:8085
        """
        self.base_url = base_url

    def predict(self, text: str, return_probabilities: bool = False) -> dict:
        """
        单条文本预测

        Args:
            text: 输入文本
            return_probabilities: 是否返回所有类别概率

        Returns:
            预测结果字典
        """
        url = f"{self.base_url}/predict"
        payload = {
            "text": text,
            "return_probabilities": return_probabilities
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def check_health(self) -> dict:
        """健康检查"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


def print_result(result: dict, show_probabilities: bool = False):
    """打印预测结果"""
    print(f"  预测意图: {result['intent']}")
    print(f"  置信度: {result['confidence']:.4f}")
    print(f"  推理时间: {result['inference_time']:.4f} 秒")

    if show_probabilities and result.get('probabilities'):
        print("  概率分布:")
        # 按概率降序排序，只显示前5
        sorted_probs = sorted(result['probabilities'].items(),
                             key=lambda x: x[1], reverse=True)[:5]
        for intent, prob in sorted_probs:
            print(f"    {intent}: {prob:.4f}")


def main():
    print("=" * 60)
    print("意图分类 API 测试")
    print("=" * 60)

    client = IntentAPIClient()

    # 1. 健康检查
    print("\n[1] 健康检查...")
    try:
        health = client.check_health()
        print(f"  状态: {health['status']}")
    except Exception as e:
        print(f"  连接失败: {e}")
        return

    # 2. 单条预测测试（不含概率）
    print("\n[2] 单条预测测试（不含概率）...")
    test_text = "国内租车用户: 我想查询订单信息"
    print(f"  输入: {test_text}")

    try:
        result = client.predict(test_text, return_probabilities=False)
        print_result(result, show_probabilities=False)
    except Exception as e:
        print(f"  预测失败: {e}")

    # 3. 单条预测测试（含概率）
    print("\n[3] 单条预测测试（含概率分布）...")
    test_text = "国际租车用户: 怎么修改订单？"
    print(f"  输入: {test_text}")

    try:
        result = client.predict(test_text, return_probabilities=True)
        print_result(result, show_probabilities=True)
    except Exception as e:
        print(f"  预测失败: {e}")

    # 4. 批量预测测试
    print("\n[4] 批量预测测试...")
    test_texts = [
        "国内租车用户: 一天多少钱？",
        "国际租车用户: 在哪里取车？",
        "新西兰租车"
    ]

    total_time = 0
    for i, text in enumerate(test_texts, 1):
        print(f"\n  [{i}/{len(test_texts)}] 输入: {text}")
        try:
            result = client.predict(text)
            print_result(result)
            total_time += result['inference_time']
        except Exception as e:
            print(f"  预测失败: {e}")

    print(f"\n  平均推理时间: {total_time / len(test_texts):.4f} 秒")

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
