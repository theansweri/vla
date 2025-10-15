import requests
import json
from datetime import datetime

def check_loaded_models():
    """查看当前加载在内存中的模型"""
    try:
        response = requests.get("http://localhost:11434/api/ps")
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            if models:
                print("=" * 60)
                print("当前内存中的模型:")
                print("=" * 60)
                for model in models:
                    print(f"\n模型: {model.get('name', 'Unknown')}")
                    print(f"  大小: {model.get('size', 0) / (1024**3):.2f} GB")
                    print(f"  加载时间: {model.get('details', {}).get('format', 'N/A')}")
                    
                    # 如果有过期时间信息
                    expires_at = model.get('expires_at', '')
                    if expires_at:
                        print(f"  过期时间: {expires_at}")
            else:
                print("✗ 当前没有模型加载在内存中")
        else:
            print(f"API调用失败: {response.status_code}")
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    check_loaded_models()


    # 或者直接运行curl http://localhost:11434/api/ps