"""
Ollama API 测试脚本
用于测试和调试本地Ollama服务
"""

import requests
import json
import time


def test_ollama_connection():
    """测试Ollama基本连接"""
    print("=" * 60)
    print("1. 测试Ollama服务连接")
    print("-" * 60)
    
    try:
        response = requests.get("http://localhost:11434/")
        print(f"✓ Ollama服务状态: {response.text.strip()}")
        return True
    except Exception as e:
        print(f"✗ 无法连接到Ollama服务: {e}")
        print("  请确保Ollama已启动")
        return False


def list_available_models():
    """列出所有可用的模型"""
    print("\n" + "=" * 60)
    print("2. 列出可用模型")
    print("-" * 60)
    
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print("可用的模型列表:")
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0) / (1024**3)  # 转换为GB
                    print(f"  - {name} ({size:.2f} GB)")
                return [m['name'] for m in models]
            else:
                print("✗ 没有找到任何模型")
                print("  请先使用命令下载模型: ollama pull qwen2.5")
                return []
        else:
            print(f"✗ API调用失败: {response.status_code}")
            return []
    except Exception as e:
        print(f"✗ 列出模型失败: {e}")
        return []


def test_model_generate(model_name="qwen2.5:latest"):
    """测试模型生成功能"""
    print("\n" + "=" * 60)
    print(f"3. 测试模型生成 ({model_name})")
    print("-" * 60)
    
    test_prompt = "请用一句话介绍你自己。"
    
    try:
        print(f"发送测试prompt: {test_prompt}")
        print("等待响应...")
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": test_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                }
            },
            timeout=30
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"✓ 生成成功 (耗时: {elapsed_time:.2f}秒)")
            print(f"模型回复: {generated_text}")
            return True
        else:
            print(f"✗ 生成失败: HTTP {response.status_code}")
            print(f"错误信息: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("✗ 请求超时 (30秒)")
        return False
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        return False


def test_ocr_parsing(model_name="qwen2.5:latest"):
    """测试OCR文本解析"""
    print("\n" + "=" * 60)
    print("4. 测试OCR任务解析")
    print("-" * 60)
    
    # 测试用的OCR文本
    test_ocr_text = "在90秒内到达锦汇华庭A栋二单元门口"
    
    prompt = f"""你是一个智能助手，需要从OCR文字中提取驾驶任务信息。

OCR文字："{test_ocr_text}"

可用的建筑物：锦汇华庭A栋、锦汇华庭B栋、南辉金融广场1号楼等

请提取以下信息：
1. time_limit: 时间限制（秒数）
2. destination: 目的地建筑物
3. sub_destination: 子位置（如"二单元"）
4. action: 动作（"停车"或"到达"）

请以JSON格式返回：
{{
    "time_limit": 90,
    "destination": "锦汇华庭A栋",
    "sub_destination": "二单元门口",
    "action": "到达"
}}

只返回JSON，不要有其他说明。"""
    
    try:
        print(f"测试OCR文本: {test_ocr_text}")
        print("发送解析请求...")
        
        start_time = time.time()
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "keep_alive": -1,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                }
            },
            timeout=30
        )
        
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            print(f"✓ 解析完成 (耗时: {elapsed_time:.2f}秒)")
            print(f"原始回复:\n{generated_text}")
            
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                try:
                    parsed_json = json.loads(json_match.group(0))
                    print(f"\n提取的JSON:")
                    print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                    return True
                except json.JSONDecodeError as e:
                    print(f"\n✗ JSON解析失败: {e}")
                    return False
            else:
                print("\n✗ 未找到JSON格式的响应")
                return False
        else:
            print(f"✗ 请求失败: HTTP {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False


def test_different_models():
    """测试不同的模型"""
    print("\n" + "=" * 60)
    print("5. 测试不同的模型选项")
    print("-" * 60)
    
    # 常见的中文模型列表
    models_to_try = [
        "qwen2.5:latest",
        "qwen2.5:0.5b",
        "qwen2.5:1.5b",
        "qwen2.5:3b",
        "qwen2.5:7b",
        "qwen2:latest",
        "qwen2:0.5b",
        "qwen2:1.5b",
        "qwen2:7b",
        "qwen:latest",
        "qwen:0.5b",
        "qwen:1.8b",
        "qwen:4b",
        "qwen:7b",
        "deepseek-v2:latest",
        "yi:latest",
        "llama3.2:latest",
        "llama3.1:latest",
    ]
    
    print("尝试查找可用的模型...")
    available_models = list_available_models()
    
    if not available_models:
        print("\n建议运行以下命令安装模型:")
        print("  ollama pull qwen2.5")
        print("  ollama pull qwen2.5:1.5b")
        print("  ollama pull qwen2.5:0.5b")
        return
    
    print("\n测试已安装的模型:")
    for model in available_models:
        print(f"\n测试模型: {model}")
        success = test_model_generate(model)
        if success:
            print(f"  ✓ {model} 工作正常")
            break
        else:
            print(f"  ✗ {model} 测试失败")


def main():
    """主测试函数"""
    print("=" * 60)
    print("Ollama API 测试工具")
    print("=" * 60)
    
    # 1. 测试连接
    if not test_ollama_connection():
        print("\n请先启动Ollama服务:")
        print("  Windows: 确保Ollama已安装并运行")
        print("  检查任务栏是否有Ollama图标")
        return
    
    # 2. 列出模型
    models = list_available_models()
    
    if not models:
        print("\n没有找到任何模型，请先安装模型:")
        print("  ollama pull qwen2.5")
        print("  ollama pull qwen2.5:1.5b")
        print("  ollama pull qwen2.5:0.5b")
        return
    
    # 3. 让用户选择模型
    print("\n请选择要测试的模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print(f"  0. 测试所有模型")
    
    try:
        choice = input("\n输入选择 (默认1): ").strip()
        if not choice:
            choice = "1"
        choice = int(choice)
        
        if choice == 0:
            test_different_models()
        elif 1 <= choice <= len(models):
            selected_model = models[choice - 1]
            print(f"\n选择的模型: {selected_model}")
            
            # 测试基本生成
            if test_model_generate(selected_model):
                # 测试OCR解析
                test_ocr_parsing(selected_model)
        else:
            print("无效的选择")
            
    except ValueError:
        print("无效的输入")
    except KeyboardInterrupt:
        print("\n测试中止")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    # 提供配置建议
    print("\n建议的OCR处理器配置:")
    if models:
        recommended_model = models[0]
        print(f'self.ocr_processor = OCRProcessor(self.buildings, ollama_model="{recommended_model}")')
    else:
        print('请先安装模型: ollama pull qwen2.5')


if __name__ == "__main__":
    main()