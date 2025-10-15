"""
OCR文字处理模块 - Ollama版本
使用本地Ollama模型进行智能文字理解和任务解析
"""

import re
import json
import requests
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from task_types import TaskInfo


class OCRProcessor:
    """OCR文字处理器 - 使用Ollama进行智能解析"""
    
    def __init__(self, buildings, ollama_model="qwen3:0.6b", ollama_url="http://localhost:11434"):
        """
        初始化OCR处理器
        
        Args:
            buildings: 建筑物列表
            ollama_model: Ollama模型名称
            ollama_url: Ollama服务地址
        """
        self.buildings = buildings
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        
        # 建筑物名称列表（原始和标准化版本）
        self.building_names = [building.name for building in buildings]
        self.normalized_building_names = [self._normalize_text(name) for name in self.building_names]
        
        # 建筑物名称映射表（标准化后的名称 -> 原始名称）
        self.normalized_to_original = {}
        for building in buildings:
            normalized = self._normalize_text(building.name)
            self.normalized_to_original[normalized] = building.name
        
        # 建筑物名称映射表（用于模糊匹配）
        self.building_name_map = self._build_name_mapping()
        
        # 测试Ollama连接
        self._test_ollama_connection()
    
    def _normalize_text(self, text: str) -> str:
        """
        标准化文本：去除空格、特殊字符等
        
        Args:
            text: 原始文本
            
        Returns:
            标准化后的文本
        """
        # 去除空格、·、.、-等特殊字符
        normalized = text.replace(' ', '').replace('　', '')
        normalized = normalized.replace('·', '').replace('•', '')
        normalized = normalized.replace('.', '').replace('-', '')
        normalized = normalized.replace('_', '').replace('/', '')
        return normalized
    
    def _test_ollama_connection(self):
        """测试Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.ollama_model not in model_names:
                    print(f"警告: 模型 {self.ollama_model} 未找到")
                    print(f"可用模型: {model_names}")
                else:
                    print(f"Ollama连接成功，使用模型: {self.ollama_model}")
            else:
                print(f"警告: 无法连接到Ollama服务")
        except Exception as e:
            print(f"警告: Ollama连接测试失败: {e}")
            print("将使用备用解析方法")
    
    def _build_name_mapping(self) -> Dict[str, str]:
        """构建建筑物名称映射表，用于模糊匹配"""
        name_map = {}
        for building in self.buildings:
            # 原始名称
            name_map[building.name] = building.name
            
            # 标准化版本
            normalized = self._normalize_text(building.name)
            name_map[normalized] = building.name
            
            # 提取关键词
            keywords = self._extract_keywords(building.name)
            for keyword in keywords:
                if keyword not in name_map:
                    name_map[keyword] = building.name
                # 标准化关键词
                normalized_keyword = self._normalize_text(keyword)
                if normalized_keyword not in name_map:
                    name_map[normalized_keyword] = building.name
        
        return name_map
    
    def _extract_keywords(self, building_name: str) -> List[str]:
        """从建筑物名称中提取关键词"""
        keywords = []
        # 提取中文建筑名称部分
        chinese_name = re.findall(r'[\u4e00-\u9fff]+', building_name)
        keywords.extend(chinese_name)
        # 提取字母+数字组合（如A栋、B1栋等）
        alphanumeric = re.findall(r'[A-Za-z]\d*栋?', building_name)
        keywords.extend(alphanumeric)
        # 提取数字+号楼/号大厦模式
        number_building = re.findall(r'\d+号[楼厦]?', building_name)
        keywords.extend(number_building)
        return keywords
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """调用Ollama模型"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 降低温度以获得更稳定的输出
                        "top_p": 0.9,
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                print(f"Ollama API调用失败: {response.status_code}")
                return None
                
        except requests.exceptions.Timeout:
            print("Ollama请求超时")
            return None
        except Exception as e:
            print(f"Ollama调用错误: {e}")
            return None
    
    def parse_task_instruction(self, ocr_text: str) -> Optional[TaskInfo]:
        """使用Ollama解析任务指令"""
        # 清理文本
        clean_text = ocr_text.strip().replace('\n', ' ').replace('\r', '')
        
        # 标准化OCR文本（用于传给LLM）
        normalized_ocr = self._normalize_text(clean_text)
        
        # 构建prompt（使用标准化后的文本）
        prompt = self._build_parse_prompt(normalized_ocr, clean_text)
        
        # 调用Ollama
        llm_response = self._call_ollama(prompt)
        
        if llm_response:
            # 尝试使用LLM解析结果
            task_info = self._parse_llm_response(llm_response, clean_text)
            if task_info:
                return task_info
        
        # 如果LLM解析失败，使用备用的规则方法
        print("LLM解析失败，使用备用规则解析")
        return self._fallback_parse(clean_text)
    
    def _build_parse_prompt(self, normalized_ocr: str, original_ocr: str) -> str:
        """构建解析任务的prompt"""
        # 使用标准化后的建筑物名称列表
        buildings_str = "; ".join(self.normalized_building_names)
        
        prompt = f"""# 角色
你是一个高精度、带纠错能力的地理位置信息解析AI。

# 任务
你的核心任务是：
1.  解析用户提供的OCR文本，提取指令中的关键信息。
2.  将提取到的“位置”信息与【地图已知位置列表】进行匹配和纠正。
3.  严格按照指定的JSON格式输出结果，绝不添加任何额外说明或注释。

# 地图已知位置列表 (Ground Truth)
这是地图上所有有效的位置名称。你提取和纠正后的位置 **必须** 是这个列表中的一个。

{buildings_str}

# 字段提取与纠错规则
- **"时间"**: 从文本中提取表示时限的词语 (例如 "10秒", "5分钟")。
- **"位置"**: 提取指令中的目标地点。**关键规则：** 如果OCR文本中的地点与【地图已知位置列表】中的名称不完全匹配（例如有错别字、漏字或不存在的编号），你必须**选择列表中最相似、最合理的名称**来替换它。例如，如果OCR识别出“东方之门大夏”，你应纠正为“东方之门大厦”。
- **"子位置"**: 如果文本中提到了更详细的地点信息（如单元、门口），则提取此信息。此字段也应尽可能与列表中的名称部分匹配。如果主要位置已包含详细信息（如“南安中心 1号楼”），则子位置可以重复该详细信息（“1号楼”）。如果没有，则为空字符串 ""。
- **"动作"**: 从文本中识别出核心动作 (例如 "抵达", "出发", "经过", "停车")。
- **"速度要求"**: 识别速度限制的类型，必须是 "无", "小于", "大于", "小于等于", "大于等于" 中的一个。
- **"速度数值"**: 提取速度的具体数值和单位 (km/h 或 m/s)。如果没有速度要求，此字段应为空字符串 ""。

# 学习示例
下面是一个完美的处理示例，请学习它的纠错和解析方式。

---
[示例输入]
OCR文字结果：要求车辆在5分钟内开到“锦汇毕庭8栋一单元”，路上的速度要大于25k。
---
[示例输出]
{{
  "时间": "5分钟",
  "位置": "锦汇华庭B栋",
  "子位置": "一单元",
  "动作": "开到",
  "速度要求": "大于",
  "速度数值": "25km/h"
}}
---
(示例解释：输入中的“锦汇毕庭8栋”在列表中不存在，模型根据“最相似、最合理”的原则，将其纠正为列表中存在的“锦汇华庭B栋”，输入中的“25k”存在识别不全的情况，自动补齐为“25km/h”。)

# 开始任务
现在，请严格遵守以上所有规则，处理以下文本，只输出最终的JSON对象：

[待处理输入]
OCR文字结果：{normalized_ocr}
"""
        print("prompt:", prompt)
        return prompt
    
    def _parse_time_to_seconds(self, time_str: str) -> int:
        """将时间字符串转换为秒数"""
        if '分钟' in time_str:
            minutes = int(re.findall(r'(\d+)', time_str)[0])
            return minutes * 60
        elif '秒' in time_str:
            return int(re.findall(r'(\d+)', time_str)[0])
        else:
            return 60  # 默认60秒

    def _parse_llm_response(self, llm_response: str, original_text: str) -> Optional[TaskInfo]:
        """解析LLM的响应"""
        try:
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                print(f"无法从LLM响应中提取JSON: {llm_response}")
                return None
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            # 验证必要字段
            if not data or 'destination' not in data:
                return None
            
            # 提取字段
            time_str = data.get('时间', '60秒')
            time_limit = self._parse_time_to_seconds(time_str)  # 需要添加这个函数
            destination = data.get('位置', '')
            sub_destination = data.get('子位置', '')
            action = data.get('动作', '停车')
            
            # 标准化destination
            normalized_destination = self._normalize_text(destination)
            
            # 查找对应的原始建筑物名称
            original_building_name = self.normalized_to_original.get(normalized_destination)
            
            if not original_building_name:
                # 尝试模糊匹配
                original_building_name = self._fuzzy_match_building(normalized_destination)
            
            if not original_building_name:
                print(f"LLM提取的建筑物 '{destination}' (标准化: '{normalized_destination}') 无法匹配")
                return None
            
            print(f"LLM成功解析任务:")
            print(f"  原文: {original_text}")
            print(f"  时间: {time_limit}秒")
            print(f"  主目的地: {original_building_name}")
            print(f"  子目的地: {sub_destination}")
            print(f"  动作: {action}")
            
            # 构建任务信息
            return TaskInfo(
                ocr_text=original_text,
                time_limit=int(time_limit),
                destination=original_building_name,  # 使用原始建筑物名称
                sub_destination=sub_destination if sub_destination else "",
                action=action,
                time_phrase=f"{time_limit}秒内",
                location_phrase=f"{original_building_name}{sub_destination}" if sub_destination else original_building_name,
                action_phrase=action
            )
            
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"LLM响应: {llm_response}")
            return None
        except Exception as e:
            print(f"解析LLM响应时出错: {e}")
            return None
    
    def _fuzzy_match_building(self, location_text: str) -> Optional[str]:
        """模糊匹配建筑物名称"""
        # 标准化输入文本
        normalized_input = self._normalize_text(location_text)
        
        # 直接匹配标准化后的名称
        if normalized_input in self.normalized_to_original:
            return self.normalized_to_original[normalized_input]
        
        # 在映射表中查找
        if location_text in self.building_name_map:
            return self.building_name_map[location_text]
        if normalized_input in self.building_name_map:
            return self.building_name_map[normalized_input]
        
        # 使用Ollama进行智能匹配
        if self.normalized_building_names:
            match_prompt = f"""从以下建筑物列表中，找出与"{normalized_input}"最匹配的建筑物名称。

建筑物列表（已去除空格和特殊字符）：
{chr(10).join(self.normalized_building_names)}

请直接返回最匹配的建筑物名称（标准化格式，无空格），不要有其他说明。如果没有合适的匹配，返回"无"。

例如：
- 输入"锦汇华厅A幢"应该匹配"锦汇华庭A栋"
- 输入"南辉金融广场1号"应该匹配"南辉金融广场1号楼"
"""
            print("match_prompt:", match_prompt)
            llm_match = self._call_ollama(match_prompt)
            if llm_match and llm_match.strip() != "无":
                normalized_match = self._normalize_text(llm_match.strip())
                # 查找对应的原始名称
                if normalized_match in self.normalized_to_original:
                    original_name = self.normalized_to_original[normalized_match]
                    print(f"LLM匹配: '{location_text}' -> '{original_name}'")
                    return original_name
        
        # 使用传统模糊匹配作为备用
        best_match = None
        best_score = 0
        
        for normalized_name, original_name in self.normalized_to_original.items():
            similarity = SequenceMatcher(None, normalized_input, normalized_name).ratio()
            if similarity > best_score and similarity > 0.6:
                best_score = similarity
                best_match = original_name
        
        if best_match:
            print(f"模糊匹配: '{location_text}' -> '{best_match}' (得分: {best_score:.2f})")
            return best_match
        
        return None
    
    def _fallback_parse(self, text: str) -> Optional[TaskInfo]:
        """备用的规则解析方法"""
        # 标准化文本用于匹配
        normalized_text = self._normalize_text(text)
        
        # 简单的正则匹配（在标准化文本上进行）
        patterns = [
            (r'(\d+)秒内去到(.+?)停车', '停车'),
            (r'(\d+)秒内去到(.+)', '停车'),
            (r'(\d+)秒内找到(.+?)停车', '停车'),
            (r'(\d+)秒内到达(.+)', '到达'),
        ]
        
        for pattern, default_action in patterns:
            match = re.search(pattern, normalized_text)
            if match:
                time_limit = int(match.group(1))
                location_full = match.group(2).strip()
                
                # 分离主目的地和子目的地
                main_destination, sub_destination = self._split_destination(location_full)
                
                # 匹配建筑物
                matched_building = self._fuzzy_match_building(main_destination)
                if matched_building:
                    return TaskInfo(
                        ocr_text=text,
                        time_limit=time_limit,
                        destination=matched_building,
                        sub_destination=sub_destination,
                        action=default_action,
                        time_phrase=f"{time_limit}秒内",
                        location_phrase=location_full,
                        action_phrase=default_action
                    )
        
        return None
    
    def _split_destination(self, location_text: str) -> Tuple[str, str]:
        """分离主目的地和子目的地"""
        sub_patterns = [
            r'([一二三四五六七八九十\d]+单元)',
            r'([一二三四五六七八九十\d]+号楼)',
            r'(门口)',
            r'(入口)',
            r'(大厅)',
        ]
        
        sub_destination = ""
        main_destination = location_text
        
        for pattern in sub_patterns:
            match = re.search(pattern, location_text)
            if match:
                sub_destination = match.group(1)
                main_destination = location_text[:match.start()] + location_text[match.end():]
                main_destination = main_destination.strip()
                break
        
        return main_destination, sub_destination
    
    def check_detail_location_found(self, ocr_texts: List[str], target_detail: str) -> bool:
        """检查是否找到了详细位置"""
        # 标准化目标文本
        normalized_target = self._normalize_text(target_detail)
        
        # 先用简单匹配（在标准化文本上）
        for text in ocr_texts:
            normalized_text = self._normalize_text(text)
            if normalized_target in normalized_text:
                return True
        
        # 如果没找到，可以用LLM进行智能匹配
        if ocr_texts and target_detail:
            # 标准化所有OCR文本
            normalized_texts = [self._normalize_text(t) for t in ocr_texts]
            texts_str = " ".join(normalized_texts)
            
            check_prompt = f"""判断以下OCR文字中是否包含"{normalized_target}"或类似的内容：

OCR文字（已标准化）：{texts_str}

目标内容（已标准化）：{normalized_target}

如果包含目标内容或非常相似的内容，回答"是"；否则回答"否"。只回答一个字。
"""
            
            response = self._call_ollama(check_prompt)
            if response and "是" in response:
                print(f"LLM识别到目标位置: {target_detail}")
                return True
        
        return False