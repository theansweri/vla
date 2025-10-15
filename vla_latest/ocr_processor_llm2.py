"""
OCR文字处理模块 - 异步优化版本
使用本地Ollama模型进行智能文字理解和任务解析
包含OCR质量评估和异步处理
"""

import re
import json
import requests
import threading
import time
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from task_types import TaskInfo
from queue import Queue
from dataclasses import dataclass


@dataclass
class OCRQualityScore:
    """OCR质量评分"""
    text: str
    has_time: bool
    has_building: bool
    has_action: bool
    has_speed: bool
    char_count: int
    keyword_count: int
    total_score: float
    is_valid: bool


class OCRProcessor:
    """OCR文字处理器 - 使用Ollama进行智能解析（异步版本）"""
    
    def __init__(self, buildings, ollama_model="qwen3:0.6b", ollama_url="http://localhost:11434"):  # "qwen3:0.6b"
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
        
        # 关键词列表（用于快速评估）
        self.time_keywords = ['秒', '分']
        self.action_keywords = ['到达', '到', '去', '停车', '前往', '开到', '经过']
        # 包含所有可能的速度单位及其大小写组合
        self.speed_keywords = [
            'm/s', 'M/S', 'M/s', 'm/S',
            'km/h', 'km/H', 'KM/h', 'KM/H', 'Km/h', 'Km/H', 'kM/h', 'kM/H',
            '米/秒', '米每秒',
            '千米/小时', '千米每小时',
            '公里/小时', '公里每小时',
        ]
        self.building_keywords = self._extract_all_building_keywords()
        
        # 异步处理相关
        self.llm_processing = False
        self.llm_thread = None
        self.llm_result_queue = Queue()
        self.last_valid_ocr_text = None
        self.last_llm_request_time = 0
        self.min_llm_interval = 2.0  # 最小LLM调用间隔（秒）
        
        # 子目的地常见OCR混淆词对（双向替换）
        # 注意：仅包含高置信的常见错读，避免过度替换
        self.detail_confusion_pairs: List[Tuple[str, str]] = [
            ("门诊", "口诊"),
            ("入口", "人口"),
            ("门口", "口口"),
        ]

        # 测试Ollama连接
        self._test_ollama_connection()
    
    def _extract_all_building_keywords(self) -> List[str]:
        """提取所有建筑物的关键词"""
        keywords = set()
        for building_name in self.building_names:
            # 提取中文部分
            chinese_parts = re.findall(r'[\u4e00-\u9fff]+', building_name)
            keywords.update(chinese_parts)
            # 提取字母栋号
            building_codes = re.findall(r'[A-Za-z]\d*栋?', building_name)
            keywords.update(building_codes)
        return list(keywords)
    
    def evaluate_ocr_quality(self, ocr_text: str) -> OCRQualityScore:
        """
        评估OCR文本质量，判断是否值得送入LLM
        
        Args:
            ocr_text: OCR识别的文本
            
        Returns:
            OCRQualityScore: 质量评分对象
        """
        # 清理文本
        clean_text = ocr_text.strip().replace('\n', ' ').replace('\r', '')
        normalized = self._normalize_text(clean_text)
        
        # 初始化评分
        score = OCRQualityScore(
            text=clean_text,
            has_time=False,
            has_building=False,
            has_action=False,
            has_speed=False,
            char_count=len(clean_text),
            keyword_count=0,
            total_score=0.0,
            is_valid=False
        )
        
        # 1. 长度检查
        if score.char_count < 10:
            return score
        
        # 2. 检查时间关键词
        for keyword in self.time_keywords:
            if keyword in normalized:
                score.has_time = True
                score.keyword_count += 1
                break
        
        # 3. 检查动作关键词
        for keyword in self.action_keywords:
            if keyword in normalized:
                score.has_action = True
                score.keyword_count += 1
                break

        # 4. 检查速度关键词
        for keyword in self.speed_keywords:
            if keyword in normalized:
                score.has_speed = True
                score.keyword_count += 1
                break
        
        # 4. 检查建筑物关键词
        for keyword in self.building_keywords:
            if keyword in clean_text or keyword in normalized:
                score.has_building = True
                score.keyword_count += 1
                break
        
        # 5. 计算总分
        score.total_score = 0.0
        if score.has_time:
            score.total_score += 0.3
        if score.has_action:
            score.total_score += 0.3
        if score.has_speed:
            score.total_score += 0.3
        if score.has_building:
            score.total_score += 0.3
        
        # 长度奖励
        if score.char_count > 20:
            score.total_score += 0.1
        if score.char_count > 30:
            score.total_score += 0.1
        
        # 6. 判断是否有效
        score.is_valid = (score.total_score >= 0.8)
        
        return score
    
    def parse_task_instruction(self, ocr_text: str) -> Optional[TaskInfo]:
        """
        智能解析任务指令（带质量评估和异步处理）
        
        Args:
            ocr_text: OCR识别的文本
            
        Returns:
            TaskInfo: 如果立即可用则返回任务信息，否则返回None
        """
        # 1. 评估OCR质量
        quality_score = self.evaluate_ocr_quality(ocr_text)
        
        print(f"OCR质量评估: 文本长度={quality_score.char_count}, "
              f"关键词数={quality_score.keyword_count}, "
              f"有时间吗={quality_score.has_time}, "
              f"有动作吗={quality_score.has_action}, "
              f"有速度吗={quality_score.has_speed}, "
              f"有建筑吗={quality_score.has_building}, "
              f"总分={quality_score.total_score:.2f}, "
              f"有效={quality_score.is_valid}")
        
        if not quality_score.is_valid:
            print(f"OCR文本质量不足，跳过: {ocr_text[:30]}...")
            return None
        
        # 2. 检查是否正在处理或最近刚处理过
        current_time = time.time()
        if self.llm_processing:
            print("LLM正在处理上一个请求，跳过新请求")
            return None
        
        if current_time - self.last_llm_request_time < self.min_llm_interval:
            print(f"距离上次LLM请求间隔太短，跳过")
            return None
        
        # 3. 检查是否与上次相同（避免重复处理）
        if self.last_valid_ocr_text and self._is_similar_text(ocr_text, self.last_valid_ocr_text):
            print("OCR文本与上次相似，跳过重复处理")
            return None
        
        # 4. 异步调用LLM
        print(f"启动异步LLM处理: {ocr_text}")
        self.last_valid_ocr_text = ocr_text
        self.last_llm_request_time = current_time
        self._start_async_llm_processing(ocr_text)
        
        # 5. 立即返回None，让车辆刹停
        return None
    
    def get_llm_result(self) -> Optional[TaskInfo]:
        """
        获取异步LLM处理的结果（非阻塞）
        
        Returns:
            TaskInfo: 如果有结果则返回，否则返回None
        """
        if not self.llm_result_queue.empty():
            result = self.llm_result_queue.get_nowait()
            self.llm_processing = False
            return result
        return None
    
    def _is_similar_text(self, text1: str, text2: str, threshold: float = 0.85) -> bool:
        """判断两个文本是否相似"""
        normalized1 = self._normalize_text(text1)
        normalized2 = self._normalize_text(text2)
        similarity = SequenceMatcher(None, normalized1, normalized2).ratio()
        return similarity > threshold
    
    def _start_async_llm_processing(self, ocr_text: str):
        """启动异步LLM处理线程"""
        if self.llm_processing:
            return
        
        self.llm_processing = True
        self.llm_thread = threading.Thread(
            target=self._async_llm_worker,
            args=(ocr_text,),
            daemon=True
        )
        self.llm_thread.start()
    
    def _async_llm_worker(self, ocr_text: str):
        """异步LLM处理工作线程"""
        try:
            # 清理文本
            clean_text = ocr_text.strip().replace('\n', ' ').replace('\r', '')
            normalized_ocr = self._normalize_text(clean_text)
            
            # 构建prompt
            prompt = self._build_parse_prompt(normalized_ocr, clean_text)
            
            # 调用Ollama
            llm_response = self._call_ollama(prompt)
            
            if llm_response:
                # 解析LLM响应
                task_info = self._parse_llm_response(llm_response, clean_text)
                if task_info:
                    self.llm_result_queue.put(task_info)
                    print(f"LLM异步处理完成，任务已加入队列")
                else:
                    print("LLM解析失败，尝试备用方法")
                    # 尝试备用解析
                    fallback_result = self._fallback_parse(clean_text)
                    if fallback_result:
                        self.llm_result_queue.put(fallback_result)
            else:
                print("LLM无响应，使用备用解析")
                fallback_result = self._fallback_parse(clean_text)
                if fallback_result:
                    self.llm_result_queue.put(fallback_result)
            
        except Exception as e:
            print(f"异步LLM处理出错: {e}")
        finally:
            self.llm_processing = False
    
    def _normalize_text(self, text: str) -> str:
        """标准化文本：去除空格、特殊字符等"""
        normalized = text.replace(' ', '').replace('　', '')
        normalized = normalized.replace('·', '').replace('•', '').replace('.', '')
        normalized = normalized.replace('-', '').replace('—', '').replace('–', '').replace('―', '')
        normalized = normalized.replace('_', '')  # .replace('/', '')
        return normalized
    
    def _test_ollama_connection(self):
        """测试Ollama服务是否可用"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
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
        """构建建筑物名称映射表"""
        name_map = {}
        for building in self.buildings:
            name_map[building.name] = building.name
            normalized = self._normalize_text(building.name)
            name_map[normalized] = building.name
            
            keywords = self._extract_keywords(building.name)
            for keyword in keywords:
                if keyword not in name_map:
                    name_map[keyword] = building.name
                normalized_keyword = self._normalize_text(keyword)
                if normalized_keyword not in name_map:
                    name_map[normalized_keyword] = building.name
        
        return name_map
    
    def _extract_keywords(self, building_name: str) -> List[str]:
        """从建筑物名称中提取关键词"""
        keywords = []
        chinese_name = re.findall(r'[\u4e00-\u9fff]+', building_name)
        keywords.extend(chinese_name)
        alphanumeric = re.findall(r'[A-Za-z]\d*栋?', building_name)
        keywords.extend(alphanumeric)
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
                    "keep_alive": -1,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
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
    
    def _build_parse_prompt(self, normalized_ocr: str, original_ocr: str) -> str:
        """构建解析任务的prompt（仅使用主要目的地，方向作为后处理）"""
        buildings_str = "、".join(self.normalized_building_names)
        
        prompt = f"""你是一个智能助手，需要从OCR识别的文字中提取驾驶任务信息，带纠错。

OCR文字："{normalized_ocr}"
可用的建筑物名称：{buildings_str}（这是地图上所有有效的位置名称。destination 必须是这个列表中的一个；若语句出现“某建筑的东/西/南/北边”或近似意思，则 destination=该建筑名称，direction=对应方位。）

请分析这段文字，提取以下信息：
1. time_limit: 时间限制（秒数）
2. destination: 主要目的地（必须是上述建筑物之一）
3. sub_destination: 子目的地（如"一单元"、"二单元"、"门诊"等）
4. action: 动作（通常是"停车"或"到达"）
5. direction: 相对方位（默认为空，如果OCR文字中包含相对方位（"东"/"西"/"南"/"北"）时，输出"东"/"西"/"南"/"北"之一）

请以JSON格式返回：
{{{{
    "time_limit": 数字,
    "destination": "建筑物名称",
    "sub_destination": "子位置",
    "action": "动作",
    "direction": "东/西/南/北或空字符串"
}}}}

只返回JSON，不要有其他说明。"""
        
        # print("输入到模型的prompt:", prompt)
        return prompt
    
    def _parse_llm_response(self, llm_response: str, original_text: str) -> Optional[TaskInfo]:
        """解析LLM的响应"""
        try:
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if not json_match:
                return None
            
            json_str = json_match.group(0)
            data = json.loads(json_str)

            print("data:", data)
            
            if not data or 'destination' not in data:
                return None
            
            time_limit = data.get('time_limit', 60)
            destination = data.get('destination', '')
            sub_destination = data.get('sub_destination', '')
            action = data.get('action', '停车')
            direction = data.get('direction', '') or ''
            
            # 匹配建筑物
            matched_building = self._fuzzy_match_building(destination) if destination else None
            if not matched_building:
                return None
            
            print(f"LLM成功解析任务:")
            print(f"  时间: {time_limit}秒")
            print(f"  目的地: {matched_building}")
            print(f"  子目的地: {sub_destination}")
            print(f"  动作: {action}")
            if direction:
                print(f"  方位: {direction}")
            
            return TaskInfo(
                ocr_text=original_text,
                time_limit=int(time_limit),
                destination=matched_building,
                sub_destination=sub_destination if sub_destination else "",
                action=action,
                time_phrase=f"{time_limit}秒内",
                location_phrase=f"{matched_building}{sub_destination}" if sub_destination else matched_building,
                action_phrase=action,
                direction=direction
            )
            
        except Exception as e:
            print(f"解析LLM响应时出错: {e}")
            return None
    
    def _fuzzy_match_building(self, location_text: str) -> Optional[str]:
        """模糊匹配建筑物名称"""
        normalized_input = self._normalize_text(location_text)
        
        if normalized_input in self.normalized_to_original:
            return self.normalized_to_original[normalized_input]
        
        if location_text in self.building_name_map:
            return self.building_name_map[location_text]
        if normalized_input in self.building_name_map:
            return self.building_name_map[normalized_input]
        
        # 传统模糊匹配
        best_match = None
        best_score = 0
        
        for normalized_name, original_name in self.normalized_to_original.items():
            similarity = SequenceMatcher(None, normalized_input, normalized_name).ratio()
            if similarity > best_score and similarity > 0.6:
                best_score = similarity
                best_match = original_name
        
        if best_match:
            print(f"模糊匹配: '{location_text}' -> '{best_match}'")
            return best_match
        
        return None
    
    def _fallback_parse(self, text: str) -> Optional[TaskInfo]:
        """备用的规则解析方法"""
        normalized_text = self._normalize_text(text)
        
        # 优先匹配相对方位：“X秒内去到{目的地}{方向}(边|侧)那?栋?楼(停车)”
        dir_patterns = [
            (r'(\d+)秒内去?到(.+?)(东|西|南|北)[边侧]那?栋?楼', '停车'),
            (r'(\d+)秒内到达(.+?)(东|西|南|北)[边侧]那?栋?楼', '到达'),
        ]
        for pattern, default_action in dir_patterns:
            m = re.search(pattern, normalized_text)
            if m:
                time_limit = int(m.group(1))
                dest_text = m.group(2).strip()
                direction = m.group(3)
                matched_building = self._fuzzy_match_building(dest_text)
                if matched_building:
                    return TaskInfo(
                        ocr_text=text,
                        time_limit=time_limit,
                        destination=matched_building,
                        sub_destination="",
                        action=default_action,
                        time_phrase=f"{time_limit}秒内",
                        location_phrase=f"{matched_building}{direction}侧",
                        action_phrase=default_action,
                        direction=direction
                    )
        
        patterns = [
            (r'(\d+)秒内去?到(.+?)停车', '停车'),
            (r'(\d+)秒内去?到(.+)', '停车'),
            (r'(\d+)秒内找到(.+?)停车', '停车'),
            (r'(\d+)秒内到达(.+)', '到达'),
        ]
        
        for pattern, default_action in patterns:
            match = re.search(pattern, normalized_text)
            if match:
                time_limit = int(match.group(1))
                location_full = match.group(2).strip()
                
                main_destination, sub_destination = self._split_destination(location_full)
                
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
                        action_phrase=default_action,
                        direction=""
                    )
        
        return None

    def reset_cache(self) -> None:
        """清空处理器内部的OCR相关缓存/队列/状态。"""
        try:
            self.last_valid_ocr_text = None
            self.llm_processing = False
            # 清空队列
            while not self.llm_result_queue.empty():
                try:
                    self.llm_result_queue.get_nowait()
                except Exception:
                    break
            self.last_llm_request_time = 0
        except Exception:
            pass
    
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
        """检查是否找到了详细位置（加入混淆表与模糊匹配容错）"""
        def generate_confusion_variants(s: str) -> List[str]:
            variants = {s}
            for a, b in getattr(self, "detail_confusion_pairs", []):
                if a in s:
                    variants.add(s.replace(a, b))
                if b in s:
                    variants.add(s.replace(b, a))
            return list(variants)

        normalized_target = self._normalize_text(target_detail)
        target_variants = [self._normalize_text(v) for v in generate_confusion_variants(normalized_target)]
        
        for text in ocr_texts:
            normalized_text = self._normalize_text(text)
            text_variants = [self._normalize_text(v) for v in generate_confusion_variants(normalized_text)]
            # 1) 变体包含判定（双向）
            for tv in target_variants:
                for ntv in text_variants:
                    if tv and ntv and (tv in ntv or ntv in tv):
                        return True
            # # 2) 相似度兜底
            # try:
            #     from difflib import SequenceMatcher as _SM
            #     for tv in target_variants:
            #         if _SM(None, tv, normalized_text).ratio() > 0.85:
            #             return True
            # except Exception:
            #     pass
        
        return False