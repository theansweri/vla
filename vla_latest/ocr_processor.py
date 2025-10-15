"""
OCR文字处理模块
负责OCR文字识别、纠错和任务解析
"""

import re
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from task_types import TaskInfo


class OCRProcessor:
    """OCR文字处理器"""
    
    def __init__(self, buildings):
        self.buildings = buildings
        
        # OCR字符替换表（用于处理识别错误）
        self.ocr_confusion_matrix = [
            ['8', 'B'],
        ]
        
        # 建筑物名称映射表（用于模糊匹配）
        self.building_name_map = self._build_name_mapping()
        
        # 任务模板（用于识别常见任务类型）
        self.task_patterns = [
            (r'(\d+)秒内去到(.+?)停车', '停车'),
            (r'(\d+)秒内去到(.+)', '停车'),
            (r'(\d+)秒内找到(.+?)停车', '停车'),
            (r'(\d+)秒内到达(.+)', '到达'),
            (r'(\d+)秒内(.+?)停车', '停车'),
            (r'(\d+)秒内(.+?)', '停车'),
        ]
    
    def _build_name_mapping(self) -> Dict[str, str]:
        """构建建筑物名称映射表，用于模糊匹配"""
        name_map = {}
        for building in self.buildings:
            # 原始名称
            name_map[building.name] = building.name
            # 去除空格的版本
            clean_name = building.name.replace(' ', '').replace('　', '')
            name_map[clean_name] = building.name
            # 提取关键词（如"锦汇华庭"、"A栋"等）
            keywords = self._extract_keywords(building.name)
            for keyword in keywords:
                if keyword not in name_map:
                    name_map[keyword] = building.name
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
        return keywords
    
    def _generate_text_variants(self, text: str) -> List[str]:
        """生成文本的所有可能变体（基于OCR混淆矩阵）"""
        variants = [text]
        
        for confusion_group in self.ocr_confusion_matrix:
            new_variants = []
            for variant in variants:
                for char in confusion_group:
                    if char in variant:
                        # 替换每个混淆字符
                        for replacement in confusion_group:
                            if replacement != char:
                                new_text = variant.replace(char, replacement)
                                if new_text not in new_variants and new_text not in variants:
                                    new_variants.append(new_text)
            variants.extend(new_variants)
        
        return variants
    
    def parse_task_instruction(self, ocr_text: str) -> Optional[TaskInfo]:
        """解析任务指令（包含OCR错误处理和子目的地提取）"""
        # 清理文本
        clean_text = ocr_text.strip().replace('\n', '').replace('\r', '')
        
        # 生成文本变体
        text_variants = self._generate_text_variants(clean_text)
        print(f"生成 {len(text_variants)} 个文本变体用于匹配")
        
        for variant in text_variants:
            for pattern, default_action in self.task_patterns:
                print(f"尝试匹配模式: '{pattern}' 对变体: '{variant}'")
                match = re.search(pattern, variant)
                if match:
                    time_limit = int(match.group(1))
                    location_full = match.group(2).strip()
                    
                    # 提取时间片段
                    time_phrase = f"{time_limit}秒内"
                    
                    # 分离主目的地和子目的地
                    main_destination, sub_destination = self._split_destination(location_full)
                    
                    # 匹配建筑物（主目的地）
                    matched_building = self._fuzzy_match_building(main_destination)
                    if not matched_building:
                        continue  # 尝试下一个变体
                    
                    print(f"成功解析任务 - 原文:'{ocr_text}'")
                    print(f"  变体:'{variant}'")
                    print(f"  时间:{time_limit}秒")
                    print(f"  主目的地:{matched_building}")
                    print(f"  子目的地:{sub_destination}")
                    print(f"  动作:{default_action}")
                    
                    return TaskInfo(
                        ocr_text=clean_text,
                        time_limit=time_limit,
                        destination=matched_building,
                        sub_destination=sub_destination,
                        action=default_action,
                        time_phrase=time_phrase,
                        location_phrase=location_full,
                        action_phrase=default_action
                    )
        
        return None
    
    def _split_destination(self, location_text: str) -> Tuple[str, str]:
        """分离主目的地和子目的地"""
        # 子目的地模式
        sub_patterns = [
            r'([一二三四五六七八九十\d]+单元)',
            r'([一二三四五六七八九十\d]+号楼)',
            r'(门口)',
            r'(入口)',
            r'(大厅)',
            r'([一二三四五六七八九十\d]+楼)',
        ]
        
        sub_destination = ""
        main_destination = location_text
        
        # 查找并提取子目的地
        for pattern in sub_patterns:
            match = re.search(pattern, location_text)
            if match:
                sub_destination = match.group(1)
                # 从原文中移除子目的地部分，得到主目的地
                main_destination = location_text[:match.start()] + location_text[match.end():]
                main_destination = main_destination.strip()
                break
        
        return main_destination, sub_destination
    
    def _fuzzy_match_building(self, location_text: str) -> Optional[str]:
        """模糊匹配建筑物名称（包含OCR错误处理）"""
        # 生成所有可能的文本变体
        text_variants = self._generate_text_variants(location_text)
        
        # 对每个变体尝试匹配
        for variant in text_variants:
            # 首先尝试直接匹配
            if variant in self.building_name_map:
                print(f"OCR纠错：'{location_text}' -> '{variant}' -> '{self.building_name_map[variant]}'")
                return self.building_name_map[variant]
            
            # 清理变体文本（去除空格等）
            clean_variant = variant.replace(' ', '').replace('　', '')
            if clean_variant in self.building_name_map:
                print(f"OCR纠错：'{location_text}' -> '{clean_variant}' -> '{self.building_name_map[clean_variant]}'")
                return self.building_name_map[clean_variant]
        
        # 如果所有变体都无法直接匹配，进行模糊匹配
        best_match = None
        best_score = 0
        best_variant = None
        
        for variant in text_variants:
            for building in self.buildings:
                # 检查是否包含关键词
                building_keywords = self._extract_keywords(building.name)
                variant_keywords = self._extract_keywords(variant)
                
                # 计算匹配度
                match_count = 0
                for var_keyword in variant_keywords:
                    for build_keyword in building_keywords:
                        # 也对关键词进行变体匹配
                        keyword_variants = self._generate_text_variants(var_keyword)
                        for kv in keyword_variants:
                            similarity = SequenceMatcher(None, kv, build_keyword).ratio()
                            if similarity > 0.8:
                                match_count += 1
                                break
                
                if match_count > 0:
                    score = match_count / max(len(variant_keywords), 1)
                    if score > best_score:
                        best_score = score
                        best_match = building.name
                        best_variant = variant
        
        if best_match and best_score > 0.5:
            print(f"OCR模糊匹配：'{location_text}' -> '{best_variant}' -> '{best_match}' (得分: {best_score:.2f})")
            return best_match
        
        return None
    
    def check_detail_location_found(self, ocr_texts: List[str], target_detail: str) -> bool:
        """检查是否找到了详细位置（如"二单元"）"""
        for text in ocr_texts:
            if target_detail in text:
                return True
        return False