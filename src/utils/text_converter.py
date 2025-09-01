#!/usr/bin/env python3
"""
繁简体转换工具
集成到转录流程中，支持自动转换
"""

import logging
from typing import Dict

# 扩展的繁简对照表
TRADITIONAL_TO_SIMPLIFIED = {
    # 基础字符
    '這': '这', '個': '个', '來': '来', '過': '过', '時': '时', '間': '间',
    '問': '问', '題': '题', '還': '还', '會': '会', '後': '后', '現': '现',
    '開': '开', '關': '关', '長': '长', '說': '说', '見': '见', '聽': '听',
    '買': '买', '賣': '卖', '錢': '钱', '業': '业', '專': '专', '學': '学',
    '習': '习', '練': '练', '經': '经', '營': '营', '場': '场', '機': '机',
    '種': '种', '類': '类', '別': '别', '計': '计', '劃': '划', '進': '进',
    '選': '选', '擇': '择', '決': '决', '確': '确', '實': '实', '際': '际',
    '況': '况', '標': '标', '準': '准', '則': '则', '規': '规', '務': '务',
    '議': '议', '論': '论', '講': '讲', '話': '话', '語': '语', '詞': '词',
    
    # 金融交易相关
    '資': '资', '產': '产', '貨': '货', '幣': '币', '險': '险', '價': '价',
    '漲': '涨', '跌': '跌', '盤': '盘', '線': '线', '圖': '图', '勢': '势',
    '動': '动', '靜': '静', '穩': '稳', '亂': '乱', '報': '报', '導': '导',
    '術': '术', '術': '术', '單': '单', '雙': '双', '復': '复', '團': '团',
    '員': '员', '層': '层', '級': '级', '組': '组', '織': '织', '構': '构',
    
    # 常用词汇
    '應': '应', '該': '该', '無': '无', '沒': '没', '離': '离', '達': '达',
    '連': '连', '運': '运', '轉': '转', '變': '变', '態': '态', '網': '网',
    '頭': '头', '臉': '脸', '導': '导', '師': '师', '領': '领', '導': '导',
    '創': '创', '負': '负', '責': '责', '任': '任', '務': '务', '功': '功',
    '課': '课', '題': '题', '測': '测', '試': '试', '驗': '验', '證': '证',
    '識': '识', '別': '别', '認': '认', '為': '为', '處': '处', '理': '理',
    
    # 数量词
    '個': '个', '隻': '只', '條': '条', '張': '张', '層': '层', '點': '点',
    '線': '线', '邊': '边', '側': '侧', '內': '内', '外': '外', '間': '间',
    
    # 动作词
    '給': '给', '讓': '让', '帶': '带', '拿': '拿', '從': '从', '將': '将',
    '錶': '表', '錶': '表', '録': '录', '記': '记', '憶': '忆', '忘': '忘',
    
    # 其他常用
    '於': '于', '與': '与', '對': '对', '幾': '几', '樣': '样', '麼': '么',
    '廠': '厂', '歷': '历', '車': '车', '軟': '软', '硬': '硬', '體': '体',
    '係': '系', '統': '统', '網': '网', '路': '路', '電': '电', '腦': '脑',
    '視': '视', '頻': '频', '聲': '声', '音': '音', '響': '响', '號': '号',
}

# 反向映射
SIMPLIFIED_TO_TRADITIONAL = {v: k for k, v in TRADITIONAL_TO_SIMPLIFIED.items()}


class TextConverter:
    """文本繁简转换器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def to_simplified(self, text: str) -> str:
        """繁体转简体"""
        if not text:
            return text
            
        result = text
        converted_count = 0
        
        for trad, simp in TRADITIONAL_TO_SIMPLIFIED.items():
            if trad in result:
                result = result.replace(trad, simp)
                converted_count += 1
        
        if converted_count > 0:
            self.logger.debug(f"Converted {converted_count} traditional characters to simplified")
        
        return result
    
    def to_traditional(self, text: str) -> str:
        """简体转繁体"""
        if not text:
            return text
            
        result = text
        converted_count = 0
        
        for simp, trad in SIMPLIFIED_TO_TRADITIONAL.items():
            if simp in result:
                result = result.replace(simp, trad)
                converted_count += 1
        
        if converted_count > 0:
            self.logger.debug(f"Converted {converted_count} simplified characters to traditional")
        
        return result
    
    def detect_script_type(self, text: str) -> str:
        """检测文本主要是简体还是繁体"""
        if not text:
            return 'unknown'
        
        traditional_count = 0
        simplified_count = 0
        
        # 统计繁体字符
        for char in TRADITIONAL_TO_SIMPLIFIED.keys():
            traditional_count += text.count(char)
        
        # 统计简体字符  
        for char in SIMPLIFIED_TO_TRADITIONAL.keys():
            simplified_count += text.count(char)
        
        if traditional_count > simplified_count:
            return 'traditional'
        elif simplified_count > traditional_count:
            return 'simplified'
        else:
            return 'mixed'
    
    def smart_convert(self, text: str, target_format: str = 'simplified') -> str:
        """智能转换 - 根据目标格式自动转换"""
        if not text or target_format == 'auto':
            return text
        
        current_type = self.detect_script_type(text)
        
        if target_format == 'simplified':
            if current_type == 'traditional' or current_type == 'mixed':
                return self.to_simplified(text)
            return text
        
        elif target_format == 'traditional':
            if current_type == 'simplified' or current_type == 'mixed':
                return self.to_traditional(text)
            return text
        
        return text
    
    def get_conversion_stats(self, original: str, converted: str) -> Dict[str, int]:
        """获取转换统计信息"""
        if original == converted:
            return {'changed': 0, 'total_chars': len(original)}
        
        changed_count = 0
        for i, (o, c) in enumerate(zip(original, converted)):
            if o != c:
                changed_count += 1
        
        return {
            'changed': changed_count,
            'total_chars': len(original),
            'change_rate': changed_count / len(original) if original else 0
        }


# 全局实例
text_converter = TextConverter()


def convert_text(text: str, target_format: str = 'simplified') -> str:
    """便捷转换函数"""
    return text_converter.smart_convert(text, target_format)


# 兼容之前的函数名
def convert_traditional_to_simplified(text: str) -> str:
    """繁体转简体（兼容函数）"""
    return text_converter.to_simplified(text)


def convert_simplified_to_traditional(text: str) -> str:
    """简体转繁体（兼容函数）"""
    return text_converter.to_traditional(text)