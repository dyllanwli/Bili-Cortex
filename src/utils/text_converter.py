#!/usr/bin/env python3
"""
简化的繁简体转换工具
集成到转录流程中，支持自动转换
"""

import logging

# 简化的繁简对照表 - 只保留常用字符
TRADITIONAL_TO_SIMPLIFIED = {
    # 基础常用字符
    '這': '这', '個': '个', '來': '来', '過': '过', '時': '时', '間': '间',
    '問': '问', '題': '题', '還': '还', '會': '会', '後': '后', '現': '现',
    '開': '开', '關': '关', '長': '长', '說': '说', '見': '见', '聽': '听',
    '買': '买', '賣': '卖', '錢': '钱', '業': '业', '專': '专', '學': '学',
    '經': '经', '營': '营', '場': '场', '機': '机', '種': '种', '類': '类',
    '計': '计', '劃': '划', '進': '进', '選': '选', '擇': '择', '決': '决',
    '確': '确', '實': '实', '際': '际', '況': '况', '標': '标', '準': '准',
    '語': '语', '詞': '词', '資': '资', '產': '产', '應': '应', '該': '该',
    '無': '无', '沒': '没', '離': '离', '達': '达', '連': '连', '運': '运',
    '轉': '转', '變': '变', '網': '网', '頭': '头', '臉': '脸', '導': '导',
    '師': '师', '領': '领', '創': '创', '負': '负', '責': '责', '課': '课',
    '測': '测', '試': '试', '驗': '验', '證': '证', '識': '识', '認': '认',
    '為': '为', '處': '处', '給': '给', '讓': '让', '帶': '带', '從': '从',
    '將': '将', '於': '于', '與': '与', '對': '对', '幾': '几', '樣': '样',
    '麼': '么', '車': '车', '軟': '软', '體': '体', '係': '系', '統': '统',
    '電': '电', '腦': '脑', '視': '视', '頻': '频', '聲': '声', '響': '响',
}

# 反向映射
SIMPLIFIED_TO_TRADITIONAL = {v: k for k, v in TRADITIONAL_TO_SIMPLIFIED.items()}


class TextConverter:
    """简化的文本繁简转换器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def to_simplified(self, text: str) -> str:
        """繁体转简体"""
        if not text:
            return text
            
        result = text
        for trad, simp in TRADITIONAL_TO_SIMPLIFIED.items():
            result = result.replace(trad, simp)
        
        return result
    
    def to_traditional(self, text: str) -> str:
        """简体转繁体"""
        if not text:
            return text
            
        result = text
        for simp, trad in SIMPLIFIED_TO_TRADITIONAL.items():
            result = result.replace(simp, trad)
        
        return result


# 全局实例
text_converter = TextConverter()


def convert_text(text: str, target_format: str = 'simplified') -> str:
    """便捷转换函数"""
    if target_format == 'traditional':
        return text_converter.to_traditional(text)
    else:
        return text_converter.to_simplified(text)


# 兼容之前的函数名
def convert_traditional_to_simplified(text: str) -> str:
    """繁体转简体（兼容函数）"""
    return text_converter.to_simplified(text)


def convert_simplified_to_traditional(text: str) -> str:
    """简体转繁体（兼容函数）"""
    return text_converter.to_traditional(text)