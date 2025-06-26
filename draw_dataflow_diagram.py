import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def create_dataflow_diagrams():
    """创建StockPredict数据流详解图表"""
    
    print("正在创建StockPredict数据流详解图表...")
    
    try:
        # 创建DrawIO格式的XML
        create_dataflow_drawio()
        
        # 创建Visio绘制指南
        create_dataflow_guide()
        
        # 创建基础VSDX文件
        create_dataflow_vsdx()
        
        print("\n✅ StockPredict数据流图表创建成功！")
        print("生成的文件:")
        print("1. StockPredict_DataFlow.drawio - DrawIO格式数据流图")
        print("2. DataFlow_Guide.txt - 数据流图绘制指南")
        print("3. StockPredict_DataFlow.vsdx - Visio格式文件")
        
    except Exception as e:
        print(f"创建文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

def create_dataflow_drawio():
    """创建DrawIO格式的数据流图"""
    
    print("🔄 正在创建数据流DrawIO文件...")
    
    # 完整的DrawIO XML内容
    drawio_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-01-01T00:00:00.000Z" agent="Python Script" version="24.0.0">
  <diagram name="StockPredict Data Flow" id="dataflow">
    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1600" pageHeight="1200" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- 标题 -->
        <mxCell id="title" value="StockPredict 模型数据流详解" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=20;fontStyle=1;fontColor=#2E3440" vertex="1" parent="1">
          <mxGeometry x="550" y="30" width="500" height="40" as="geometry" />
        </mxCell>
        
        <!-- 第1步：输入层 -->
        <mxCell id="step1_title" value="第1步：输入层 (Input Layer)" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=14;fontStyle=1;fontColor=#1976D2" vertex="1" parent="1">
          <mxGeometry x="50" y="100" width="200" height="30" as="geometry" />
        </mxCell>
        
        <mxCell id="input_data" value="原始数据集&#xa;Raw Dataset" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1976D2;fontSize=12;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="80" y="140" width="120" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="input_tensor" value="输入张量&#xa;(N, T, F)&#xa;N=561股票&#xa;T=60天&#xa;F=5特征(OHLCV)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#BBDEFB;strokeColor=#1976D2;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="80" y="240" width="120" height="80" as="geometry" />
        </mxCell>
        
        <!-- 第2步：时间特征提取 -->
        <mxCell id="step2_title" value="第2步：时间特征提取 (Temporal Feature Extraction)" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=14;fontStyle=1;fontColor=#FF5722" vertex="1" parent="1">
          <mxGeometry x="300" y="100" width="350" height="30" as="geometry" />
        </mxCell>
        
        <!-- 多尺度分支 -->
        <mxCell id="branch1" value="分支1: 原始尺度&#xa;Identity Path&#xa;(T, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCCBC;strokeColor=#FF5722;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="300" y="140" width="100" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="branch2" value="分支2: 卷积尺度1&#xa;Conv1D(s=2)&#xa;(T/2, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FF8A65;strokeColor=#FF5722;fontSize=10;fontStyle=1;fontColor=white" vertex="1" parent="1">
          <mxGeometry x="300" y="220" width="100" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="branch3" value="分支3: 卷积尺度2&#xa;Conv1D(s=4)&#xa;(T/4, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FF5722;strokeColor=#D84315;fontSize=10;fontStyle=1;fontColor=white" vertex="1" parent="1">
          <mxGeometry x="300" y="300" width="100" height="60" as="geometry" />
        </mxCell>
        
        <!-- 特征拼接 -->
        <mxCell id="concat" value="特征拼接&#xa;Concatenation&#xa;(N, T_cat, D_out)&#xa;T_cat=T+T/2+T/4" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E1BEE7;strokeColor=#9C27B0;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="450" y="200" width="120" height="80" as="geometry" />
        </mxCell>
        
        <!-- TriU网络 -->
        <mxCell id="triu" value="TriU Network&#xa;时序信息聚合&#xa;可学习上三角矩阵" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#BCAAA4;strokeColor=#795548;fontSize=10;fontStyle=1;fontColor=white" vertex="1" parent="1">
          <mxGeometry x="450" y="320" width="120" height="60" as="geometry" />
        </mxCell>
        
        <!-- 第3步：通道压缩 -->
        <mxCell id="step3_title" value="第3步：通道压缩 (Channel Compression)" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=14;fontStyle=1;fontColor=#4CAF50" vertex="1" parent="1">
          <mxGeometry x="650" y="100" width="280" height="30" as="geometry" />
        </mxCell>
        
        <mxCell id="channel_fc" value="Channel FC&#xa;Linear(D_out → 1)&#xa;通道压缩" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#C8E6C9;strokeColor=#4CAF50;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="680" y="180" width="120" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="compressed" value="压缩后特征&#xa;(N, T_cat)&#xa;每股票一个长向量" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#A5D6A7;strokeColor=#4CAF50;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="680" y="280" width="120" height="60" as="geometry" />
        </mxCell>
        
        <!-- 第4步：双路径预测 -->
        <mxCell id="step4_title" value="第4步：双路径预测 (Dual-Path Prediction)" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=14;fontStyle=1;fontColor=#9C27B0" vertex="1" parent="1">
          <mxGeometry x="900" y="100" width="300" height="30" as="geometry" />
        </mxCell>
        
        <!-- 路径A -->
        <mxCell id="pathA_title" value="路径A: 纯时间聚合预测" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=12;fontStyle=1;fontColor=#FF9800" vertex="1" parent="1">
          <mxGeometry x="880" y="140" width="160" height="30" as="geometry" />
        </mxCell>
        
        <mxCell id="pathA_fc" value="Time Aggregation&#xa;Linear(T_cat → 1)&#xa;个股时序聚合" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFE0B2;strokeColor=#FF9800;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="900" y="180" width="120" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="pred1" value="预测1 (Pred_1)&#xa;(N, 1)&#xa;基础预测结果" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCC02;strokeColor=#FF8F00;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="900" y="280" width="120" height="60" as="geometry" />
        </mxCell>
        
        <!-- 路径B -->
        <mxCell id="pathB_title" value="路径B: 股票间交互预测" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=12;fontStyle=1;fontColor=#673AB7" vertex="1" parent="1">
          <mxGeometry x="880" y="380" width="160" height="30" as="geometry" />
        </mxCell>
        
        <mxCell id="attention_mixer" value="StockAttentionMixer&#xa;• LayerNorm&#xa;• Industry-Sparse Attention&#xa;• Residual Connection&#xa;• FFN" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#D1C4E9;strokeColor=#673AB7;fontSize=9;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="900" y="420" width="120" height="100" as="geometry" />
        </mxCell>
        
        <mxCell id="pathB_fc" value="Final Aggregation&#xa;Linear(T_cat → 1)&#xa;交互后聚合" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#B39DDB;strokeColor=#673AB7;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="900" y="540" width="120" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="pred2" value="预测2 (Pred_2)&#xa;(N, 1)&#xa;市场关联预测" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#9575CD;strokeColor=#512DA8;fontSize=10;fontStyle=1;fontColor=white" vertex="1" parent="1">
          <mxGeometry x="900" y="620" width="120" height="60" as="geometry" />
        </mxCell>
        
        <!-- 第5步：输出融合 -->
        <mxCell id="step5_title" value="第5步：输出融合 (Output Fusion)" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=14;fontStyle=1;fontColor=#F44336" vertex="1" parent="1">
          <mxGeometry x="1200" y="100" width="250" height="30" as="geometry" />
        </mxCell>
        
        <mxCell id="fusion" value="输出融合&#xa;Pred_1 + Pred_2&#xa;逐元素相加" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFCDD2;strokeColor=#F44336;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="1250" y="400" width="120" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="final_output" value="最终预测&#xa;(N, 1)&#xa;股票收益率预测" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F44336;strokeColor=#D32F2F;fontSize=11;fontStyle=1;fontColor=white" vertex="1" parent="1">
          <mxGeometry x="1250" y="500" width="120" height="60" as="geometry" />
        </mxCell>
        
        <!-- 连接箭头 -->
        <!-- 输入到时间特征提取 -->
        <mxCell id="arrow1" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#1976D2" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="200" y="280" as="sourcePoint" />
            <mxPoint x="300" y="220" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 多尺度分支到拼接 -->
        <mxCell id="arrow2" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#FF5722" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="400" y="170" as="sourcePoint" />
            <mxPoint x="450" y="220" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow3" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#FF5722" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="400" y="250" as="sourcePoint" />
            <mxPoint x="450" y="240" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow4" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#FF5722" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="400" y="330" as="sourcePoint" />
            <mxPoint x="450" y="260" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 拼接到TriU -->
        <mxCell id="arrow5" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#9C27B0" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="510" y="280" as="sourcePoint" />
            <mxPoint x="510" y="320" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- TriU到通道压缩 -->
        <mxCell id="arrow6" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#4CAF50" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="570" y="350" as="sourcePoint" />
            <mxPoint x="680" y="210" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 通道压缩到压缩特征 -->
        <mxCell id="arrow7" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#4CAF50" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="740" y="240" as="sourcePoint" />
            <mxPoint x="740" y="280" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 压缩特征到双路径 -->
        <mxCell id="arrow8" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#FF9800" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="800" y="300" as="sourcePoint" />
            <mxPoint x="900" y="210" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow9" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#673AB7" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="800" y="320" as="sourcePoint" />
            <mxPoint x="900" y="460" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 路径A内部连接 -->
        <mxCell id="arrow10" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#FF9800" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="960" y="240" as="sourcePoint" />
            <mxPoint x="960" y="280" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 路径B内部连接 -->
        <mxCell id="arrow11" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#673AB7" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="960" y="520" as="sourcePoint" />
            <mxPoint x="960" y="540" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow12" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#673AB7" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="960" y="600" as="sourcePoint" />
            <mxPoint x="960" y="620" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 双路径到融合 -->
        <mxCell id="arrow13" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#F44336" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="1020" y="310" as="sourcePoint" />
            <mxPoint x="1250" y="420" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow14" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=2;strokeColor=#F44336" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="1020" y="650" as="sourcePoint" />
            <mxPoint x="1250" y="440" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 融合到最终输出 -->
        <mxCell id="arrow15" value="" style="endArrow=classic;html=1;rounded=0;strokeWidth=3;strokeColor=#F44336" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="1310" y="460" as="sourcePoint" />
            <mxPoint x="1310" y="500" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 关键创新点标注 -->
        <mxCell id="innovation1" value="💡 核心创新1:&#xa;多尺度时间特征提取" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFF8E1;strokeColor=#FFA000;fontSize=10;fontStyle=1;fontColor=#E65100" vertex="1" parent="1">
          <mxGeometry x="300" y="400" width="150" height="50" as="geometry" />
        </mxCell>
        
        <mxCell id="innovation2" value="💡 核心创新2:&#xa;行业稀疏注意力机制" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F3E5F5;strokeColor=#AB47BC;fontSize=10;fontStyle=1;fontColor=#4A148C" vertex="1" parent="1">
          <mxGeometry x="1050" y="450" width="150" height="50" as="geometry" />
        </mxCell>
        
        <mxCell id="innovation3" value="💡 核心创新3:&#xa;双路径预测融合" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFEBEE;strokeColor=#E57373;fontSize=10;fontStyle=1;fontColor=#C62828" vertex="1" parent="1">
          <mxGeometry x="1400" y="420" width="150" height="50" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
    
    try:
        with open("StockPredict_DataFlow.drawio", "w", encoding="utf-8") as f:
            f.write(drawio_xml)
        
        print("✅ 数据流DrawIO文件创建成功 - StockPredict_DataFlow.drawio")
        
    except Exception as e:
        print(f"创建DrawIO文件时发生错误: {e}")
        raise e

def create_dataflow_guide():
    """创建数据流图绘制指南"""
    
    guide_text = """
StockPredict 数据流详解图 - Visio 绘制指南
=======================================

本指南帮助您在Microsoft Visio中绘制StockPredict模型的完整数据流详解图。

整体布局设计
===========
- 采用从左到右的水平流程布局
- 5个主要步骤按顺序排列
- 第4步采用双路径并行设计
- 使用不同颜色区分各个处理阶段

第1步：输入层 (Input Layer)
=========================
颜色方案: 浅蓝色系 (#E3F2FD, #BBDEFB)

组件:
1. 原始数据集框 (浅蓝色矩形)
   - 文本: "原始数据集 Raw Dataset"
   - 位置: 左上角

2. 输入张量框 (中蓝色矩形)
   - 文本: "输入张量 (N, T, F)"
   - 详细标注: "N=561股票, T=60天, F=5特征(OHLCV)"
   - 位置: 第1步下方

连接: 垂直箭头连接两个组件

第2步：时间特征提取 (Temporal Feature Extraction)
=============================================
颜色方案: 橙红色系 (#FFCCBC, #FF8A65, #FF5722)

组件:
1. 三个并行分支 (不同深度的橙红色):
   - 分支1 (浅橙色): "原始尺度 Identity Path (T, F)"
   - 分支2 (中橙色): "卷积尺度1 Conv1D(s=2) (T/2, F)"
   - 分支3 (深橙色): "卷积尺度2 Conv1D(s=4) (T/4, F)"

2. 特征拼接框 (紫色):
   - 文本: "特征拼接 Concatenation"
   - 标注: "(N, T_cat, D_out) T_cat=T+T/2+T/4"

3. TriU网络框 (棕色):
   - 文本: "TriU Network 时序信息聚合"
   - 标注: "可学习上三角矩阵"

连接: 
- 输入层到三个分支的分叉箭头
- 三个分支到拼接框的汇聚箭头
- 拼接框到TriU网络的垂直箭头

第3步：通道压缩 (Channel Compression)
=================================
颜色方案: 绿色系 (#C8E6C9, #A5D6A7)

组件:
1. Channel FC框 (浅绿色):
   - 文本: "Channel FC Linear(D_out → 1)"
   - 标注: "通道压缩"

2. 压缩后特征框 (中绿色):
   - 文本: "压缩后特征 (N, T_cat)"
   - 标注: "每股票一个长向量"

连接: 
- TriU网络到Channel FC的对角箭头
- Channel FC到压缩特征的垂直箭头

第4步：双路径预测 (Dual-Path Prediction)
====================================
这是整个图表的核心部分，采用并行双路径设计。

路径A - 纯时间聚合预测:
颜色方案: 橙黄色系 (#FFE0B2, #FFCC02)

组件:
1. 路径A标题: "路径A: 纯时间聚合预测"
2. Time Aggregation框 (浅橙色):
   - 文本: "Time Aggregation Linear(T_cat → 1)"
   - 标注: "个股时序聚合"
3. 预测1框 (金黄色):
   - 文本: "预测1 (Pred_1) (N, 1)"
   - 标注: "基础预测结果"

路径B - 股票间交互预测:
颜色方案: 紫色系 (#D1C4E9, #B39DDB, #9575CD)

组件:
1. 路径B标题: "路径B: 股票间交互预测"
2. StockAttentionMixer框 (浅紫色):
   - 文本: "StockAttentionMixer"
   - 详细列表: "• LayerNorm • Industry-Sparse Attention • Residual Connection • FFN"
3. Final Aggregation框 (中紫色):
   - 文本: "Final Aggregation Linear(T_cat → 1)"
   - 标注: "交互后聚合"
4. 预测2框 (深紫色):
   - 文本: "预测2 (Pred_2) (N, 1)"
   - 标注: "市场关联预测"

连接:
- 压缩特征到两个路径的分叉箭头
- 每个路径内部的垂直连接箭头

第5步：输出融合 (Output Fusion)
=============================
颜色方案: 红色系 (#FFCDD2, #F44336)

组件:
1. 输出融合框 (浅红色):
   - 文本: "输出融合 Pred_1 + Pred_2"
   - 标注: "逐元素相加"

2. 最终输出框 (深红色):
   - 文本: "最终预测 (N, 1)"
   - 标注: "股票收益率预测"

连接:
- 两个预测结果到融合框的汇聚箭头
- 融合框到最终输出的粗箭头(强调最终结果)

核心创新点标注
=============
在图表适当位置添加三个创新点标注框:

1. 💡 核心创新1: 多尺度时间特征提取
   - 位置: 第2步附近
   - 颜色: 浅黄色 (#FFF8E1)

2. 💡 核心创新2: 行业稀疏注意力机制
   - 位置: 路径B附近
   - 颜色: 浅紫色 (#F3E5F5)

3. 💡 核心创新3: 双路径预测融合
   - 位置: 第5步附近
   - 颜色: 浅红色 (#FFEBEE)

完整颜色代码表
=============
- 输入层: #E3F2FD (浅蓝), #BBDEFB (中蓝)
- 时间特征提取: #FFCCBC (浅橙), #FF8A65 (中橙), #FF5722 (深橙)
- 特征拼接: #E1BEE7 (浅紫)
- TriU网络: #BCAAA4 (棕色)
- 通道压缩: #C8E6C9 (浅绿), #A5D6A7 (中绿)
- 路径A: #FFE0B2 (浅橙黄), #FFCC02 (金黄)
- 路径B: #D1C4E9 (浅紫), #B39DDB (中紫), #9575CD (深紫)
- 输出融合: #FFCDD2 (浅红), #F44336 (深红)
- 创新标注: #FFF8E1 (浅黄), #F3E5F5 (淡紫), #FFEBEE (淡红)

绘制要点
========
1. 保持水平流程的清晰性
2. 突出双路径的并行关系
3. 用不同粗细的箭头表示数据流的重要程度
4. 确保所有张量形状标注清晰可见
5. 创新点标注要醒目但不抢夺主体
6. 使用emoji (💡) 增加视觉吸引力

导入DrawIO使用说明
================
1. 访问 https://app.diagrams.net/ 或使用桌面版
2. 打开 StockPredict_DataFlow.drawio 文件
3. 可以编辑和调整布局
4. 导出为Visio格式 (.vsdx) 或其他格式

技术细节说明
===========
- 张量形状变化: (N,T,F) → (N,T_cat,D_out) → (N,T_cat) → (N,1)
- 双路径设计体现了时序信息和截面信息的分离与融合
- 稀疏注意力机制大大降低了计算复杂度
- TriU网络是一个重要的技术创新点
"""

    with open("DataFlow_Guide.txt", "w", encoding="utf-8") as f:
        f.write(guide_text)
    
    print("✅ 数据流绘制指南创建成功 - DataFlow_Guide.txt")

def create_dataflow_vsdx():
    """创建数据流的基础VSDX文件"""
    try:
        import zipfile
        import io
        
        # 创建内存中的ZIP文件
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # [Content_Types].xml
            content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
    <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
    <Default Extension="xml" ContentType="application/xml"/>
    <Override PartName="/visio/document.xml" ContentType="application/vnd.ms-visio.drawing.main+xml"/>
    <Override PartName="/visio/pages/page1.xml" ContentType="application/vnd.ms-visio.page+xml"/>
</Types>'''
            zf.writestr('[Content_Types].xml', content_types)
            
            # _rels/.rels  
            rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/document" Target="visio/document.xml"/>
</Relationships>'''
            zf.writestr('_rels/.rels', rels)
            
            # visio/document.xml
            document = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<VisioDocument xmlns="http://schemas.microsoft.com/office/visio/2012/main">
    <DocumentProperties>
        <Title>StockPredict 数据流详解</Title>
        <Creator>Python Script</Creator>
    </DocumentProperties>
    <Pages>
        <Page ID="1" Name="Data Flow">
            <PageSheet>
                <PageProps>
                    <PageWidth>16</PageWidth>
                    <PageHeight>12</PageHeight>
                </PageProps>
            </PageSheet>
        </Page>
    </Pages>
</VisioDocument>'''
            zf.writestr('visio/document.xml', document)
            
            # visio/_rels/document.xml.rels
            doc_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
    <Relationship Id="rId1" Type="http://schemas.microsoft.com/visio/2010/relationships/page" Target="pages/page1.xml"/>
</Relationships>'''
            zf.writestr('visio/_rels/document.xml.rels', doc_rels)
            
            # visio/pages/page1.xml
            page1 = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<PageContents xmlns="http://schemas.microsoft.com/office/visio/2012/main">
    <Shapes>
        <Shape ID="1" Type="Shape">
            <Text>StockPredict 模型数据流详解</Text>
            <XForm>
                <PinX>8</PinX>
                <PinY>11</PinY>
                <Width>6</Width>
                <Height>0.5</Height>
            </XForm>
        </Shape>
        <Shape ID="2" Type="Shape">
            <Text>请参考绘制指南创建完整的数据流图</Text>
            <XForm>  
                <PinX>8</PinX>
                <PinY>6</PinY>
                <Width>5</Width>
                <Height>1</Height>
            </XForm>
        </Shape>
    </Shapes>
</PageContents>'''
            zf.writestr('visio/pages/page1.xml', page1)
        
        buffer.seek(0)
        vsdx_content = buffer.read()
        
        # 写入文件
        with open("StockPredict_DataFlow.vsdx", "wb") as f:
            f.write(vsdx_content)
        
        print("✅ 数据流VSDX文件创建成功 - StockPredict_DataFlow.vsdx")
        
    except Exception as e:
        print(f"创建VSDX文件时发生错误: {e}")

if __name__ == "__main__":
    create_dataflow_diagrams() 