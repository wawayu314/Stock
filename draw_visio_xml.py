import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def create_visio_xml_diagrams():
    """创建Visio兼容的XML格式神经网络架构图"""
    
    print("正在创建Visio兼容的XML格式图表...")
    
    try:
        # 创建DrawIO格式的XML，可以导入到Visio
        create_drawio_xml()
        
        # 同时生成简化的文本描述文件，便于在Visio中手动绘制
        create_visio_text_guide()
        
        # 创建基础VSDX文件
        create_simple_vsdx()
        
        print("\n✅ Visio兼容文件创建成功！")
        print("生成的文件:")
        print("1. StockPredict_Architecture.drawio - 可导入DrawIO，然后导出为Visio格式")
        print("2. Visio_Drawing_Guide.txt - Visio手动绘制指南")
        print("3. StockPredict_Architecture.vsdx - 可直接在Visio中打开的文件")
        
    except Exception as e:
        print(f"创建文件时发生错误: {e}")
        import traceback
        traceback.print_exc()

def create_drawio_xml():
    """创建DrawIO格式的XML文件，可以导入到Visio"""
    
    print("🔄 正在创建DrawIO格式文件...")
    
    # DrawIO格式的XML，包含完整的图表定义
    drawio_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-01-01T00:00:00.000Z" agent="Python Script" etag="example" version="24.0.0">
  <diagram name="Main Framework" id="main">
    <mxGraphModel dx="1422" dy="854" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <!-- 标题 -->
        <mxCell id="title" value="StockPredict Neural Network Architecture" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=18;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="350" y="30" width="400" height="30" as="geometry" />
        </mxCell>
        
        <!-- 输入层 -->
        <mxCell id="stock1" value="Stock1&#xa;(T, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1976D2;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="80" y="120" width="80" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="stock2" value="Stock2&#xa;(T, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1976D2;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="200" y="120" width="80" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="stock3" value="Stock3&#xa;(T, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1976D2;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="320" y="120" width="80" height="60" as="geometry" />
        </mxCell>
        
        <mxCell id="ellipsis1" value="..." style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=16;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="430" y="135" width="30" height="30" as="geometry" />
        </mxCell>
        
        <!-- 时间处理层 -->
        <mxCell id="temporal1" value="MultiScale&#xa;Temporal&#xa;Fusion" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#2196F3;strokeColor=#1976D2;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="80" y="250" width="80" height="70" as="geometry" />
        </mxCell>
        
        <mxCell id="temporal2" value="MultiScale&#xa;Temporal&#xa;Fusion" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#2196F3;strokeColor=#1976D2;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="200" y="250" width="80" height="70" as="geometry" />
        </mxCell>
        
        <mxCell id="temporal3" value="MultiScale&#xa;Temporal&#xa;Fusion" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#2196F3;strokeColor=#1976D2;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="320" y="250" width="80" height="70" as="geometry" />
        </mxCell>
        
        <!-- 横截面处理层 -->
        <mxCell id="attention" value="Stock Attention Mixer&#xa;(Industry-Sparse Attention)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#4CAF50;strokeColor=#388E3C;fontSize=12;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="550" y="250" width="200" height="70" as="geometry" />
        </mxCell>
        
        <!-- 输出层 -->
        <mxCell id="fc1" value="FC" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#607D8B;strokeColor=#455A64;fontSize=11;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="570" y="420" width="40" height="40" as="geometry" />
        </mxCell>
        
        <mxCell id="fc2" value="FC" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#607D8B;strokeColor=#455A64;fontSize=11;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="630" y="420" width="40" height="40" as="geometry" />
        </mxCell>
        
        <mxCell id="fc3" value="FC" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#607D8B;strokeColor=#455A64;fontSize=11;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="690" y="420" width="40" height="40" as="geometry" />
        </mxCell>
        
        <mxCell id="pred1" value="P1" style="ellipse;whiteSpace=wrap;html=1;fillColor=#FF9800;strokeColor=#F57C00;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="575" y="520" width="30" height="30" as="geometry" />
        </mxCell>
        
        <mxCell id="pred2" value="P2" style="ellipse;whiteSpace=wrap;html=1;fillColor=#FF9800;strokeColor=#F57C00;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="635" y="520" width="30" height="30" as="geometry" />
        </mxCell>
        
        <mxCell id="pred3" value="P3" style="ellipse;whiteSpace=wrap;html=1;fillColor=#FF9800;strokeColor=#F57C00;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="695" y="520" width="30" height="30" as="geometry" />
        </mxCell>
        
        <!-- 连接箭头 -->
        <mxCell id="arrow1" value="" style="endArrow=classic;html=1;rounded=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="stock1" target="temporal1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="120" y="200" as="sourcePoint" />
            <mxPoint x="120" y="240" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow2" value="" style="endArrow=classic;html=1;rounded=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="stock2" target="temporal2">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="240" y="200" as="sourcePoint" />
            <mxPoint x="240" y="240" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow3" value="" style="endArrow=classic;html=1;rounded=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;entryX=0.5;entryY=0;entryDx=0;entryDy=0;" edge="1" parent="1" source="stock3" target="temporal3">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="360" y="200" as="sourcePoint" />
            <mxPoint x="360" y="240" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow4" value="" style="endArrow=classic;html=1;rounded=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="temporal1" target="attention">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="160" y="285" as="sourcePoint" />
            <mxPoint x="550" y="285" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow5" value="" style="endArrow=classic;html=1;rounded=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="temporal2" target="attention">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="280" y="285" as="sourcePoint" />
            <mxPoint x="550" y="285" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <mxCell id="arrow6" value="" style="endArrow=classic;html=1;rounded=0;exitX=1;exitY=0.5;exitDx=0;exitDy=0;entryX=0;entryY=0.5;entryDx=0;entryDy=0;" edge="1" parent="1" source="temporal3" target="attention">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="400" y="285" as="sourcePoint" />
            <mxPoint x="550" y="285" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        
        <!-- 数据流标注 -->
        <mxCell id="input_label" value="Input: (N, T, F)&#xa;N stocks, T timesteps, F features" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#9E9E9E;fontSize=9;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="80" y="70" width="150" height="40" as="geometry" />
        </mxCell>
        
        <mxCell id="hidden_label" value="Hidden: (N, h)&#xa;N stocks, h features" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#9E9E9E;fontSize=9;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="550" y="370" width="120" height="40" as="geometry" />
        </mxCell>
        
        <mxCell id="output_label" value="Output: (N, 1)&#xa;Predictions for N stocks" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#F5F5F5;strokeColor=#9E9E9E;fontSize=9;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="570" y="570" width="120" height="40" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
  
  <diagram name="MultiScale Temporal Fusion" id="temporal">
    <mxGraphModel dx="1422" dy="854" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <mxCell id="title2" value="Figure 2: MultiScale Temporal Fusion Module" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=16;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="350" y="30" width="350" height="30" as="geometry" />
        </mxCell>
        
        <!-- 输入 -->
        <mxCell id="input_tf" value="Single Stock&#xa;Input&#xa;(T, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1976D2;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="50" y="120" width="80" height="70" as="geometry" />
        </mxCell>
        
        <!-- 多尺度路径 -->
        <mxCell id="orig_scale" value="Original Scale&#xa;(T, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#2196F3;strokeColor=#1976D2;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="200" y="80" width="100" height="50" as="geometry" />
        </mxCell>
        
        <mxCell id="conv1" value="Conv1D (k=2, s=2)&#xa;(T/2, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FF5722;strokeColor=#D84315;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="200" y="150" width="100" height="50" as="geometry" />
        </mxCell>
        
        <mxCell id="conv2" value="Conv1D (k=4, s=4)&#xa;(T/4, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FF5722;strokeColor=#D84315;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="200" y="220" width="100" height="50" as="geometry" />
        </mxCell>
        
        <!-- 拼接 -->
        <mxCell id="concat" value="Concatenation&#xa;(T+T/2+T/4, F)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#9C27B0;strokeColor=#7B1FA2;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="380" y="150" width="100" height="80" as="geometry" />
        </mxCell>
        
        <!-- TriU网络 -->
        <mxCell id="triu" value="TriU&#xa;Network" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#795548;strokeColor=#5D4037;fontSize=11;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="380" y="280" width="100" height="60" as="geometry" />
        </mxCell>
        
        <!-- 全连接 -->
        <mxCell id="fc_tf" value="Fully&#xa;Connected" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#607D8B;strokeColor=#455A64;fontSize=11;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="380" y="380" width="100" height="60" as="geometry" />
        </mxCell>
        
        <!-- 输出 -->
        <mxCell id="output_tf" value="Feature&#xa;Vector h" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FF9800;strokeColor=#F57C00;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="550" y="395" width="80" height="50" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
  
  <diagram name="Stock Attention Mixer" id="attention">
    <mxGraphModel dx="1422" dy="854" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        
        <mxCell id="title3" value="Figure 3: Stock Attention Mixer Module" style="text;html=1;align=center;verticalAlign=middle;resizable=0;points=[];autosize=1;strokeColor=none;fillColor=none;fontSize=16;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="350" y="30" width="320" height="30" as="geometry" />
        </mxCell>
        
        <!-- 输入 -->
        <mxCell id="input_sam" value="Input Matrix&#xa;(N, h)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E3F2FD;strokeColor=#1976D2;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="50" y="100" width="100" height="50" as="geometry" />
        </mxCell>
        
        <!-- LayerNorm -->
        <mxCell id="norm1" value="LayerNorm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFC107;strokeColor=#FF8F00;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="200" y="100" width="80" height="50" as="geometry" />
        </mxCell>
        
        <!-- Linear QKV -->
        <mxCell id="linear1" value="Linear&#xa;(Q,K,V)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#607D8B;strokeColor=#455A64;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="320" y="100" width="80" height="50" as="geometry" />
        </mxCell>
        
        <!-- Industry Sparse Attention -->
        <mxCell id="sparse_att" value="Industry&#xa;Sparse&#xa;Attention" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#9C27B0;strokeColor=#7B1FA2;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="320" y="200" width="80" height="80" as="geometry" />
        </mxCell>
        
        <!-- Linear Output -->
        <mxCell id="linear2" value="Linear&#xa;(Output)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#607D8B;strokeColor=#455A64;fontSize=10;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="320" y="320" width="80" height="50" as="geometry" />
        </mxCell>
        
        <!-- Add & Norm 1 -->
        <mxCell id="addnorm1" value="Add &amp;&#xa;Norm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFC107;strokeColor=#FF8F00;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="200" y="320" width="80" height="50" as="geometry" />
        </mxCell>
        
        <!-- FFN -->
        <mxCell id="ffn" value="FFN: Linear → Hardswish → Linear" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#607D8B;strokeColor=#455A64;fontSize=9;fontStyle=1;fontColor=#FFFFFF" vertex="1" parent="1">
          <mxGeometry x="150" y="420" width="180" height="60" as="geometry" />
        </mxCell>
        
        <!-- Add & Norm 2 -->
        <mxCell id="addnorm2" value="Add &amp;&#xa;Norm" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FFC107;strokeColor=#FF8F00;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="200" y="520" width="80" height="50" as="geometry" />
        </mxCell>
        
        <!-- 输出 -->
        <mxCell id="output_sam" value="Output Matrix&#xa;(N, h)" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#FF9800;strokeColor=#F57C00;fontSize=11;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="50" y="520" width="100" height="50" as="geometry" />
        </mxCell>
        
        <!-- 稀疏注意力掩码 -->
        <mxCell id="mask" value="稀疏注意力掩码:&#xa;1: 行业内注意力&#xa;0: 行业间注意力" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#E6F3FF;strokeColor=#2196F3;fontSize=10;fontStyle=1" vertex="1" parent="1">
          <mxGeometry x="500" y="200" width="120" height="80" as="geometry" />
        </mxCell>
        
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
    
    try:
        # 保存DrawIO格式文件
        with open("StockPredict_Architecture.drawio", "w", encoding="utf-8") as f:
            f.write(drawio_xml)
        
        print("✅ DrawIO格式文件创建成功 - StockPredict_Architecture.drawio")
        
    except Exception as e:
        print(f"创建DrawIO文件时发生错误: {e}")
        raise e

def create_visio_text_guide():
    """创建Visio手动绘制指南"""
    
    guide_text = """
StockPredict 神经网络架构 - Visio 绘制指南
===========================================

本指南帮助您在Microsoft Visio中手动绘制StockPredict神经网络架构图。

第1页：主框架图 (Main Framework)
==============================

1. 标题：
   - 文本框："StockPredict Neural Network Architecture"
   - 字体：18号，加粗，居中

2. 输入层（浅蓝色矩形，#E3F2FD）：
   - Stock1 (T, F) - 位置：左上
   - Stock2 (T, F) - 位置：中上  
   - Stock3 (T, F) - 位置：右上
   - "..." - 省略号
   - "N Stocks" - 标签

3. 时间处理层（蓝色矩形，#2196F3，白色文字）：
   - MultiScale Temporal Fusion × 3
   - 垂直箭头连接输入层
   - 输出标签：h₁, h₂, h₃

4. 横截面处理层（绿色矩形，#4CAF50，白色文字）：
   - Stock Attention Mixer (Industry-Sparse Attention)
   - 水平箭头从时间处理层连入

5. 输出层：
   - FC层（灰蓝色矩形，#607D8B，白色文字）× 3
   - 预测输出（橙色圆圈，#FF9800）：P1, P2, P3
   - 垂直箭头连接

6. 数据流标注（浅灰色矩形，#F5F5F5）：
   - Input: (N, T, F) - N stocks, T timesteps, F features
   - Hidden: (N, h) - N stocks, h features  
   - Output: (N, 1) - Predictions for N stocks

7. 指向说明（浅黄色矩形，#FFFFCC，红色文字）：
   - "See Figure 2 for MultiScale Temporal Fusion details"
   - "See Figure 3 for Stock Attention Mixer details"

第2页：MultiScale Temporal Fusion 详细图
=====================================

1. 标题：
   - "Figure 2: MultiScale Temporal Fusion Module"
   - 副标题："Multi-Scale Feature Extraction"

2. 输入（浅蓝色矩形）：
   - "Single Stock Input (T, F)"

3. 多尺度路径：
   - 路径1（蓝色）：Original Scale (T, F)
   - 路径2（红橙色）：Conv1D (k=2, s=2) (T/2, F)
   - 路径3（红橙色）：Conv1D (k=4, s=4) (T/4, F)

4. 特征拼接（紫色矩形）：
   - "Concatenation (T+T/2+T/4, F)"

5. 处理网络：
   - TriU Network（棕色矩形）
   - Fully Connected（灰蓝色矩形）

6. 输出（橙色矩形）：
   - "Feature Vector h"

7. 核心创新说明（红色文字）：
   - "核心创新: 多尺度时间特征提取 + TriU三角网络"
   - "有效捕获不同时间尺度的股价模式"

第3页：Stock Attention Mixer 详细图
=================================

1. 标题：
   - "Figure 3: Stock Attention Mixer Module"

2. 组件序列（从上到下）：
   - Input Matrix (N, h) - 浅蓝色
   - LayerNorm - 琥珀色（#FFC107）
   - Linear (Q,K,V) - 灰蓝色，白色文字
   - Industry Sparse Attention - 紫色，白色文字
   - Linear (Output) - 灰蓝色，白色文字
   - Add & Norm - 琥珀色（残差连接1）
   - FFN: Linear → Hardswish → Linear - 灰蓝色，白色文字
   - Add & Norm - 琥珀色（残差连接2）
   - Output Matrix (N, h) - 橙色

3. 连接：
   - 垂直箭头连接主要路径
   - 虚线表示跳跃连接（Skip Connections）

4. 稀疏注意力掩码示例（浅蓝色矩形）：
   - "稀疏注意力掩码示例:"
   - "1: 行业内注意力"
   - "0: 行业间注意力"

5. 核心创新说明（红色文字）：
   - "核心创新: 基于行业的稀疏注意力机制"
   - "复杂度从 O(N²) 降低到 O(k²×m)，其中 k=每行业平均股票数，m=行业数"

颜色代码
========
- 浅蓝色（输入）：#E3F2FD
- 蓝色（时间处理）：#2196F3
- 绿色（横截面处理）：#4CAF50
- 橙色（输出）：#FF9800
- 紫色（注意力）：#9C27B0
- 红橙色（卷积）：#FF5722
- 灰蓝色（全连接）：#607D8B
- 棕色（TriU网络）：#795548
- 琥珀色（LayerNorm）：#FFC107
- 浅灰色（标注）：#F5F5F5
- 浅黄色（说明）：#FFFFCC

绘制技巧
========
1. 使用Visio的"流程图"模板开始
2. 先绘制主要组件（矩形和圆形）
3. 添加连接线和箭头
4. 设置颜色和文字格式
5. 添加文本标注和说明
6. 调整布局确保清晰美观

导入DrawIO文件
=============
1. 下载并安装draw.io桌面版或使用在线版
2. 打开 StockPredict_Architecture.drawio 文件
3. 编辑和调整图表
4. 导出为Visio格式 (.vsdx)
"""
    
    # 保存指南文件
    with open("Visio_Drawing_Guide.txt", "w", encoding="utf-8") as f:
        f.write(guide_text)
    
    print("✅ Visio绘制指南创建成功 - Visio_Drawing_Guide.txt")

def create_simple_vsdx():
    """创建一个简单的VSDX格式文件"""
    try:
        # 创建最基本的VSDX文件结构
        vsdx_content = create_basic_vsdx_structure()
        
        # 写入文件
        with open("StockPredict_Architecture.vsdx", "wb") as f:
            f.write(vsdx_content)
        
        print("✅ 基础VSDX文件创建成功 - StockPredict_Architecture.vsdx")
        
    except Exception as e:
        print(f"创建VSDX文件时发生错误: {e}")

def create_basic_vsdx_structure():
    """创建基本的VSDX文件结构"""
    import zipfile
    import io
    
    # 创建一个内存中的ZIP文件
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # 添加基本的VSDX文件结构
        
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
        <Title>StockPredict Neural Network Architecture</Title>
        <Creator>Python Script</Creator>
    </DocumentProperties>
    <Pages>
        <Page ID="1" Name="Main Framework">
            <PageSheet>
                <PageProps>
                    <PageWidth>11</PageWidth>
                    <PageHeight>8.5</PageHeight>
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
            <Text>StockPredict Neural Network Architecture</Text>
            <XForm>
                <PinX>5.5</PinX>
                <PinY>7.5</PinY>
                <Width>4</Width>
                <Height>0.5</Height>
            </XForm>
        </Shape>
        <Shape ID="2" Type="Shape">
            <Text>请使用Visio打开此文件并参考绘制指南添加详细内容</Text>
            <XForm>
                <PinX>5.5</PinX>
                <PinY>4</PinY>
                <Width>6</Width>
                <Height>1</Height>
            </XForm>
        </Shape>
    </Shapes>
</PageContents>'''
        zf.writestr('visio/pages/page1.xml', page1)
    
    buffer.seek(0)
    return buffer.read()

if __name__ == "__main__":
    create_visio_xml_diagrams()
    create_simple_vsdx() 